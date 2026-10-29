#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API layer: one turn of conversation, including bounded local tool rounds.

This module owns the boundary between the app and the model:

  _api_request            one HTTP round-trip, nothing else
  _parse_agent_envelope   strict validation of the model's JSON reply
  call_api                the turn loop: request -> parse -> maybe run a tool -> repeat
  call_summary_api        the single-shot request used by /memory compact
  _add_usage_totals       token accounting shared by both callers

The model is treated as an untrusted JSON generator. It may only emit
``{"type":"output",...}`` or ``{"type":"tool_call",...}``; everything else is a
protocol error that is reported back to the model once before giving up.

Dependency direction
--------------------
Host services (command loading, the tool log, the dispatcher) live in
``agent_core`` and ``agent_dispatcher``. Importing them from here would create a
cycle, so they are *injected* at startup:

    api.set_host(load_commands=..., command_context=..., run_tool=..., ...)

Until a host is registered every hook falls back to a stub that fails loudly, so
a misconfigured import chain surfaces as a clear error instead of an
``AttributeError`` deep inside a turn. ``agent_core.register_api_host()`` does
the wiring and is called once at import time.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import build_endpoint
from translator import DEFAULT_LANG, Translator

REQUEST_TIMEOUT = 120
MAX_HISTORY_MESSAGES = 10
MAX_TOOL_ROUNDS = 8
MAX_PROTOCOL_REPAIRS = 1

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Rebound by the entry point; module-level so protocol text follows the UI language.
t = Translator(DEFAULT_LANG)


class ProtocolError(ValueError):
    """The model's reply did not satisfy the local Agent protocol."""

    def __init__(self, code, **params):
        super().__init__(code)
        self.code = code
        self.params = params

    def render(self, translator=None):
        """Human-readable text in the active language, for the model and the user."""
        catalog = translator or t
        return catalog(self.code, **self.params)


class ApiError(RuntimeError):
    """The HTTP layer returned a response we cannot use."""

    def __init__(self, code, **params):
        super().__init__(code)
        self.code = code
        self.params = params

    def render(self, translator=None):
        catalog = translator or t
        return catalog(self.code, **self.params)


# --------------------------------------------------------------------------
# Host services, injected by agent_core at import time.
# --------------------------------------------------------------------------

def _missing_host(name):
    def stub(*args, **kwargs):
        raise RuntimeError(
            "api host service %r is not registered; call api.set_host() first" % name
        )
    return stub


_HOST = {
    "load_commands": _missing_host("load_commands"),
    "command_context": _missing_host("command_context"),
    "run_tool": _missing_host("run_tool"),
    "log_tool_state": _missing_host("log_tool_state"),
    "save_tool_output": _missing_host("save_tool_output"),
    "timestamp": _missing_host("timestamp"),
}


def set_host(**services):
    """Register the callbacks this module needs. Unknown names are ignored."""
    for name, fn in services.items():
        if name in _HOST and callable(fn):
            _HOST[name] = fn


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

def _api_request(config, messages):
    """POST one chat-completion request and return (content, usage)."""
    payload = {
        "model": config["model"],
        "messages": messages,
        "stream": False,
    }
    request = urllib.request.Request(
        build_endpoint(config["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config["api_key"],
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("api_error_no_choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ApiError("api_error_no_content")
    usage = result.get("usage", {})
    return content, usage if isinstance(usage, dict) else {}


# --------------------------------------------------------------------------
# Protocol validation
# --------------------------------------------------------------------------

def _parse_agent_envelope(raw):
    """Parse exactly one JSON response envelope; never execute arbitrary text."""
    candidate = raw.strip()
    # Be tolerant of a single accidental Markdown fence, but prompt the model not to use it.
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        envelope = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            "protocol_error_invalid_json", lineno=exc.lineno, colno=exc.colno
        ) from exc
    if not isinstance(envelope, dict):
        raise ProtocolError("protocol_error_not_object")
    kind = envelope.get("type")
    if kind == "output":
        if set(envelope) != {"type", "text"} or not isinstance(envelope.get("text"), str):
            raise ProtocolError("protocol_error_output_shape")
        return envelope
    if kind == "tool_call":
        allowed = {"type", "command", "arguments", "reason"}
        if set(envelope) - allowed:
            raise ProtocolError("protocol_error_tool_unknown_field")
        if not isinstance(envelope.get("command"), str) or not envelope["command"]:
            raise ProtocolError("protocol_error_tool_missing_command")
        if not isinstance(envelope.get("arguments"), dict):
            raise ProtocolError("protocol_error_tool_arguments_not_object")
        if "reason" in envelope and not isinstance(envelope["reason"], str):
            raise ProtocolError("protocol_error_tool_reason_not_string")
        return envelope
    raise ProtocolError("protocol_error_unknown_type")


def _add_usage_totals(total, usage):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key, 0)
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + int(value)
    total["_requests"] = total.get("_requests", 0) + 1


# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------

def _build_system_messages(memory, command_context, reply_language_hint=None):
    """Assemble the leading system block shared by every request of a turn.

    Index 0 is the user's own system prompt, index 1 the protocol + command
    contract (rewritten on every request), followed by the output contract and
    the memory background. The reply-language hint, when present, is inserted
    right after index 0 so it applies to everything that follows.
    """
    messages = [{"role": "system", "content": memory.get("system_prompt") or DEFAULT_SYSTEM_PROMPT}]
    if reply_language_hint:
        messages.append({"role": "system", "content": reply_language_hint})
    messages.append({"role": "system", "content": command_context})
    messages.append({"role": "system", "content": t("protocol_output_contract")})
    memory_context = {
        "active_memory_version": memory.get("version"),
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
        "summary": memory.get("summary", ""),
    }
    messages.append({
        "role": "system",
        "content": t("protocol_memory_context", context=json.dumps(memory_context, ensure_ascii=False, indent=2)),
    })
    if memory.get("summary"):
        messages.append({
            "role": "system",
            "content": t("protocol_memory_summary", summary=memory["summary"]),
        })
    return messages


# --------------------------------------------------------------------------
# The turn loop
# --------------------------------------------------------------------------

def call_api(config, memory, current_user_text, reply_language_hint=None):
    """Run one user turn, including bounded local tool calls, without persisting traces."""
    # Build a fresh context for every user turn. The command file is re-read for every API call.
    command_data = _HOST["load_commands"]()
    messages = _build_system_messages(
        memory, _HOST["command_context"](command_data), reply_language_hint
    )
    # Locate the command-contract message once; it is the one rebuilt each request.
    context_index = 2 if reply_language_hint else 1
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in memory["messages"][-MAX_HISTORY_MESSAGES:]
    )

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "_requests": 0}
    tool_rounds = 0
    protocol_repairs = 0

    while True:
        # Refresh the command definitions on each API request so changes apply immediately.
        latest_command_data = _HOST["load_commands"]()
        messages[context_index] = {
            "role": "system",
            "content": _HOST["command_context"](latest_command_data),
        }
        raw, usage = _api_request(config, messages)
        _add_usage_totals(usage_total, usage)
        try:
            envelope = _parse_agent_envelope(raw)
        except ProtocolError as exc:
            if protocol_repairs >= MAX_PROTOCOL_REPAIRS:
                raise ProtocolError(
                    "protocol_error_unrecoverable", detail=exc.render()
                ) from exc
            protocol_repairs += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": t("protocol_repair_prompt", detail=exc.render()),
            })
            continue

        if envelope["type"] == "output":
            usage_total["requests"] = usage_total.pop("_requests", 0)
            return envelope["text"], usage_total

        if tool_rounds >= MAX_TOOL_ROUNDS:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "type": "tool_result",
                    "status": "error",
                    "message": t("protocol_tool_limit_reached", limit=MAX_TOOL_ROUNDS),
                }, ensure_ascii=False),
            })
            # One final model request is permitted only to summarize the results.
            final_raw, final_usage = _api_request(config, messages)
            _add_usage_totals(usage_total, final_usage)
            try:
                final_envelope = _parse_agent_envelope(final_raw)
            except ProtocolError as exc:
                raise ProtocolError(
                    "protocol_error_after_tool_limit", detail=exc.render()
                ) from exc
            if final_envelope.get("type") != "output":
                raise ProtocolError("protocol_error_tool_limit_not_honoured")
            usage_total["requests"] = usage_total.pop("_requests", 0)
            return final_envelope["text"], usage_total

        tool_rounds += 1
        messages.append({"role": "assistant", "content": raw})
        # Filesystem-safe call id: timestamp() is ISO-8601 and contains ':', which
        # Windows rejects in a file name (the id doubles as a tool_output file name).
        call_id = _HOST["timestamp"]().replace(":", "").replace("+", "p") + "-" + str(tool_rounds)
        command_name = envelope.get("command", "")
        print(t("agent_request_requesting_local_command_command",
                command_name=command_name, tool_rounds=tool_rounds,
                MAX_TOOL_ROUNDS=MAX_TOOL_ROUNDS, call_id=call_id))
        # Durable request record is written before any confirmation prompt or execution.
        _HOST["log_tool_state"](
            call_id, state="requested", command=command_name,
            arguments=envelope.get("arguments", {}), reason=envelope.get("reason", ""),
            requested_at=_HOST["timestamp"](),
        )
        _HOST["log_tool_state"](call_id, state="running_or_handling", command=command_name)
        try:
            result = _HOST["run_tool"](envelope)
            if not isinstance(result, dict):
                result = {"status": "error", "message": t("tool_result_not_object")}
        except Exception as exc:
            result = {
                "status": "error",
                "command": command_name,
                "message": t("tool_dispatch_failed", exc_type=type(exc).__name__, exc=exc),
            }
        output_meta = _HOST["save_tool_output"](call_id, result)
        _HOST["log_tool_state"](
            call_id,
            state="completed" if result.get("status") in ("ok", "success") else "returned",
            command=command_name, result_status=result.get("status", "unknown"),
            exit_code=result.get("exit_code"),
            result_message=str(result.get("message", ""))[:1000],
            result_received_at=_HOST["timestamp"](), **(output_meta or {}))
        # Tool result stays out of persistent chat memory. It is only appended to this turn's API context.
        messages.append({
            "role": "user",
            "content": t("protocol_tool_result", result=json.dumps(
                {"type": "tool_result", "result": result}, ensure_ascii=False)),
        })
        _HOST["log_tool_state"](
            call_id, state="result_added_to_api_context", command=command_name,
            result_status=result.get("status", "unknown"), timestamp=_HOST["timestamp"]())


# --------------------------------------------------------------------------
# Compaction request
# --------------------------------------------------------------------------

def call_summary_api(config, memory, path):
    """Ask the configured model to create a compact, factual memory summary."""
    messages = [
        {
            "role": "system",
            "content": t("summary_system_prompt"),
        },
        {
            "role": "user",
            "content": t(
                "summary_user_intro",
                branch=path.stem,
                summary=memory.get("summary", "") or t("summary_none"),
            ) + "\n\n".join(
                f"[{item['role']}]\n{item['content']}"
                for item in memory["messages"]
            ),
        },
    ]
    payload = {"model": config["model"], "messages": messages, "stream": False}
    request = urllib.request.Request(
        build_endpoint(config["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["api_key"]},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("api_error_no_choices")
    summary = choices[0].get("message", {}).get("content")
    if not isinstance(summary, str) or not summary.strip():
        raise ApiError("api_error_empty_summary")
    return summary.strip(), result.get("usage", {}), messages
