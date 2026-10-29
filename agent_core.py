#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek local-memory chat core (v10).

Everything that is not terminal I/O lives here: memories, the command contract,
tool-call plumbing, compaction and the request/reply pair. repl.py owns the
interactive shell on top of it.

User-facing strings are looked up through translator.py and i18n/<lang>.json.
chat_v10.py is the only entry point: the UI language comes from --lang, then the
value persisted in config.json, then the fallback in translator.py. /settings can
switch language and model at runtime; both are saved back to config.json.

Standard-library only, except the optional `tokenizers` package used by the
tokenizer/ package for exact token counting (falls back to a byte estimate when
it is not installed).

Layout:
  chat_v10.py                 entry point
  repl.py                     interactive shell: argument parsing, banner, commands
  agent_core.py               this module: chat logic with no terminal loop
  api.py                      request/response transport and protocol prompts
  ui/                         terminal rendering and input primitives
  models.py                   model catalogue and slot layout
  agent_dispatcher.py         validates and runs local tool calls
  command_library_bridge.py   JSON/Base64 envelope around a backend
  bridges/                    python (default), powershell, posix backends
  tokenizer/                  token_counter + bundled vocabulary
  translator.py, i18n/        UI strings per language
  config.json
  memories/default.json (created automatically; multiple named memories)
  memories/commands.json (prompt-visible command contract, reloaded for every API request)
  memories/commands.impl.json (implementation details, never sent to the model)
  memory.json (legacy file, imported into default.json if needed)
  library/    (working area for tool calls; the model reads files here via cat/head/grep)

Commands (interpreted by repl.py):
  /files           List files in library/
  /memory          List/select memory profiles
  /memory add NAME Create and switch to a new memory profile
  /memory NAME     Switch to an existing memory profile
  /memory compact  Ask AI to summarize history; confirm before applying
  /pwd             Show current memory branch and file path
  /clear           Clear chat history after confirmation
  /stats           Show cumulative token usage
  /system          View/change the system prompt
  /settings        Switch language or model (persisted to config.json)
  /help            Show help
  /exit            Exit
"""

from ui import common as ui_common
from ui import memory_ui


def ask_yes_no(action):
    """Ask a consistent yes/no question. Default is No; accepts y/yes and n/no case-insensitively.

    Kept as the app-core-side entry point so existing call sites stay unchanged.
    The prompt loop itself lives in ui.common, where it is shared with anything
    else that needs to read a confirmation.
    """
    prompt = t("do_you_want_to_action_y", action=action)
    return ui_common.ask_choice(
        prompt,
        _YES_NO_ACCEPT,
        default=False,
        on_invalid=lambda: print(t("please_enter_y_yes_or_n")),
    )


def _build_yes_no_accept():
    """Map every accepted answer onto its boolean result."""
    accept = {answer: True for answer in ui_common.YES}
    accept.update({answer: False for answer in ui_common.NO})
    return accept


_YES_NO_ACCEPT = _build_yes_no_accept()

import hashlib
import json
import os
import re
import urllib.error
from datetime import datetime

from tokenizer import token_counter, try_load_token_counter as _try_load_token_counter

from translator import Translator, DEFAULT_LANG

# Aliased because 'memory' is a ubiquitous local variable name in this module.
import memory as memory_store
import api
from api import (
    ApiError,
    ProtocolError,
    _add_usage_totals,
    _api_request,
    _parse_agent_envelope,
    call_api,
    call_summary_api,
)
from config import save_json
from paths import (
    BASE_DIR,
    COMMANDS_FILE,
    LIBRARY_DIR,
    MEMORIES_DIR,
    TOOL_LOG_FILE,
    TOOL_OUTPUT_DIR,
)

_TOKEN_COUNTER = _try_load_token_counter()

# Rebound by repl.main(); module-level so every helper can render UI strings.
t = Translator(DEFAULT_LANG)

# Both files live in memories/ but are not memory branches.
RESERVED_MEMORY_FILES = {"commands.json", "commands.impl.json"}
MAX_COMMANDS_FILE_BYTES = 256 * 1024

DEFAULT_SYSTEM_PROMPT = api.DEFAULT_SYSTEM_PROMPT
MAX_HISTORY_MESSAGES = api.MAX_HISTORY_MESSAGES
RECENT_MESSAGES_AFTER_COMPACT = 10
REQUEST_TIMEOUT = api.REQUEST_TIMEOUT
MAX_TOOL_ROUNDS = api.MAX_TOOL_ROUNDS
MAX_PROTOCOL_REPAIRS = api.MAX_PROTOCOL_REPAIRS

MAX_TOOL_LOG_LINES = 5000


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def list_memories(active_path):
    paths = memory_store.list_memory_paths()
    print(t("available_memories"))
    if not paths:
        print(t("no_memory_files"))
        return
    for path in paths:
        mark = t("current_marker") if path.resolve() == active_path.resolve() else ""
        try:
            data = memory_store.load_memory_file(path)
            count = len(data["messages"]) if data else 0
            print(t("path_stem_count_messages_mark", arg0=path.stem, count=count, mark=mark))
        except (memory_store.MemoryFormatError, OSError) as exc:
            print(t("path_stem_cannot_read_skipped_will", arg0=path.stem, mark=mark, exc=exc))


def handle_memory_command(command, active_path, memory):
    parts = command.split(maxsplit=2)
    if len(parts) == 1:
        list_memories(active_path)
        print(t("usage_memory_name_or_memory_add"))
        return active_path, memory

    if parts[1].casefold() == "add":
        if len(parts) != 3:
            print(t("usage_memory_add_name"))
            return active_path, memory
        try:
            target, name = memory_store.memory_path(parts[2])
            if target.exists():
                # Existing file is never overwritten by add.
                try:
                    existing = memory_store.load_memory_file(target)
                    if existing is None:
                        print(t("failed_to_create_target_name_already", arg0=target.name))
                        return active_path, memory
                except (memory_store.MemoryFormatError, OSError) as exc:
                    print(t("failed_to_create_exc", exc=exc))
                    return active_path, memory
                print(t("memory_name_already_exists_and_will", name=name))
                if ask_yes_no(t("switch_to_memory_name", name=name)):
                    return target, existing
                return active_path, memory
            fresh = memory_store.new_memory()
            save_json(target, fresh)
            print(t("created_and_switched_to_memory_name", name=name))
            return target, fresh
        except (ValueError, OSError) as exc:
            if isinstance(exc, memory_store.MemoryNameError):
                print(t("failed_to_create_memory_exc",
                        exc=memory_ui.describe_memory_name_error(exc, t)))
            else:
                print(t("failed_to_create_memory_exc", exc=exc))
            return active_path, memory

    requested = parts[1]
    try:
        target, name = memory_store.memory_path(requested)
        if not target.exists():
            print(t("unknown_memory_name_no_switch_or", name=name))
            print(t("first_use_memory_add_name_to"))
            return active_path, memory
        try:
            loaded = memory_store.load_memory_file(target)
        except (RuntimeError, OSError) as exc:
            print(t("failed_to_switch_memory_exc", exc=exc))
            print(t("current_memory_remains_unchanged_original_file"))
            return active_path, memory
        if loaded is None:
            print(t("failed_to_switch_memory_file_is"))
            return active_path, memory
        print(t("switched_to_memory_name_len_loaded", name=name, arg1=len(loaded['messages'])))
        return target, loaded
    except ValueError as exc:
        if isinstance(exc, memory_store.MemoryNameError):
            print(memory_ui.describe_memory_name_error(exc, t))
        else:
            print(t("invalid_memory_name_exc", exc=exc))
        return active_path, memory


def _count_tokens(text):
    """Use the local DeepSeek tokenizer when available, else fall back to a byte estimate."""
    if _TOKEN_COUNTER is not None:
        try:
            return _TOKEN_COUNTER(text)
        except Exception:
            pass
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def estimate_tokens_from_text(text):
    """Token count for the given text (precise when the local tokenizer loads)."""
    return _count_tokens(text)


def compact_memory(config, memory, active_path, keep=RECENT_MESSAGES_AFTER_COMPACT):
    """Preview cost, ask AI, then ask user before archiving and applying summary."""
    if not memory["messages"]:
        print(t("current_memory_has_no_chat_history"))
        return

    source_text = "\n\n".join(
        f"[{item['role']}]\n{item['content']}" for item in memory["messages"]
    )
    source_bytes = len(source_text.encode("utf-8"))
    rough = estimate_tokens_from_text(source_text)
    if keep < 0 or keep > 1000:
        print(t("the_number_of_messages_to_keep"))
        return
    recent_count = min(keep, len(memory["messages"]))
    print(t("memory_compaction_preview"))
    print(t("current_branch_active_path_stem", arg0=active_path.stem))
    print(t("history_len_memory_messages_messages", arg0=len(memory['messages'])))
    print(t("history_text_source_bytes_bytes", source_bytes=source_bytes))
    if _TOKEN_COUNTER is not None:
        print(t("input_tokens_rough_counted_by_the", rough=rough))
    else:
        print(t("rough_input_token_estimate_rough_tokenizer", rough=rough))
    print(t("after_compaction_the_most_recent_recent", recent_count=recent_count))
    print(t("actual_cost_is_based_on_the"))
    if not ask_yes_no(t("send_the_full_current_chat_history")): 
        print(t("cancelled_the_memory_was_not_changed"))
        return

    try:
        summary, usage, _ = call_summary_api(config, memory, active_path)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(t("compaction_request_failed_http_exc_code", arg0=exc.code, details=details))
        print(t("the_original_memory_was_not_changed"))
        return
    except Exception as exc:
        print(t("compaction_request_failed_type_exc_name", arg0=type(exc).__name__, exc=exc))
        print(t("the_original_memory_was_not_changed_2"))
        return

    # The API call has already incurred usage even if the user later declines applying it.
    add_usage(memory, usage)
    try:
        memory_store.save_memory(memory, active_path)
    except OSError as exc:
        print(t("warning_failed_to_save_token_statistics", exc=exc))

    print(t("ai_generated_memory_summary_preview"))
    print(summary)
    if usage:
        print(t("compaction_request_tokens_input_usage_get", arg0=usage.get('prompt_tokens', '?'), arg1=usage.get('completion_tokens', '?'), arg2=usage.get('total_tokens', '?')))

    if not ask_yes_no(t("save_this_summary_and_archive_the")): 
        print(t("cancelled_applying_the_summary_the_current"))
        return

    archive_dir = MEMORIES_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    archive_path = archive_dir / f"{active_path.stem}-{stamp}.json"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / f"{active_path.stem}-{stamp}-{suffix}.json"
        suffix += 1
    archive_data = dict(memory)
    archive_data["messages"] = list(memory["messages"])
    archive_data["archived_at"] = timestamp()
    try:
        save_json(archive_path, archive_data)
        memory["summary"] = summary
        memory["messages"] = memory["messages"][-keep:] if keep else []
        memory_store.save_memory(memory, active_path)
    except Exception as exc:
        print(t("failed_to_save_organization_results_type", arg0=type(exc).__name__, exc=exc))
        print(t("if_an_archive_was_created_it", archive_path=archive_path))
        print(t("please_check_the_file_before_continuing"))
        return
    print(t("memory_compaction_completed"))
    print(t("current_memory_active_path", active_path=active_path))
    print(t("archived_history_archive_path", archive_path=archive_path))
    print(t("raw_messages_retained_len_memory_messages", arg0=len(memory['messages'])))
    print(t("summary_length_len_summary_encode_utf", arg0=len(summary.encode('utf-8'))))


def load_command_definitions():
    """
    Load the current command definitions from memories/commands.json.
    This is intentionally called for every API request so edits take effect
    without restarting the chat program.
    """
    if not COMMANDS_FILE.exists():
        raise RuntimeError(t("commands_missing_file", path=COMMANDS_FILE))
    try:
        size = COMMANDS_FILE.stat().st_size
        if size > MAX_COMMANDS_FILE_BYTES:
            raise RuntimeError(
                t("commands_file_too_large", size=size, limit=MAX_COMMANDS_FILE_BYTES)
            )
        with COMMANDS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            t("commands_invalid_json", lineno=exc.lineno, colno=exc.colno)
        ) from exc
    except OSError as exc:
        raise RuntimeError(t("commands_read_failed", exc=exc)) from exc

    if not isinstance(data, (dict, list)):
        raise RuntimeError(t("commands_bad_toplevel"))
    return data


def build_prompt_view(command_data):
    """Render the compact command contract sent to the model.

    Implementation details (shell mapping, limits, notes) live in
    memories/commands.impl.json and must never appear here. Parameter details are
    included only as names; full type/enum/default information is returned by the
    dispatcher's teaching-style validation errors.
    """
    commands = command_data.get("commands")
    if not isinstance(commands, list):
        commands = []
    lines = []
    for item in commands:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = item.get("parameters")
        signature = ", ".join(params) if isinstance(params, dict) else ""
        description = item.get("description")
        description = description.strip() if isinstance(description, str) else ""
        if item.get("enabled") is not True:
            continue
        line = f"{name}({signature})"
        if description:
            line += f" - {description}"
        if item.get("require_confirmation") is True:
            line += " [needs confirmation]"
        lines.append(line)
    return "\n".join(lines)


def build_command_context(command_data):
    """Describe the local command interface and strict raw-JSON response contract."""
    serialized = build_prompt_view(command_data)
    return (
        t("protocol_prompt")
        + "\n----- available commands BEGIN -----\n"
        + serialized
        + "\n----- available commands END -----"
    )


def _jsonl_append(path, record):
    """Append one complete JSONL record; flush it before proceeding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _update_tool_log(call_id, **changes):
    """Append an event, rather than rewriting old audit records."""
    record = {"event": "state_update", "call_id": call_id, "timestamp": timestamp()}
    record.update(changes)
    _jsonl_append(TOOL_LOG_FILE, record)


def _read_tool_log(call_id=None, limit=10):
    """Return metadata records only; never include raw command output."""
    if not TOOL_LOG_FILE.exists():
        return {"status": "ok", "message": t("tool_log_empty"), "records": []}
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        return {"status": "error", "message": t("tool_log_bad_limit")}
    records = []
    try:
        with TOOL_LOG_FILE.open("r", encoding="utf-8") as f:
            # Bound work and tolerate a truncated final line after a crash.
            lines = f.readlines()[-MAX_TOOL_LOG_LINES:]
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if call_id is None or item.get("call_id") == call_id:
                records.append(item)
    except OSError as exc:
        return {"status": "error", "message": t("tool_log_read_failed", exc_type=type(exc).__name__)}
    return {"status": "ok", "records": records[-limit:], "count_returned": min(len(records), limit)}


def _save_permitted_tool_output(call_id, result):
    """Persist only output already permitted by dispatcher; never persist rejected raw output."""
    if not isinstance(result, dict) or result.get("status") == "transmission_rejected":
        return None
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None
    raw = ("[stdout]\n" + stdout + "\n[stderr]\n" + stderr).encode("utf-8")
    if not stdout and not stderr:
        return None
    TOOL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TOOL_OUTPUT_DIR / (call_id + ".txt")
    # Generated name only; never accept a model-supplied path.
    with output_path.open("wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return {"output_file": output_path.relative_to(LIBRARY_DIR).as_posix(),
            "output_bytes": len(raw), "output_sha256": hashlib.sha256(raw).hexdigest()}


def _handle_tool_log(args):
    if not isinstance(args, dict) or set(args) - {"call_id", "limit"}:
        return {"status": "error", "message": t("tool_log_bad_args")}
    call_id = args.get("call_id")
    if call_id is not None and (not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", call_id)):
        return {"status": "error", "message": t("tool_log_bad_call_id")}
    limit = args.get("limit", 10)
    result = _read_tool_log(call_id, limit)
    # Output is metadata-only and bounded; do not expose absolute device paths.
    return result


def _run_tool(envelope):
    """Execute one validated tool_call; the dispatcher remains the security authority."""
    if envelope.get("command") == "tool_log":
        result = _handle_tool_log(envelope.get("arguments", {}))
        result.setdefault("command", "tool_log")
        result.setdefault("exit_code", 0 if result.get("status") == "ok" else 1)
        return result
    # Imported here rather than at module scope: the dispatcher imports nothing
    # from this module, but keeping it lazy preserves the original behaviour of
    # surfacing a missing dispatcher only when a tool is actually requested.
    import agent_dispatcher
    return agent_dispatcher.dispatch_tool_call(envelope, BASE_DIR)


def register_api_host():
    """Give api.py the host services it cannot import without a cycle."""
    api.set_host(
        load_commands=load_command_definitions,
        command_context=build_command_context,
        run_tool=_run_tool,
        log_tool_state=lambda call_id, **changes: _update_tool_log(call_id, **changes),
        save_tool_output=_save_permitted_tool_output,
        timestamp=timestamp,
    )


register_api_host()


def add_usage(memory, usage):
    totals = memory["usage"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key, 0)
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + int(value)
    totals["requests"] = totals.get("requests", 0) + int(usage.get("requests", 1))


def load_memory_or_raise():
    """Load the active memory branch, refusing to start on an unreadable file.

    Returning an empty memory would look like a fresh profile, and the next save
    would overwrite the real history with it. Fail instead.
    """
    active_path = memory_store.initialize_memories()
    memory = memory_store.load_memory_file(active_path)
    if memory is None:
        raise RuntimeError(t("startup_memory_file_empty"))
    return active_path, memory


def check_runtime_files():
    """Fail early when a tool-call module is missing instead of on first use."""
    for name in ("agent_dispatcher.py", "command_library_bridge.py"):
        if not (BASE_DIR / name).is_file():
            raise RuntimeError(t("startup_missing_file", name=name))


def run_turn(config, memory, active_path, user_text):
    """Send one user message, append the reply, and report failures inline."""
    memory["messages"].append({"role": "user", "content": user_text})
    try:
        memory_store.save_memory(memory, active_path)
    except OSError as exc:
        print(t("warning_failed_to_save_user_message", exc=exc))

    print(t("ai"), end=t("text"), flush=True)
    try:
        answer, usage = call_api(config, memory, user_text)
        print(answer)
        memory["messages"].append({"role": "assistant", "content": answer})
        add_usage(memory, usage)
        try:
            memory_store.save_memory(memory, active_path)
        except OSError as exc:
            print(t("warning_failed_to_save_ai_reply", exc=exc))

        if usage:
            print(
                t("this_turn_s_tokens_input_usage", arg0=usage.get('prompt_tokens', '?'), arg1=usage.get('completion_tokens', '?'), arg2=usage.get('total_tokens', '?'))
            )
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(t("http_error_exc_code_details", arg0=exc.code, details=details))
        print(t("user_message_has_been_saved_locally"))
    except urllib.error.URLError as exc:
        print(t("network_error_exc_reason", arg0=exc.reason))
        print(t("user_message_has_been_saved_locally_2"))
    except (TimeoutError, json.JSONDecodeError, RuntimeError, OSError) as exc:
        print(t("request_or_processing_failed_type_exc", arg0=type(exc).__name__, exc=exc))
        print(t("user_message_has_been_saved_locally_3"))
    except Exception as exc:
        print(t("unexpected_error_type_exc_name_exc", arg0=type(exc).__name__, exc=exc))
        print(t("user_message_has_been_saved_locally_4"))
