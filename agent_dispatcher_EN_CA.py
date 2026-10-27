#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local command dispatcher for the Deepseek Agent.

The Agent may request only a named command plus structured arguments.
This module validates the request against memories/commands.json, asks for
execution confirmation when required, invokes the tested bridge, and applies
an independent output-transmission gate.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

COMMANDS_FILE_NAME = "commands.json"
BRIDGE_FILE_NAME = "command_library_bridge.py"
OUTPUT_CONFIRM_BYTES = 4096
MAX_CAPTURE_BYTES = 1_048_576
MAX_TOOL_ROUNDS = 8


class DispatchError(Exception):
    pass


def _load_definitions(project_dir: Path) -> tuple[dict, dict[str, dict]]:
    path = project_dir / "memories" / COMMANDS_FILE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DispatchError("Missing command definition file: memories/commands.json") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchError(f"Unable to read the command definition file: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("commands"), list):
        raise DispatchError("Invalid commands.json: a 'commands' array is required.")
    commands = {}
    for item in data["commands"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            commands[item["name"]] = item
    return data, commands


def _validate_args(definition: dict, supplied: Any) -> dict:
    if not isinstance(supplied, dict):
        raise DispatchError("arguments must be a JSON object.")
    schema = definition.get("parameters", {})
    if not isinstance(schema, dict):
        raise DispatchError("Invalid command parameter definition.")
    unknown = set(supplied) - set(schema)
    if unknown:
        raise DispatchError("Unknown parameters: " + ", ".join(sorted(unknown)))
    args = {}
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            raise DispatchError(f"Invalid definition for parameter {name}.")
        if name in supplied:
            value = supplied[name]
        elif "default" in spec:
            value = spec["default"]
        elif spec.get("required", False):
            raise DispatchError(f"Missing required parameter: {name}")
        else:
            continue

        typ = spec.get("type")
        if typ == "string":
            if not isinstance(value, str):
                raise DispatchError(f"Parameter {name} must be a string.")
        elif typ == "boolean":
            if not isinstance(value, bool):
                raise DispatchError(f"Parameter {name} must be true or false.")
        elif typ == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise DispatchError(f"Parameter {name} must be an integer.")
        elif typ == "array":
            if not isinstance(value, list):
                raise DispatchError(f"Parameter {name} must be an array.")
            allowed = spec.get("items_enum")
            if allowed and any(v not in allowed for v in value):
                raise DispatchError(f"Parameter {name} contains an unsupported value.")
        if "enum" in spec and value not in spec["enum"]:
            raise DispatchError(f"Parameter {name} must be one of: {', '.join(map(str, spec['enum']))}")
        if isinstance(value, int) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise DispatchError(f"Parameter {name} is below the allowed minimum.")
            if "maximum" in spec and value > spec["maximum"]:
                raise DispatchError(f"Parameter {name} is above the allowed maximum.")
        if isinstance(value, str) and len(value) > 4096:
            raise DispatchError(f"Parameter {name} is too long.")
        args[name] = value
    return args


def _ask_yes_no(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo confirmation received; treating as declined.")
        return False
    return answer in {"y", "yes"}


def _safe_result(status: str, command: str, message: str, **extra) -> dict:
    result = {"status": status, "command": command, "message": message}
    result.update(extra)
    return result


def _invoke_bridge(project_dir: Path, command: str, args: dict) -> dict:
    bridge = project_dir / BRIDGE_FILE_NAME
    if not bridge.is_file():
        raise DispatchError("command_library_bridge.py is missing; the command was not executed.")

    # Translate named JSON parameters to the bridge's fixed positional protocol.
    if command == "ls":
        if args.get("sort", "name") != "name":
            raise DispatchError("This version only supports sort='name'; other sort orders are not implemented yet.")
        argv = ["ls", args.get("path", "."), args.get("format", "name"),
                "true" if args.get("all", False) else "false"]
    elif command in {"stat", "sha256", "cat"}:
        argv = [command, args["path"]]
    elif command == "wc":
        count = args.get("count", ["bytes", "lines", "words"])
        if not count:
            raise DispatchError("wc.count requires at least one count type.")
        # Multiple modes are combined from independent allowlisted calls.
        pieces = []
        for mode in count:
            raw = _run_bridge_once(project_dir, ["wc", args["path"], mode])
            if raw.get("status") != "ok":
                return raw
            decoded = base64.b64decode(raw.get("stdout_b64", ""), validate=True).decode("utf-8", "replace").strip()
            pieces.append(f"{mode}: {decoded}")
        body = ("\n".join(pieces) + "\n").encode("utf-8")
        return {
            "status": "ok", "exit_code": 0,
            "stdout_bytes": len(body), "stderr_bytes": 0,
            "stdout_b64": base64.b64encode(body).decode("ascii"),
            "stderr_b64": "", "truncated": False
        }
    elif command == "search_files":
        argv = ["search_files", args.get("path", "."), args["pattern"],
                "true" if args.get("recursive", False) else "false"]
    elif command == "grep":
        if args.get("regex", False):
            raise DispatchError("This version only supports literal text matching; regex=true is not implemented yet.")
        argv = ["grep", args["path"], args["pattern"],
                "true" if args.get("ignore_case", False) else "false",
                str(args.get("max_matches", 100))]
    elif command in {"head", "tail"}:
        argv = [command, args["path"], str(args.get("lines", 20))]
    else:
        raise DispatchError(f"No executable implementation exists for command {command!r} yet.")
    return _run_bridge_once(project_dir, argv)


def _run_bridge_once(project_dir: Path, argv: list[str]) -> dict:
    bridge = project_dir / BRIDGE_FILE_NAME
    proc = subprocess.run(
        [sys.executable, str(bridge), *argv],
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=12,
        check=False,
    )
    try:
        result = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DispatchError("The local command interface returned invalid JSON; the raw output was discarded.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise DispatchError("The local command interface returned an invalid result structure.")
    for field in ("stdout_b64", "stderr_b64"):
        value = result.get(field, "")
        if not isinstance(value, str):
            raise DispatchError("The local command interface returned a Base64 field of the wrong type.")
        try:
            decoded = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise DispatchError("The local command interface returned invalid Base64; the raw content was discarded.") from exc
        if field == "stdout_b64":
            declared = result.get("stdout_bytes")
        else:
            declared = result.get("stderr_bytes")
        if declared is not None and isinstance(declared, int) and len(decoded) != declared:
            raise DispatchError("Byte-count validation failed for the local command interface.")
    return result



def _invoke_web_tool(project_dir: Path, command: str, args: dict) -> dict:
    """Call DeepSeek's hosted Search/Fetch API; API key is read but never returned."""
    config_path = project_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DispatchError("config.json does not exist; no network request was sent.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchError("Unable to read config.json; no network request was sent.") from exc
    api_key = config.get("api_key") if isinstance(config, dict) else None
    if not isinstance(api_key, str) or not api_key.strip():
        raise DispatchError("config.json has no valid api_key; no network request was sent.")

    try:
        import web_tools
        if command == "web_search":
            result = web_tools.search(
                api_key,
                args["query"],
                count=args.get("count", 5),
                need_content=args.get("need_content", False),
                ori_question=args.get("ori_question"),
                timeout=args.get("timeout", 30),
            )
        elif command == "web_fetch":
            result = web_tools.fetch(
                api_key,
                args["url"],
                timeout=args.get("timeout", 30),
            )
        else:
            raise DispatchError("Unknown network tool.")
    except Exception as exc:
        # Avoid including exception reprs, request headers, or key-bearing objects.
        try:
            from web_tools import WebToolError
        except ImportError:
            WebToolError = ()  # type: ignore[assignment]
        if isinstance(exc, WebToolError):
            return _safe_result(
                "error", command, str(exc),
                http_status=getattr(exc, "http_status", None),
                retryable=bool(getattr(exc, "retryable", False)),
                exit_code=1,
            )
        if isinstance(exc, (OSError, ValueError, TypeError, KeyError)):
            return _safe_result("error", command, f"Network tool error: {type(exc).__name__}; no page content was returned.", exit_code=1)
        raise

    # Apply the same independent output-transmission gate used for local commands.
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output_bytes = len(encoded)
    estimated_tokens = (output_bytes + 3) // 4
    if output_bytes > MAX_CAPTURE_BYTES:
        return _safe_result(
            "output_limit_exceeded", command,
            "The network result exceeds the local hard limit; the page body was not forwarded.",
            output_bytes=output_bytes, estimated_tokens=estimated_tokens,
            truncated=True, exit_code=1,
        )
    if output_bytes > OUTPUT_CONFIRM_BYTES:
        print(f"\nThe network tool returned {output_bytes} bytes; an estimated {estimated_tokens} tokens.")
        print("Confirmation for the network request and confirmation for transmitting the result are independent.")
        if not _ask_yes_no("Send this network result to the cloud AI? Type y/yes to confirm: [y/N] "):
            # Do not return or persist the raw result when transmission is declined.
            return _safe_result(
                "transmission_rejected", command,
                "The user declined transmission of the network result; the raw page content was neither sent nor persisted.",
                output_bytes=output_bytes, estimated_tokens=estimated_tokens,
                truncated=False, exit_code=1,
            )
    result["output_bytes"] = output_bytes
    result["estimated_tokens"] = estimated_tokens
    result["exit_code"] = 0
    return result


def dispatch_tool_call(request: dict, project_dir: Path) -> dict:
    """Execute one Agent tool_call and return only permitted, bounded data."""
    if not isinstance(request, dict) or request.get("type") != "tool_call":
        return _safe_result("error", "", "The request must be a JSON object with type='tool_call'.")
    command = request.get("command")
    if not isinstance(command, str) or not command:
        return _safe_result("error", "", "Missing a valid command name.")
    try:
        policy, definitions = _load_definitions(project_dir)
        definition = definitions.get(command)
        if definition is None:
            return _safe_result("error", command, "The command is not defined in memories/commands.json.")
        if definition.get("enabled") is not True:
            return _safe_result("error", command, "This command is disabled and was not executed.")
        args = _validate_args(definition, request.get("arguments", {}))

        if command == "help":
            enabled = [f"{x['name']}: {x.get('description', '')}" for x in definitions.values() if x.get("enabled") is True]
            return _safe_result("ok", command, "Available commands.", stdout="\n".join(enabled), output_bytes=sum(len(x.encode()) for x in enabled), estimated_tokens=0, exit_code=0)
        if command == "pwd":
            return _safe_result("ok", command, "The only accessible root is library/; absolute device paths are not disclosed.", stdout="library/\n", output_bytes=9, estimated_tokens=3, exit_code=0)
        if command == "cancel":
            return _safe_result("ok", command, "No asynchronously running command to cancel.", stdout="", output_bytes=0, estimated_tokens=0, exit_code=0)

        if definition.get("require_confirmation", False):
            reason = request.get("reason", "")
            print(f"\nThe AI requests execution of local command: {command}")
            print(f"Arguments: {json.dumps(args, ensure_ascii=False)}")
            if reason:
                print(f"Reason: {str(reason)[:500]}")
            if not _ask_yes_no("Allow execution? Type y/yes to confirm: [y/N] "):
                return _safe_result("confirmation_required", command, "The user declined or did not confirm execution; the command did not run.")

        if command in {"web_search", "web_fetch"}:
            return _invoke_web_tool(project_dir, command, args)

        result = _invoke_bridge(project_dir, command, args)
        stdout_raw = base64.b64decode(result.get("stdout_b64", ""), validate=True)
        stderr_raw = base64.b64decode(result.get("stderr_b64", ""), validate=True)
        total_bytes = len(stdout_raw) + len(stderr_raw)
        estimated_tokens = (total_bytes + 3) // 4
        hard_cap = int(policy.get("global_policy", {}).get("max_capture_bytes", MAX_CAPTURE_BYTES))
        if total_bytes > hard_cap or result.get("status") == "output_limit_exceeded":
            return _safe_result("output_limit_exceeded", command, "The output exceeds the local hard limit; the raw output will not be forwarded.", output_bytes=total_bytes, estimated_tokens=estimated_tokens, truncated=True, exit_code=result.get("exit_code"))

        if total_bytes > OUTPUT_CONFIRM_BYTES:
            print(f"\nCommand output size: {total_bytes} bytes; an estimated {estimated_tokens} tokens.")
            print("Execution confirmation and output-transmission confirmation are independent.")
            if not _ask_yes_no("Send this output to the cloud AI? Type y/yes to confirm: [y/N] "):
                return _safe_result("transmission_rejected", command, "The user declined transmission of the output; the raw stdout/stderr was neither sent nor persisted.", output_bytes=total_bytes, estimated_tokens=estimated_tokens, truncated=False, exit_code=result.get("exit_code"))

        stdout_text = stdout_raw.decode("utf-8", errors="replace")
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        response = _safe_result(
            result.get("status", "error"),
            command,
            "Command completed." if result.get("status") == "ok" else "Command failed.",
            exit_code=result.get("exit_code"),
            stdout=stdout_text, stderr=stderr_text,
            output_bytes=total_bytes, estimated_tokens=estimated_tokens,
            truncated=bool(result.get("truncated", False)),
        )
        return response
    except DispatchError as exc:
        return _safe_result("error", command, str(exc))
    except subprocess.TimeoutExpired:
        return _safe_result("error", command, "The local command timed out; the chat loop can continue.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _safe_result("error", command, f"Local command interface error: {type(exc).__name__}: {exc}")


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Provide one JSON tool_call argument."}))
        return 2
    try:
        request = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "message": "Invalid JSON request."}))
        return 2
    result = dispatch_tool_call(request, Path(__file__).resolve().parent)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
