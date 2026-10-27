#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local command dispatcher for the Deepseek Agent.

The Agent may request only a named command plus structured arguments.
This module validates the request against memories/commands.json (the prompt-visible
contract; implementation details live in memories/commands.impl.json), asks for
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

from tokenizer import token_counter, try_load_token_counter as _try_load_token_counter

_TOKEN_COUNTER = _try_load_token_counter()

COMMANDS_FILE_NAME = "commands.json"
COMMANDS_IMPL_FILE_NAME = "commands.impl.json"
BRIDGE_FILE_NAME = "command_library_bridge.py"
OUTPUT_CONFIRM_BYTES = 4096
MAX_CAPTURE_BYTES = 1_048_576
MAX_TOOL_ROUNDS = 8


class DispatchError(Exception):
    pass


def _count_tokens_from_bytes(data: bytes) -> int:
    """Token count for raw output; falls back to the byte estimate if the tokenizer is unavailable."""
    if _TOKEN_COUNTER is not None:
        try:
            return _TOKEN_COUNTER(data.decode("utf-8", errors="replace"))
        except Exception:
            pass
    return (len(data) + 3) // 4


def _check_impl_consistency(project_dir: Path, definitions: dict) -> None:
    """Hard-fail when commands.impl.json and commands.json describe different command sets."""
    path = project_dir / "memories" / COMMANDS_IMPL_FILE_NAME
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchError(f"无法读取 {COMMANDS_IMPL_FILE_NAME}：{exc}") from exc
    entries = data.get("commands") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise DispatchError(f"{COMMANDS_IMPL_FILE_NAME} 格式错误：需要 commands 数组。")
    impl_names = {item["name"] for item in entries if isinstance(item, dict) and isinstance(item.get("name"), str)}
    only_contract = set(definitions) - impl_names
    only_impl = impl_names - set(definitions)
    if only_contract or only_impl:
        raise DispatchError(
            "命令定义不一致："
            f"仅存在于 commands.json：{sorted(only_contract) or '无'}；"
            f"仅存在于 {COMMANDS_IMPL_FILE_NAME}：{sorted(only_impl) or '无'}。"
            "请同步两份文件后再试。"
        )


def _load_definitions(project_dir: Path) -> tuple[dict, dict[str, dict]]:
    path = project_dir / "memories" / COMMANDS_FILE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DispatchError("命令定义文件缺失：memories/commands.json") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchError(f"无法读取命令定义文件：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("commands"), list):
        raise DispatchError("commands.json 格式错误：需要 commands 数组。")
    commands = {}
    for item in data["commands"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            commands[item["name"]] = item
    _check_impl_consistency(project_dir, commands)
    return data, commands


def _describe_param(name, spec):
    """Render one parameter's contract for error messages (teaching-style hints)."""
    if not isinstance(spec, dict):
        return name
    bits = [name]
    bits.append(f"type={spec.get('type', 'any')}")
    if "enum" in spec:
        bits.append("allowed=" + ",".join(map(str, spec["enum"])))
    if "items_enum" in spec:
        bits.append("items=" + ",".join(map(str, spec["items_enum"])))
    if "default" in spec:
        bits.append(f"default={json.dumps(spec['default'], ensure_ascii=False)}")
    if spec.get("required", False):
        bits.append("required")
    detail = ""
    if isinstance(spec.get("description"), str) and spec["description"].strip():
        detail = "  " + spec["description"].strip()
    return "(" + "; ".join(bits) + ")" + detail


def _unknown_param_error(unknown, schema):
    """Explain rejected parameters and show the whole contract so the model can retry once."""
    lines = [f"参数无效：{', '.join(sorted(unknown))} 不在该命令的定义中。"]
    for name, spec in schema.items():
        lines.append("  - " + _describe_param(name, spec))
    return "\n".join(lines)


def _validate_args(definition: dict, supplied: Any) -> dict:
    if not isinstance(supplied, dict):
        raise DispatchError("arguments 必须是 JSON 对象。")
    schema = definition.get("parameters", {})
    if not isinstance(schema, dict):
        raise DispatchError("命令参数定义格式错误。")
    unknown = set(supplied) - set(schema)
    if unknown:
        raise DispatchError(_unknown_param_error(unknown, schema))
    args = {}
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            raise DispatchError(f"参数 {name} 的定义无效。")
        if name in supplied:
            value = supplied[name]
        elif "default" in spec:
            value = spec["default"]
        elif spec.get("required", False):
            raise DispatchError(f"缺少必需参数：{name}\n  - " + _describe_param(name, spec))
        else:
            continue

        typ = spec.get("type")
        if typ == "string":
            if not isinstance(value, str):
                raise DispatchError(f"参数 {name} 必须是字符串。")
        elif typ == "boolean":
            if not isinstance(value, bool):
                raise DispatchError(f"参数 {name} 必须是 true 或 false。")
        elif typ == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise DispatchError(f"参数 {name} 必须是整数。")
        elif typ == "array":
            if not isinstance(value, list):
                raise DispatchError(f"参数 {name} 必须是数组。")
            allowed = spec.get("items_enum")
            if allowed and any(v not in allowed for v in value):
                raise DispatchError(
                    f"参数 {name} 包含不支持的值；允许：" + ", ".join(map(str, allowed))
                )
        if "enum" in spec and value not in spec["enum"]:
            raise DispatchError(
                f"参数 {name} 的值 {json.dumps(value, ensure_ascii=False)} 无效；允许："
                + ", ".join(map(str, spec["enum"]))
                + f"（默认：{json.dumps(spec.get('default'), ensure_ascii=False)}）"
            )
        if isinstance(value, int) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise DispatchError(f"参数 {name} 小于允许的最小值。")
            if "maximum" in spec and value > spec["maximum"]:
                raise DispatchError(f"参数 {name} 大于允许的最大值。")
        if isinstance(value, str) and len(value) > 4096:
            raise DispatchError(f"参数 {name} 过长。")
        args[name] = value
    return args


def _ask_yes_no(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n未收到确认，按拒绝处理。")
        return False
    return answer in {"y", "yes", "是", "确认", "同意"}


def _safe_result(status: str, command: str, message: str, **extra) -> dict:
    result = {"status": status, "command": command, "message": message}
    result.update(extra)
    return result


def _invoke_bridge(project_dir: Path, command: str, args: dict) -> dict:
    bridge = project_dir / BRIDGE_FILE_NAME
    if not bridge.is_file():
        raise DispatchError("command_library_bridge.py 不存在；未执行命令。")

    # Translate named JSON parameters to the bridge's fixed positional protocol.
    if command == "ls":
        if args.get("sort", "name") != "name":
            raise DispatchError("当前版本只支持 sort='name'；其他排序尚未实现。")
        argv = ["ls", args.get("path", "."), args.get("format", "name"),
                "true" if args.get("all", False) else "false"]
    elif command in {"stat", "sha256", "cat"}:
        argv = [command, args["path"]]
    elif command == "wc":
        count = args.get("count", ["bytes", "lines", "words"])
        if not count:
            raise DispatchError("wc.count 至少需要一个计数类型。")
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
            raise DispatchError("当前版本仅支持字面文本匹配；regex=true 尚未实现。")
        argv = ["grep", args["path"], args["pattern"],
                "true" if args.get("ignore_case", False) else "false",
                str(args.get("max_matches", 100))]
    elif command in {"head", "tail"}:
        argv = [command, args["path"], str(args.get("lines", 20))]
    else:
        raise DispatchError(f"命令 {command!r} 尚无可执行实现。")
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
        raise DispatchError("本地命令接口返回了无效 JSON；原始输出已被丢弃。") from exc
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise DispatchError("本地命令接口返回了无效结果结构。")
    for field in ("stdout_b64", "stderr_b64"):
        value = result.get(field, "")
        if not isinstance(value, str):
            raise DispatchError("本地命令接口的 Base64 字段类型错误。")
        try:
            decoded = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise DispatchError("本地命令接口返回了非法 Base64；原始内容已丢弃。") from exc
        if field == "stdout_b64":
            declared = result.get("stdout_bytes")
        else:
            declared = result.get("stderr_bytes")
        if declared is not None and isinstance(declared, int) and len(decoded) != declared:
            raise DispatchError("本地命令接口的字节数校验失败。")
    return result



def _invoke_web_tool(project_dir: Path, command: str, args: dict) -> dict:
    """Call DeepSeek's hosted Search/Fetch API; API key is read but never returned."""
    config_path = project_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DispatchError("config.json 不存在；未发送联网请求。") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchError("无法读取 config.json；未发送联网请求。") from exc
    api_key = config.get("api_key") if isinstance(config, dict) else None
    if not isinstance(api_key, str) or not api_key.strip():
        raise DispatchError("config.json 中没有有效的 api_key；未发送联网请求。")

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
            raise DispatchError("未知联网工具。")
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
            return _safe_result("error", command, f"联网工具错误：{type(exc).__name__}；未返回网页内容。", exit_code=1)
        raise

    # Apply the same independent output-transmission gate used for local commands.
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output_bytes = len(encoded)
    estimated_tokens = _count_tokens_from_bytes(encoded)
    if output_bytes > MAX_CAPTURE_BYTES:
        return _safe_result(
            "output_limit_exceeded", command,
            "联网结果超过本地硬限制；网页正文未转发。",
            output_bytes=output_bytes, estimated_tokens=estimated_tokens,
            truncated=True, exit_code=1,
        )
    if output_bytes > OUTPUT_CONFIRM_BYTES:
        print(f"\n联网工具返回 {output_bytes} 字节；估算约 {estimated_tokens} tokens。")
        print("联网请求确认与结果传输确认相互独立。")
        if not _ask_yes_no("允许把联网结果发送给云端 AI？输入 y/yes/是 确认：[y/N] "):
            # Do not return or persist the raw result when transmission is declined.
            return _safe_result(
                "transmission_rejected", command,
                "用户拒绝传输联网结果；原始网页内容未发送，也未持久化。",
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
        return _safe_result("error", "", "请求必须是 type='tool_call' 的 JSON 对象。")
    command = request.get("command")
    if not isinstance(command, str) or not command:
        return _safe_result("error", "", "缺少有效的 command 名称。")
    try:
        policy, definitions = _load_definitions(project_dir)
        definition = definitions.get(command)
        if definition is None:
            return _safe_result("error", command, "命令未在 memories/commands.json 中定义。")
        if definition.get("enabled") is not True:
            return _safe_result("error", command, "此命令已禁用，未执行。")
        args = _validate_args(definition, request.get("arguments", {}))

        if command == "help":
            enabled = [f"{x['name']}: {x.get('description', '')}" for x in definitions.values() if x.get("enabled") is True]
            return _safe_result("ok", command, "可用命令列表。", stdout="\n".join(enabled), output_bytes=sum(len(x.encode()) for x in enabled), estimated_tokens=0, exit_code=0)
        if command == "pwd":
            return _safe_result("ok", command, "当前允许访问的根目录为 library/；不公开设备绝对路径。", stdout="library/\n", output_bytes=9, estimated_tokens=3, exit_code=0)
        if command == "cancel":
            return _safe_result("ok", command, "当前没有异步运行的命令需要取消。", stdout="", output_bytes=0, estimated_tokens=0, exit_code=0)

        if definition.get("require_confirmation", False):
            reason = request.get("reason", "")
            print(f"\nAI 请求执行本地命令：{command}")
            print(f"参数：{json.dumps(args, ensure_ascii=False)}")
            if reason:
                print(f"理由：{str(reason)[:500]}")
            if not _ask_yes_no("允许执行？输入 y/yes/是 确认：[y/N] "):
                return _safe_result("confirmation_required", command, "用户拒绝或未确认执行；命令未运行。")

        if command in {"web_search", "web_fetch"}:
            return _invoke_web_tool(project_dir, command, args)

        result = _invoke_bridge(project_dir, command, args)
        stdout_raw = base64.b64decode(result.get("stdout_b64", ""), validate=True)
        stderr_raw = base64.b64decode(result.get("stderr_b64", ""), validate=True)
        total_bytes = len(stdout_raw) + len(stderr_raw)
        estimated_tokens = _count_tokens_from_bytes(stdout_raw + stderr_raw)
        hard_cap = int(policy.get("global_policy", {}).get("max_capture_bytes", MAX_CAPTURE_BYTES))
        if total_bytes > hard_cap or result.get("status") == "output_limit_exceeded":
            return _safe_result("output_limit_exceeded", command, "输出超过本地硬限制；原始输出不会转发。", output_bytes=total_bytes, estimated_tokens=estimated_tokens, truncated=True, exit_code=result.get("exit_code"))

        if total_bytes > OUTPUT_CONFIRM_BYTES:
            print(f"\n命令输出大小：{total_bytes} 字节；估算约 {estimated_tokens} tokens。")
            print("执行确认与输出传输确认相互独立。")
            if not _ask_yes_no("允许把这段输出发送给云端 AI？输入 y/yes/是 确认：[y/N] "):
                return _safe_result("transmission_rejected", command, "用户拒绝传输输出；原始 stdout/stderr 未发送，也未持久化。", output_bytes=total_bytes, estimated_tokens=estimated_tokens, truncated=False, exit_code=result.get("exit_code"))

        stdout_text = stdout_raw.decode("utf-8", errors="replace")
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        response = _safe_result(
            result.get("status", "error"), command,
            "命令执行完成。" if result.get("status") == "ok" else "命令执行失败。",
            exit_code=result.get("exit_code"),
            stdout=stdout_text, stderr=stderr_text,
            output_bytes=total_bytes, estimated_tokens=estimated_tokens,
            truncated=bool(result.get("truncated", False)),
        )
        return response
    except DispatchError as exc:
        return _safe_result("error", command, str(exc))
    except subprocess.TimeoutExpired:
        return _safe_result("error", command, "本地命令超时；聊天循环仍可继续。")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _safe_result("error", command, f"本地命令接口错误：{type(exc).__name__}: {exc}")


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
