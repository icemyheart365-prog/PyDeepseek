#!/usr/bin/env python3
"""Python JSON/Base64 wrapper for command_library_v4.sh.

This wrapper intentionally uses Python's standard library for JSON and Base64.
The shell backend emits raw stdout/stderr; no Base64 executable or printf is needed.
"""
from __future__ import annotations
import base64
import json
import os
import subprocess
import sys
import hashlib
import fnmatch
import stat as statmod
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SH_SCRIPT = BASE_DIR / "command_library_v4.sh"
MAX_OUTPUT_BYTES = 1_048_576
TIMEOUT_SECONDS = 10

def emit(obj: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    return code

def python_fallback(args: list[str]) -> tuple[bytes, bytes, int]:
    """Conservative Python implementation used only when the shell backend is unavailable."""
    if not args:
        return b"", b"Missing command name", 2
    cmd, *a = args
    root = (BASE_DIR / "library").resolve()
    if not root.is_dir():
        return b"", b"library/ directory does not exist", 2

    def target(rel: str) -> Path:
        q = Path(rel)
        if q.is_absolute() or "\\\\" in rel or ".." in q.parts:
            raise ValueError("Invalid or inaccessible path")
        resolved = (root / q).resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ValueError("Invalid or inaccessible path")
        if resolved.name in {"config.json", ".env", "credentials", "secrets", "global.json"}:
            raise ValueError("Access to this path is blocked")
        return resolved

    try:
        if cmd == "ls":
            rel = a[0] if len(a) > 0 else "."
            fmt = a[1] if len(a) > 1 else "name"
            all_flag = a[2] if len(a) > 2 else "false"
            if fmt not in {"name", "long", "human", "long_human"} or all_flag not in {"true", "false"}:
                raise ValueError("Invalid ls format or include_all flag")
            d = target(rel)
            if not d.is_dir(): raise ValueError("A directory is required")
            entries = sorted(d.iterdir(), key=lambda x: x.name.casefold())
            if all_flag != "true": entries = [x for x in entries if not x.name.startswith(".")]
            if fmt == "name":
                out = "".join(x.name + "\n" for x in entries)
            else:
                rows = []
                for x in entries:
                    st = x.stat()
                    kind = "d" if x.is_dir() else "-"
                    size = st.st_size
                    if fmt in {"human", "long_human"}:
                        units = ["B","K","M","G","T"]
                        val = float(size); idx = 0
                        while val >= 1024 and idx < len(units)-1:
                            val /= 1024; idx += 1
                        size_text = f"{int(val)}{units[idx]}" if idx == 0 else f"{val:.1f}{units[idx]}"
                    else:
                        size_text = str(size)
                    if fmt in {"long","long_human"}:
                        rows.append(f"{kind} {size_text:>8} {x.name}\n")
                    else:
                        rows.append(f"{size_text:>8} {x.name}\n")
                out = "".join(rows)
            return out.encode(), b"", 0
        if cmd in {"stat","wc","sha256","head","tail","cat","grep"}:
            if not a: raise ValueError(f"Usage: {cmd} <relative-path>")
            f = target(a[0])
            if not f.exists(): raise ValueError("Path does not exist")
            if cmd == "stat":
                st = f.stat()
                out = f"type={'directory' if f.is_dir() else 'file'}\nsize_bytes={st.st_size}\n".encode()
                return out, b"", 0
            if not f.is_file(): raise ValueError("A regular file is required")
            data = f.read_bytes()
            if cmd == "sha256":
                return (hashlib.sha256(data).hexdigest() + "  " + str(f) + "\n").encode(), b"", 0
            if cmd == "wc":
                mode = a[1] if len(a) > 1 else "all"
                text = data.decode("utf-8", errors="replace")
                values = {"bytes": str(len(data)), "lines": str(text.count("\n")), "words": str(len(text.split()))}
                if mode in values: out = values[mode] + "\n"
                elif mode == "all": out = f"{values['lines']} {values['words']} {values['bytes']}\n"
                else: raise ValueError("Invalid wc mode")
                return out.encode(), b"", 0
            if cmd == "cat":
                if len(data) > 131072: raise ValueError("File exceeds cat size limit")
                return data, b"", 0
            if cmd in {"head","tail"}:
                n = int(a[1]) if len(a) > 1 else 20
                if not 1 <= n <= 1000: raise ValueError("Line count outside allowed range")
                lines = data.splitlines(keepends=True)
                return b"".join(lines[:n] if cmd == "head" else lines[-n:]), b"", 0
            pattern = a[1] if len(a) > 1 else ""
            ignore = a[2] if len(a) > 2 else "false"
            limit = int(a[3]) if len(a) > 3 else 100
            if len(pattern) > 256 or not 1 <= limit <= 1000 or ignore not in {"true","false"}:
                raise ValueError("Invalid grep arguments")
            text = data.decode("utf-8", errors="replace").splitlines(keepends=True)
            matched = [line for line in text if (pattern.casefold() in line.casefold() if ignore == "true" else pattern in line)]
            return "".join(matched[:limit]).encode(), b"", 0
        if cmd == "search_files":
            if len(a) < 2: raise ValueError("Usage: search_files <relative-directory> <filename-pattern> [recursive]")
            d = target(a[0]); pattern = a[1]; recursive = a[2] if len(a) > 2 else "false"
            if not d.is_dir() or recursive not in {"true","false"} or len(pattern) > 128:
                raise ValueError("Invalid search_files arguments")
            iterator = d.rglob("*") if recursive == "true" else d.iterdir()
            hits = sorted((x for x in iterator if x.is_file() and fnmatch.fnmatch(x.name, pattern)), key=lambda x: str(x))
            return "".join(str(x) + "\n" for x in hits).encode(), b"", 0
        raise ValueError("Unknown or unsupported command")
    except (ValueError, OSError, IndexError) as exc:
        return b"", str(exc).encode("utf-8", errors="replace"), 2

def main() -> int:
    if len(sys.argv) < 2:
        return emit({"status":"error","exit_code":2,"message":"Missing command name"}, 2)

    # A-Shell's exposed command names are not equivalent to a conventional
    # POSIX environment. Use the Python implementation as the reliable default.
    # The SH backend remains available for later opt-in testing, but is not
    # launched implicitly because some A-Shell command wrappers can hang.
    args = sys.argv[1:]
    stdout, stderr, rc = python_fallback(args)
    return emit({
        "status":"ok" if rc == 0 else "error",
        "exit_code":rc,
        "stdout_bytes":len(stdout),
        "stderr_bytes":len(stderr),
        "stdout_b64":base64.b64encode(stdout).decode("ascii"),
        "stderr_b64":base64.b64encode(stderr).decode("ascii"),
        "truncated":False,
        "backend":"python"
    }, 0)

if __name__ == "__main__":
    raise SystemExit(main())
