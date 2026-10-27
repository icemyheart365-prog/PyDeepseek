#!/usr/bin/env python3
"""Python JSON/Base64 wrapper around the local command backends.

The backend is selected per platform (see bridges/__init__.py). This module only
handles the JSON envelope, so the agent never sees raw process output.
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import bridges


def emit(obj: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    return code


def main() -> int:
    if len(sys.argv) < 2:
        return emit({"status": "error", "exit_code": 2, "message": "Missing command name"}, 2)

    # The backend is chosen per platform; see bridges/__init__.py. Every backend
    # returns raw bytes and never raises for command-level failures.
    args = sys.argv[1:]
    backend = bridges.select_backend()
    stdout, stderr, rc = bridges.run(args, backend)
    return emit({
        "status": "ok" if rc == 0 else "error",
        "exit_code": rc,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "truncated": False,
        "backend": backend,
    }, 0)


if __name__ == "__main__":
    raise SystemExit(main())
