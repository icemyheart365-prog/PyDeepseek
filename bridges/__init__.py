"""Platform backends for the local command library.

Backends return (stdout_bytes, stderr_bytes, exit_code) and never raise for
command-level failures. Selection is explicit and overridable:

  DEEPSEEK_BRIDGE=python       pure-Python implementation (default everywhere)
  DEEPSEEK_BRIDGE=powershell   native Windows backend (falls back when unavailable)
  DEEPSEEK_BRIDGE=posix        command_library_v4.sh (requires a POSIX shell)

The Python backend is the default because it is the only one that cannot hang on
A-Shell command wrappers and needs no external interpreter.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PS_SCRIPT = Path(__file__).resolve().parent / "command_library.ps1"
SH_SCRIPT = Path(__file__).resolve().parent / "command_library_v4.sh"

BRIDGE_ENV = "DEEPSEEK_BRIDGE"
BACKENDS = ("python", "powershell", "posix")
DEFAULT_BACKEND = "python"
PS_TIMEOUT_SECONDS = 20


def available_backends() -> list[str]:
    return list(BACKENDS)


def select_backend() -> str:
    """Resolve the backend from the environment, downgrading unsupported choices."""
    requested = os.environ.get(BRIDGE_ENV, "").strip().lower()
    if requested in BACKENDS:
        if requested == "powershell" and not PS_SCRIPT.is_file():
            return DEFAULT_BACKEND
        if requested == "posix" and not SH_SCRIPT.is_file():
            return DEFAULT_BACKEND
        return requested
    return DEFAULT_BACKEND


def run_python(args: list[str]) -> tuple[bytes, bytes, int]:
    from bridges.python_backend import python_fallback
    return python_fallback(args)


def run_powershell(args: list[str]) -> tuple[bytes, bytes, int]:
    """Invoke the native backend; every argument is passed as JSON on stdin.

    No user or model text is ever concatenated into a PowerShell command line.
    """
    import base64
    import json

    if not PS_SCRIPT.is_file():
        return b"", b"PowerShell backend script is missing", 2

    request = json.dumps({"args": args}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-File", str(PS_SCRIPT),
            ],
            input=request.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return b"", f"PowerShell backend failed to start: {type(exc).__name__}".encode(), 2

    try:
        envelope = json.loads(proc.stdout.decode("utf-8"))
        stdout = base64.b64decode(envelope.get("stdout_b64", "") or "")
        stderr = base64.b64decode(envelope.get("stderr_b64", "") or "")
        return stdout, stderr, int(envelope.get("exit_code", 0))
    except (ValueError, TypeError, UnicodeDecodeError):
        return b"", b"PowerShell backend returned an invalid envelope", 2


def run_posix(args: list[str]) -> tuple[bytes, bytes, int]:
    if not SH_SCRIPT.is_file():
        return b"", b"POSIX backend script is missing", 2
    try:
        proc = subprocess.run(
            ["sh", str(SH_SCRIPT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return b"", f"POSIX backend failed to start: {type(exc).__name__}".encode(), 2
    return proc.stdout, proc.stderr, proc.returncode


_RUNNERS = {
    "python": run_python,
    "powershell": run_powershell,
    "posix": run_posix,
}


def run(args: list[str], backend: str | None = None) -> tuple[bytes, bytes, int]:
    name = backend or select_backend()
    runner = _RUNNERS.get(name, run_python)
    try:
        return runner(args)
    except Exception as exc:  # defensive: a backend bug must not crash the chat
        return b"", f"Backend {name} error: {type(exc).__name__}".encode(), 2
