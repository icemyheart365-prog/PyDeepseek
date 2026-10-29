"""Filesystem locations and the tiny helpers that create them.

Every path the app touches is derived from BASE_DIR here, so there is exactly
one place to look when asking "where does this file live". Modules import the
constants they need instead of recomputing paths from __file__.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE = BASE_DIR / "config.json"
LEGACY_MEMORY_FILE = BASE_DIR / "memory.json"

MEMORIES_DIR = BASE_DIR / "memories"
DEFAULT_MEMORY_NAME = "default"
COMMANDS_FILE = MEMORIES_DIR / "commands.json"
MEMORY_ARCHIVE_DIR = MEMORIES_DIR / "archive"

LIBRARY_DIR = BASE_DIR / "library"
TOOL_LOG_DIR = LIBRARY_DIR / "logs"
TOOL_LOG_FILE = TOOL_LOG_DIR / "tool_calls.jsonl"
TOOL_OUTPUT_DIR = TOOL_LOG_DIR / "tool_outputs"


def ensure_library():
    """Create the attachment directory if it does not exist yet."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_tool_log_dir():
    """Create the tool-audit log directory if it does not exist yet."""
    TOOL_LOG_DIR.mkdir(parents=True, exist_ok=True)
