"""Memory branches: naming, validation, persistence and listing.

Rendering is injected rather than imported: callers pass a ``ui`` object (or a
plain namespace with the same methods) so this module can be exercised without a
terminal. The four helpers in ``DefaultUI`` are the minimum surface the memory
code needs.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from config import load_json, save_json
from paths import DEFAULT_MEMORY_NAME, LEGACY_MEMORY_FILE, MEMORIES_DIR

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MEMORY_VERSION = 4

# Both files live in memories/ but are not memory branches.
RESERVED_MEMORY_FILES = {"commands.json", "commands.impl.json"}

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "requests")

# Names are single safe path components: letters, digits, underscore, hyphen,
# dot and space, but never a leading/trailing space.
_VALID_NAME = re.compile(r"^[^\s](?:[^\s/\\\x00]*[^\s])?$")


class MemoryNameError(ValueError):
    """Raised when a memory branch name is unusable.

    ``reason`` is a stable code so callers can render it in any language:
    ``separator``, ``reserved``, ``invalid`` or ``characters``.
    """

    def __init__(self, reason, name=None):
        super().__init__(reason)
        self.reason = reason
        self.name = name


class MemoryFormatError(RuntimeError):
    """Raised when a memory file is not a valid memory document."""

    def __init__(self, name):
        super().__init__(name)
        self.name = name


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_memory(system_prompt=DEFAULT_SYSTEM_PROMPT):
    return {
        "version": MEMORY_VERSION,
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "system_prompt": system_prompt,
        "summary": "",
        "messages": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        },
    }


def validate_memory(memory, path):
    """Normalise a loaded memory document in place and return it.

    Raises MemoryFormatError when the document is unusable, rather than printing:
    the caller decides how to report it.
    """
    if not isinstance(memory, dict) or not isinstance(memory.get("messages"), list):
        raise MemoryFormatError(path.name)
    memory.setdefault("version", MEMORY_VERSION)
    memory.setdefault("created_at", timestamp())
    memory.setdefault("updated_at", timestamp())
    memory.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    memory.setdefault("summary", "")
    if not isinstance(memory["summary"], str):
        memory["summary"] = ""
    memory.setdefault("usage", {})
    if not isinstance(memory["usage"], dict):
        memory["usage"] = {}
    for key in USAGE_KEYS:
        if not isinstance(memory["usage"].get(key, 0), (int, float)):
            memory["usage"][key] = 0
        else:
            memory["usage"].setdefault(key, 0)
    memory["messages"] = [
        item for item in memory["messages"]
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ]
    return memory


def memory_path(name):
    """Resolve a branch name to (path, clean_name), rejecting unsafe input."""
    if not name or name in (".", "..") or any(ch in name for ch in ("/", "\\", "\x00")):
        raise MemoryNameError("separator", name)
    if name.casefold() in RESERVED_MEMORY_FILES:
        raise MemoryNameError("reserved", name)
    if name.lower().endswith(".json"):
        name = name[:-5]
    if not name or name in (".", "..") or any(ch in name for ch in ("/", "\\", "\x00")):
        raise MemoryNameError("invalid", name)
    if not all(ch.isalnum() or ch in "_- ." for ch in name) or not _VALID_NAME.match(name):
        raise MemoryNameError("characters", name)
    return MEMORIES_DIR / ("%s.json" % name), name


def save_memory(memory, path):
    memory["updated_at"] = timestamp()
    save_json(path, memory)


def load_memory_file(path):
    """Load and normalise one branch; None when the file is absent."""
    loaded = load_json(path, default=None)
    if loaded is None:
        return None
    return validate_memory(loaded, path)


def initialize_memories():
    """Ensure memories/ exists and default.json is present; return its path."""
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    default_path = MEMORIES_DIR / ("%s.json" % DEFAULT_MEMORY_NAME)
    if default_path.exists():
        # Validate existing default instead of overwriting it.
        load_memory_file(default_path)
    else:
        legacy = load_json(LEGACY_MEMORY_FILE, default=None)
        if legacy is not None:
            # Import legacy data without modifying/deleting memory.json.
            legacy = validate_memory(legacy, LEGACY_MEMORY_FILE)
            legacy["version"] = MEMORY_VERSION
            legacy.setdefault("summary", "")
            save_json(default_path, legacy)
        else:
            save_json(default_path, new_memory())
    return default_path


def list_memory_paths():
    """All branch files, alphabetically, excluding reserved files."""
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (p for p in MEMORIES_DIR.glob("*.json")
         if p.name.casefold() not in RESERVED_MEMORY_FILES),
        key=lambda x: x.name.casefold(),
    )
