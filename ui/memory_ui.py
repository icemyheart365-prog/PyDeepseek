"""Rendering for memory branch errors and listings.

Backend code raises coded errors (MemoryNameError); this module turns them into
sentences in the current UI language, so memory.py never needs to know which
language is active.
"""
from __future__ import annotations


def describe_memory_name_error(exc, tr):
    """Render a MemoryNameError as one user-facing string."""
    reason = getattr(exc, "reason", None)
    if reason == "separator":
        return tr("memory_name_error_separator")
    if reason == "reserved":
        return tr("memory_name_error_reserved")
    if reason == "characters":
        return tr("memory_name_error_characters")
    if reason == "invalid":
        return tr("memory_name_error_invalid")
    return str(exc)
