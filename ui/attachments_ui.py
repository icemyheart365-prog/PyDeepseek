"""Rendering for attachment errors and listings.

Backend code raises coded exceptions; this module turns them into sentences in
the current UI language. Keeping the mapping here means attachments.py never
needs to know which language is active.
"""
from __future__ import annotations

from attachments import AttachmentError, AttachmentNotFound


def describe_attachment_error(exc, tr, library_dir):
    """Render an attachment exception as one user-facing string."""
    if isinstance(exc, AttachmentNotFound):
        return tr("attach_error_no_match", pattern=exc.pattern, library_dir=library_dir)
    if isinstance(exc, AttachmentError):
        reason = exc.reason
        if reason == "empty_pattern":
            return tr("attach_error_usage")
        if reason == "outside_library":
            return tr("attach_error_outside_library")
        if reason == "too_many":
            return tr("attach_error_too_many", count=exc.count, limit=exc.limit)
        if reason == "unsupported_type":
            return tr("attach_error_unsupported_type", name=exc.path)
        if reason == "file_too_large":
            return tr("attach_error_file_too_large", name=exc.path, size=exc.count, limit=exc.limit)
        if reason == "total_too_large":
            return tr("attach_error_total_too_large", limit=exc.limit)
        if reason == "not_utf8":
            return tr("attach_error_not_utf8", name=exc.path)
        return tr("attach_error_unknown", reason=reason)
    return str(exc)


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
