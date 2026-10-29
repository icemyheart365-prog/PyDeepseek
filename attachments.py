"""Attachment resolution and reading from the library/ directory.

Everything here is pure logic: functions raise coded errors instead of printing,
so the caller decides how to word the problem in the user's language.

Security posture (unchanged from the original implementation):
  * only relative paths inside library/ are accepted
  * glob results are re-resolved and re-checked against the library root, so a
    symlink or junction pointing outside is dropped rather than followed
  * extension, per-file size and total size are bounded
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

from paths import LIBRARY_DIR, ensure_library

MAX_ATTACHMENTS = 10
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 512 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".csv", ".tsv", ".xml", ".html", ".htm", ".css", ".js",
    ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".c", ".h", ".cpp",
    ".hpp", ".rs", ".go", ".rb", ".php", ".sh", ".bash", ".ps1",
    ".bat", ".sql", ".log", ".properties", ".gradle", ".gradle.kts",
}


class AttachmentError(ValueError):
    """A coded attachment failure.

    ``reason``           empty_pattern | outside_library | no_match |
                         too_many | unsupported_type | file_too_large |
                         total_too_large | not_utf8
    ``path``             the offending path when there is one
    ``pattern``          the pattern the user supplied
    ``count`` / ``limit`` numeric details for the size and count reasons
    """

    def __init__(self, reason, path=None, pattern=None, count=None, limit=None):
        super().__init__(reason)
        self.reason = reason
        self.path = path
        self.pattern = pattern
        self.count = count
        self.limit = limit


class AttachmentNotFound(FileNotFoundError):
    """Raised when a pattern matches nothing; kept as FileNotFoundError for callers."""

    def __init__(self, pattern):
        super().__init__(pattern)
        self.pattern = pattern


def resolve_library_pattern(pattern):
    """Resolve a relative pattern inside library; reject absolute/traversal paths."""
    pattern = pattern.strip().replace("\\", "/")
    if not pattern:
        raise AttachmentError("empty_pattern")

    candidate = Path(pattern)
    # On Windows, "/etc/passwd" is not reported as absolute, yet joining it to
    # the library root silently escapes to the drive root. Reject any leading
    # separator explicitly, on every platform.
    if candidate.is_absolute() or pattern.startswith("/") or ".." in candidate.parts:
        raise AttachmentError("outside_library")

    ensure_library()
    matches = []

    # Glob across nested relative paths, but only files that remain within library.
    if any(char in pattern for char in "*?["):
        for path in LIBRARY_DIR.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(LIBRARY_DIR).as_posix()
            if fnmatch.fnmatchcase(rel, pattern):
                matches.append(path)
    else:
        path = LIBRARY_DIR / pattern
        if path.is_file():
            matches = [path]

    safe_matches = []
    library_root = LIBRARY_DIR.resolve()
    for path in matches:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(library_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            safe_matches.append(resolved)

    # Stable ordering and deduplication.
    return sorted(set(safe_matches), key=lambda p: str(p).lower())


def read_attachments(pattern):
    """Read every file matching ``pattern``.

    Returns ``(attachments, total_bytes)`` where attachments is a list of
    ``(relative_name, text)`` pairs. Raises a coded error on any failure.
    """
    paths = resolve_library_pattern(pattern)
    if not paths:
        raise AttachmentNotFound(pattern)

    if len(paths) > MAX_ATTACHMENTS:
        raise AttachmentError("too_many", pattern=pattern,
                              count=len(paths), limit=MAX_ATTACHMENTS)

    attachments = []
    total_bytes = 0

    for path in paths:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise AttachmentError("unsupported_type", path=path.name)

        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise AttachmentError("file_too_large", path=path.name,
                                  count=size, limit=MAX_FILE_BYTES)

        total_bytes += size
        if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise AttachmentError("total_too_large", path=path.name,
                                  count=total_bytes, limit=MAX_TOTAL_ATTACHMENT_BYTES)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentError("not_utf8", path=path.name) from exc

        relative_name = path.relative_to(LIBRARY_DIR.resolve()).as_posix()
        attachments.append((relative_name, text))

    return attachments, total_bytes


# Attachment markers are protocol, not prose: they separate user text from file
# contents inside one message. They stay language-neutral so the model sees the
# same framing regardless of the UI language.
ATTACHMENT_OPEN = "--- attachments begin ---"
ATTACHMENT_CLOSE = "--- attachments end ---"
ATTACHMENT_DEFAULT_INSTRUCTION = "Read and analyse the attached files."
ATTACHMENT_FILE_HEADER = "[file: {name}]"


def format_attachment_message(user_text, attachments):
    """Wrap ``attachments`` into one outgoing message body."""
    parts = []
    parts.append(user_text.strip() or ATTACHMENT_DEFAULT_INSTRUCTION)
    parts.append("\n" + ATTACHMENT_OPEN)
    for name, text in attachments:
        parts.append("\n%s\n```text\n%s\n```" % (ATTACHMENT_FILE_HEADER.format(name=name), text))
    parts.append("\n" + ATTACHMENT_CLOSE)
    return "\n".join(parts)
