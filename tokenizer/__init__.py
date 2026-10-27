"""Local DeepSeek token counting.

Uses the `tokenizers` package with the bundled vocabulary. Everything degrades
gracefully: if the package or tokenizer.json is unavailable, callers get None
from try_load_token_counter() and fall back to a byte-based estimate.

Usage:
    from tokenizer import token_counter          # from the project root
    n = token_counter("text")
"""
from __future__ import annotations

import os

try:
    from tokenizers import Tokenizer
except ImportError:  # e.g. A-Shell, where native wheels are unavailable
    Tokenizer = None

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")

_TOKENIZER = None
_LOAD_ERROR = None

if Tokenizer is not None:
    try:
        _TOKENIZER = Tokenizer.from_file(_MODEL_PATH)
    except Exception as exc:  # missing or corrupt vocabulary
        _LOAD_ERROR = exc


def token_counter(text: str) -> int:
    """Return the token count for text."""
    if _TOKENIZER is None:
        raise RuntimeError(f"tokenizer unavailable: {_LOAD_ERROR or 'tokenizers not installed'}")
    return len(_TOKENIZER.encode(text).ids)


def approximate_token_count(text: str) -> int:
    """Byte-based fallback used when the tokenizer cannot load (~4 bytes per token)."""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def try_load_token_counter():
    """Return token_counter when the tokenizer loaded, else None."""
    return None if _TOKENIZER is None else token_counter


def is_available() -> bool:
    return _TOKENIZER is not None


def load_error():
    """Return the exception that prevented loading, if any."""
    return _LOAD_ERROR
