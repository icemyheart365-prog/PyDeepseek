#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal JSON-backed translation helper shared by the chat entry points.

Strings live in i18n/<lang>.json as flat key -> template mappings. Templates use
str.format placeholders, so `t("greeting", name="x")` maps "你好，{name}" to the
final text. Missing keys fail loudly: a half-translated build is worse than a
crash, because silent fallbacks leak the wrong language into the UI.
"""
from __future__ import annotations

import json
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parent / "i18n"
SUPPORTED_LANGS = ("zh_cn", "en_ca")
# Reference catalog for check_parity() and the per-key fallback. It is also the
# language the app falls back to when --lang is absent and config.json names no
# supported language, so it must stay a complete translation of every key.
DEFAULT_LANG = "zh_cn"
# Last-resort UI language when neither --lang nor config.json has a usable value.
FALLBACK_LANG = "en_ca"

_cache: dict[str, dict[str, str]] = {}


class TranslationError(Exception):
    pass


def available_langs() -> tuple[str, ...]:
    return SUPPORTED_LANGS


def load_catalog(lang: str) -> dict[str, str]:
    if lang in _cache:
        return _cache[lang]
    if lang not in SUPPORTED_LANGS:
        raise TranslationError(f"Unsupported language: {lang!r}")
    path = I18N_DIR / f"{lang}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranslationError(f"Missing translation file: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationError(f"Unable to read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise TranslationError(f"{path.name} must contain a JSON object")
    _cache[lang] = data
    return data


def check_parity() -> None:
    """Fail fast when catalogs disagree, instead of surfacing gaps at runtime."""
    catalogs = {lang: load_catalog(lang) for lang in SUPPORTED_LANGS}
    reference = set(catalogs[DEFAULT_LANG])
    for lang, catalog in catalogs.items():
        if lang == DEFAULT_LANG:
            continue
        missing = sorted(reference - set(catalog))
        extra = sorted(set(catalog) - reference)
        if missing or extra:
            raise TranslationError(
                f"{lang}.json does not match {DEFAULT_LANG}.json: "
                f"missing={missing or 'none'} extra={extra or 'none'}"
            )


class Translator:
    """Callable translator bound to one language."""

    def __init__(self, lang: str = DEFAULT_LANG):
        self.lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
        self._catalog = load_catalog(self.lang)
        self._fallback = load_catalog(DEFAULT_LANG) if self.lang != DEFAULT_LANG else self._catalog

    def __call__(self, key: str, **kwargs) -> str:
        template = self._catalog.get(key)
        if template is None:
            template = self._fallback.get(key)
        if template is None:
            return key
        if not isinstance(template, str):
            return key
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return template
