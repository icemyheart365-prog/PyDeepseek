"""First-run setup wizard.

Collects the three settings the app cannot start without: UI language, API key
and default model, then writes config.json.

The wizard owns no global state. The translator is passed in and returned
through a small holder, because step 1 changes the language and every later
prompt must use the new one. Callers rebind their own translator to
``holder["t"]`` when the wizard returns.
"""
from __future__ import annotations

import models
from config import DEFAULT_BASE_URL, save_json, verify_api_key
from paths import CONFIG_FILE
from translator import DEFAULT_LANG, SUPPORTED_LANGS, Translator

# Native names, shown before any language has been chosen.
NATIVE_LANGUAGE_NAMES = {"zh_cn": "中文", "en_ca": "English"}


def _ask_language():
    """Wizard step 1: pick the UI language before printing anything else."""
    entries = list(enumerate(SUPPORTED_LANGS, start=1))
    for index, code in entries:
        print(f"  [{index}] {NATIVE_LANGUAGE_NAMES.get(code, code)}")
    raw = input("> ").strip() or "2"
    for index, code in entries:
        if str(index) == raw:
            return code
    return DEFAULT_LANG


def _ask_api_key(tr, confirm):
    """Wizard step 2: collect and verify a key, retrying on failure.

    ``confirm`` is the yes/no helper, injected so this module does not depend on
    the terminal layer.
    """
    print("\n" + tr("wizard_key_hint"))
    print(tr("wizard_key_warning"))
    while True:
        try:
            key = input(tr("wizard_key_prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive stdin or Ctrl+C: cancel cleanly instead of crashing
            # with a traceback, so no half-written config is left behind.
            print()
            return None
        if not key:
            return None
        if not key.startswith("sk-"):
            print(tr("wizard_key_bad_prefix"))
            continue

        print(tr("wizard_key_verifying"))
        ok, reason = verify_api_key(key, DEFAULT_BASE_URL)
        if ok:
            print(tr("wizard_key_ok"))
            return key
        print(tr("wizard_key_invalid", reason=reason))
        try:
            if not confirm(tr("wizard_key_retry")):
                return None
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        # Loop back to a full prompt, so a retry never shows a bare ">" with no
        # context about what is being asked for.


def _ask_model():
    """Wizard step 3: choose from known models."""
    for index, model in models.preset_entries():
        print(f"  [{index}] {model}")
    raw = input("> ").strip() or "1"
    return models.resolve_preset(raw)


def run_wizard(tr, confirm):
    """Interactive first-run setup.

    Returns ``(config, translator)``: the translator may have changed if the user
    picked a different language at step 1. Returns ``(None, tr)`` when cancelled.
    """
    print(tr("wizard_title"))
    print(tr("wizard_intro"))

    print(tr("wizard_step_language"))
    lang = _ask_language()
    tr = Translator(lang)
    print(f"  -> {tr(f'language_name_{lang}')}")

    print(tr("wizard_step_key"))
    api_key = _ask_api_key(tr, confirm)
    if not api_key:
        print(tr("wizard_key_empty"))
        return None, tr

    print(tr("wizard_step_model"))
    model = _ask_model()

    config = {
        "api_key": api_key,
        "base_url": DEFAULT_BASE_URL,
        "model": model,
        "language": lang,
    }
    try:
        save_json(CONFIG_FILE, config)
    except OSError as exc:
        print(tr("wizard_save_failed", exc=exc))
        return None, tr

    print(tr("wizard_done"))
    return config, tr
