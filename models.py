"""Model catalogue and selection helpers.

Single source of truth for which models the app offers. Adding a model means
editing KNOWN_MODELS here and nothing else: both the first-run wizard and the
/settings menu read from this module.

Slot layout
-----------
``/settings`` renders a numbered menu. Presets occupy slots 1..PRESET_MODEL_SLOTS
and the manual-entry row is pinned at MANUAL_MODEL_INDEX so it never moves as the
preset list grows. The check below fails loudly at import time if the catalogue
outgrows the reserved slots, rather than silently shifting the menu.
"""
from __future__ import annotations

KNOWN_MODELS = ("deepseek-flash", "deepseek-v4-pro")
DEFAULT_MODEL = KNOWN_MODELS[0]

# Slots 1-8 are reserved for presets so the manual entry row can stay pinned at 9.
MANUAL_MODEL_INDEX = 9
PRESET_MODEL_SLOTS = MANUAL_MODEL_INDEX - 1
MANUAL_MODEL_SENTINEL = "__manual__"

if len(KNOWN_MODELS) > PRESET_MODEL_SLOTS:
    raise RuntimeError(
        f"KNOWN_MODELS has {len(KNOWN_MODELS)} entries but only {PRESET_MODEL_SLOTS} slots "
        f"are available before the manual entry at {MANUAL_MODEL_INDEX}. "
        "Raise MANUAL_MODEL_INDEX or remove a preset."
    )


def preset_entries():
    """Return [(slot, model), ...] for the numbered preset menu."""
    return list(enumerate(KNOWN_MODELS, start=1))


def resolve_preset(raw, default=DEFAULT_MODEL):
    """Map a menu slot typed by the user to a model name, falling back to default."""
    for index, model in preset_entries():
        if str(index) == str(raw):
            return model
    return default


def settings_menu(translator, current_model=None):
    """Build (entries, labels) for the /settings model picker.

    ``translator`` is passed in rather than imported so this module stays free of
    UI-layer dependencies and cannot form an import cycle with the app core.
    """
    entries = preset_entries()
    labels = []
    for index, model in entries:
        marker = " *" if model == current_model else ""
        labels.append((index, translator("settings_model_option", model=model) + marker))
    entries.append((MANUAL_MODEL_INDEX, MANUAL_MODEL_SENTINEL))
    labels.append((MANUAL_MODEL_INDEX, translator("settings_manual_option")))
    return entries, labels
