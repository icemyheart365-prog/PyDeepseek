#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive shell for the DeepSeek local-memory chat.

This module owns everything that touches the terminal in a loop: argument
parsing, first-run setup, the banner, slash commands and turn dispatch. The chat
machinery itself (memories, command contract, tool calls, compaction) lives in
agent_core.py; the transport lives in api.py.

Language precedence, highest first:

  1. ``--lang`` on the command line
  2. the ``language`` value persisted in config.json
  3. ``translator.FALLBACK_LANG``

Note that ``DEFAULT_LANG`` is deliberately *not* used here: it names the
reference catalog for parity checks and the per-key fallback, not the language a
new user should see. A user with no saved preference gets ``FALLBACK_LANG``.

Which language the *model* answers in is a separate concern: ``agent_core.run_turn``
passes the active language to the API as a hint, and the model is free to follow
the user instead.
"""
from __future__ import annotations

import argparse
import sys

import agent_core as core
import models
import wizard
from agent_core import (
    DEFAULT_SYSTEM_PROMPT,
    RECENT_MESSAGES_AFTER_COMPACT,
    ask_yes_no,
    compact_memory,
    handle_memory_command,
)
from config import CONFIG_FILE, load_json, save_json, validate_config
from paths import (
    COMMANDS_FILE,
    LIBRARY_DIR,
    MEMORIES_DIR,
    TOOL_LOG_FILE,
    ensure_library,
)
from translator import (
    FALLBACK_LANG,
    SUPPORTED_LANGS,
    Translator,
    check_parity,
)

# Models offered by /settings live in models.py (single source of truth).
MANUAL_MODEL_INDEX = models.MANUAL_MODEL_INDEX
MANUAL_MODEL_SENTINEL = models.MANUAL_MODEL_SENTINEL
EXIT_SETTINGS_INDEX = "0"


def t(key, **kwargs):
    """Render a UI string in whatever language agent_core is currently using.

    Every lookup goes through here rather than importing the translator, so a
    language switch (``--lang``, config.json, /settings) rebinds exactly one
    object and the REPL, agent_core and api.py all follow.
    """
    return core.t(key, **kwargs)


def resolve_language(config, lang_flag):
    """Pick the active language: --lang, then config.json, then FALLBACK_LANG."""
    if lang_flag in SUPPORTED_LANGS:
        return lang_flag
    stored = config.get("language") if isinstance(config, dict) else None
    if isinstance(stored, str) and stored in SUPPORTED_LANGS:
        return stored
    return FALLBACK_LANG


def print_help():
    print(t("commands_file_name_read_text_or"))


def print_stats(memory):
    usage = memory["usage"]
    print(t("cumulative_token_usage"))
    print(t("requests_usage_get_requests_0", arg0=usage.get('requests', 0)))
    print(t("input_tokens_usage_get_prompt_tokens", arg0=usage.get('prompt_tokens', 0)))
    print(t("output_tokens_usage_get_completion_tokens", arg0=usage.get('completion_tokens', 0)))
    print(t("total_tokens_usage_get_total_tokens", arg0=usage.get('total_tokens', 0)))


def list_library():
    ensure_library()
    files = sorted(
        (p for p in LIBRARY_DIR.rglob("*") if p.is_file()),
        key=lambda p: str(p).lower(),
    )
    print(t("library_directory_library_dir", LIBRARY_DIR=LIBRARY_DIR))
    if not files:
        print(t("directory_is_empty"))
        return
    for path in files:
        try:
            print(t("path_relative_to_library_dir_as", arg0=path.relative_to(LIBRARY_DIR).as_posix(), arg1=path.stat().st_size))
        except OSError:
            print(t("path_name", arg0=path.name))


def _save_config(config):
    """Persist config.json atomically, preserving every unrelated key."""
    save_json(CONFIG_FILE, config)


def _read_choice():
    """Return the raw menu choice, or "" when the user cancels/leaves it blank."""
    return input(t("settings_choose")).strip()


def _menu_entries(entries, labels):
    """Print a numbered menu and return the value at the chosen index, else None.

    entries/labels are (index, value) and (index, label) pairs so a row can keep a
    fixed number (the manual model entry stays at 9) while presets grow.
    """
    for index, label in labels:
        print(t("settings_option_line", index=index, label=label))
    print(t("settings_exit_option"))
    raw = _read_choice()
    if raw == "" or raw == EXIT_SETTINGS_INDEX:
        return None
    if not raw.isdigit():
        print(t("settings_invalid_choice"))
        return None
    for index, value in entries:
        if str(index) == raw:
            return value
    print(t("settings_invalid_choice"))
    return None


def _apply_language(config):
    """Ask for a language; switch immediately and persist the choice."""
    codes = list(SUPPORTED_LANGS)
    entries = list(enumerate(codes, start=1))
    labels = [(index, t(f"language_name_{code}")) for index, code in entries]
    print(t("settings_language_prompt"))
    chosen = _menu_entries(entries, labels)
    if chosen is None:
        print(t("settings_unchanged"))
        return False

    if not ask_yes_no(t("settings_confirm_language", name=t(f"language_name_{chosen}"))):
        print(t("settings_cancelled"))
        return False

    config["language"] = chosen
    try:
        _save_config(config)
    except OSError as exc:
        print(t("settings_save_failed", exc=exc))
        return False

    core.t = Translator(chosen)
    print(t("settings_language_saved", name=t(f"language_name_{chosen}")))
    return True


def _apply_model(config):
    """Ask for a model; persist the choice for future sessions."""
    entries, labels = models.settings_menu(t, config.get("model"))

    print(t("settings_model_title"))
    chosen = _menu_entries(entries, labels)
    if chosen is None:
        print(t("settings_unchanged"))
        return False

    if chosen == MANUAL_MODEL_SENTINEL:
        name = input(t("settings_enter_model")).strip()
        if not name:
            print(t("settings_model_empty"))
            return False
        chosen = name

    if not ask_yes_no(t("settings_confirm_model", model=chosen)):
        print(t("settings_cancelled"))
        return False

    previous = config.get("model")
    config["model"] = chosen
    try:
        _save_config(config)
    except OSError as exc:
        config["model"] = previous
        print(t("settings_save_failed", exc=exc))
        return False
    print(t("settings_model_saved", model=chosen))
    return True


def handle_settings(config):
    """/settings: switch language or model; both persist to config.json."""
    current_lang = config.get("language") or core.t.lang
    print(t("settings_title"))
    print(t("settings_current_language",
            lang=current_lang,
            name=t(f"language_name_{current_lang}")))
    print(t("settings_current_model", model=config.get("model", "")))
    print()

    sections = ["language", "model"]
    entries = list(enumerate(sections, start=1))
    labels = [(1, t("settings_section_language")), (2, t("settings_section_model"))]
    chosen = _menu_entries(entries, labels)
    if chosen is None:
        print(t("settings_unchanged"))
        return config

    if chosen == "language":
        _apply_language(config)
    else:
        _apply_model(config)
    return config


def print_banner(config, active_memory_path, memory, max_history):
    print("=" * 48)
    print(t("deepseek_local_memory_chat_v10"))
    print(t("model_config_model", arg0=config['model']))
    print(t("current_memory_active_memory_path_stem", arg0=active_memory_path.stem, active_memory_path=active_memory_path))
    print(t("api_history_limit_max_history_messages", MAX_HISTORY_MESSAGES=max_history))
    print(t("archive_directory_memories_dir_archive", arg0=MEMORIES_DIR / 'archive'))
    print(t("history_messages_len_memory_messages", arg0=len(memory['messages'])))
    print(t("attachment_library_library_dir", LIBRARY_DIR=LIBRARY_DIR))
    print(t("tool_invocation_audit_log_tool_log", TOOL_LOG_FILE=TOOL_LOG_FILE))
    print(t("command_definitions_commands_file_re_read", COMMANDS_FILE=COMMANDS_FILE))
    print(t("enter_help_for_commands_exit_to"))
    print("=" * 48)


def _handle_memory_compact(config, memory, active_memory_path, user_text):
    parts = user_text.split()
    keep = RECENT_MESSAGES_AFTER_COMPACT
    if len(parts) == 4 and parts[2] in ("--keep", "-k"):
        try:
            keep = int(parts[3])
        except ValueError:
            print(t("usage_memory_compact_keep_n_where"))
            return
    elif len(parts) != 2:
        print(t("usage_memory_compact_keep_n_e"))
        return
    compact_memory(config, memory, active_memory_path, keep=keep)


def _handle_clear(memory, active_memory_path):
    if not ask_yes_no(t("clear_local_chat_history_cumulative_usage")):
        print(t("cancelled_no_changes_made"))
        return
    memory["messages"] = []
    memory["summary"] = ""
    memory["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }
    core.memory_store.save_memory(memory, active_memory_path)
    print(t("clear_done"))


def _handle_system(memory, active_memory_path):
    print(t("current_system_prompt"))
    print(memory.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    new_prompt = input(t("enter_a_new_system_prompt_press")).strip()
    if new_prompt:
        memory["system_prompt"] = new_prompt
        core.memory_store.save_memory(memory, active_memory_path)
        print(t("system_prompt_has_been_saved"))


def run_command(config, memory, active_memory_path, user_text):
    """Run one slash command.

    Returns ``(config, memory, active_memory_path, pending)``. ``pending`` is the
    text to send to the model, or None when the command was fully handled. A
    command may switch memory branches, so the caller must rebind all three
    values from the result -- keeping the old branch would write later turns into
    the wrong file.
    """
    if user_text == "/pwd":
        print(t("current_branch_active_memory_path_stem", arg0=active_memory_path.stem))
        print(t("memory_file_active_memory_path", active_memory_path=active_memory_path))
        return config, memory, active_memory_path, None

    if user_text == "/help":
        print_help()
        return config, memory, active_memory_path, None

    if user_text == "/stats":
        print_stats(memory)
        return config, memory, active_memory_path, None

    if user_text == "/settings":
        return handle_settings(config), memory, active_memory_path, None

    if user_text == "/files":
        list_library()
        return config, memory, active_memory_path, None

    if user_text == "/memory compact" or user_text.startswith("/memory compact "):
        _handle_memory_compact(config, memory, active_memory_path, user_text)
        return config, memory, active_memory_path, None

    if user_text.startswith("/memory"):
        active_memory_path, memory = handle_memory_command(
            user_text, active_memory_path, memory
        )
        return config, memory, active_memory_path, None

    if user_text == "/clear":
        _handle_clear(memory, active_memory_path)
        return config, memory, active_memory_path, None

    if user_text == "/system":
        _handle_system(memory, active_memory_path)
        return config, memory, active_memory_path, None

    # Not a command: fall through to the model.
    return config, memory, active_memory_path, user_text


def _build_parser(lang=None):
    """Build the argument parser with its help text in ``lang``.

    Any text baked in here is fixed for the life of the parser, so it must be
    built *after* the language is known; a parser built at import time would
    render --help in whatever language happened to be active then.
    """
    lang = lang if lang in SUPPORTED_LANGS else core.t.lang
    max_history_help = Translator(lang)("max_history_help")
    parser = argparse.ArgumentParser(description="DeepSeek local-memory chat")
    parser.add_argument(
        "--lang", default=None, choices=SUPPORTED_LANGS,
        help="UI language; overrides config.json for this run"
    )
    parser.add_argument(
        "--max-history", type=int, default=core.MAX_HISTORY_MESSAGES,
        help=max_history_help,
    )
    return parser


def _peek_lang(argv):
    """Return the --lang value from the raw tokens, without parsing anything.

    argparse handles -h/--help during parsing and exits, so a provisional parse
    would print help before the requested language was known. This scan only
    needs to find --lang ahead of any ``--`` terminator.
    """
    tokens = sys.argv[1:] if argv is None else list(argv)
    for index, token in enumerate(tokens):
        if token == "--":
            break
        if token == "--lang" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--lang="):
            return token.split("=", 1)[1]
    return None


def _parse_args(argv, lang=None):
    """Apply --lang from the command line, then parse with localised help text."""
    lang = _peek_lang(argv) or lang
    if lang in SUPPORTED_LANGS:
        core.t = Translator(lang)
    return _build_parser(lang).parse_args(argv)


def _startup():
    """Prepare everything a chat session needs.

    Raises RuntimeError/OSError with a message that can be printed as-is, so a
    broken install fails here instead of on the first message.
    """
    active_memory_path, memory = core.load_memory_or_raise()
    ensure_library()
    # Validate command definitions at startup; the file is re-read before every API call.
    core.load_command_definitions()
    core.check_runtime_files()
    return active_memory_path, memory


def main(argv=None, lang=None):
    """Run the chat REPL. Returns the process exit code.

    ``lang`` is the entry script's default (chat_v10.py passes "zh_cn"); it is
    the weakest of the three language sources.
    """
    check_parity()
    core.t = Translator(lang if lang in SUPPORTED_LANGS else FALLBACK_LANG)

    # Read config.json before parsing: without --lang it decides the language of
    # --help and of argument errors, which argparse may print and exit on.
    existing = load_json(CONFIG_FILE)
    args = _parse_args(argv, resolve_language(existing, None))
    if args.max_history < 0 or args.max_history > 1000:
        print(t("max_history_must_be_between_0"), file=sys.stderr)
        return 2
    core.MAX_HISTORY_MESSAGES = args.max_history

    try:
        # First run (no config.json, or a placeholder key) enters the setup wizard
        # instead of failing with a raw error.
        ok, _problem = validate_config(existing)
        if not ok:
            config, tr = wizard.run_wizard(core.t, ask_yes_no, lang=args.lang)
            if config is None:
                return 1
            core.t = tr
        else:
            config = existing

        # Language precedence: --lang flag, then the persisted config.json value,
        # then the fallback. Applied before startup so a startup error is already
        # in the right language.
        core.t = Translator(resolve_language(config, args.lang))

        active_memory_path, memory = _startup()
    except (RuntimeError, OSError) as exc:
        print(t("startup_failed_exc", exc=exc), file=sys.stderr)
        return 1

    print_banner(config, active_memory_path, memory, core.MAX_HISTORY_MESSAGES)

    while True:
        try:
            user_text = input(t("you")).strip()
        except (EOFError, KeyboardInterrupt):
            print(t("exited_chat_history_saved_to_active", active_memory_path=active_memory_path))
            return 0

        if not user_text:
            continue

        if user_text == "/exit":
            print(t("chat_history_saved_to_active_memory", active_memory_path=active_memory_path))
            return 0

        config, memory, active_memory_path, pending = run_command(
            config, memory, active_memory_path, user_text
        )
        if pending is None:
            continue
        core.run_turn(config, memory, active_memory_path, pending)


if __name__ == "__main__":
    raise SystemExit(main())
