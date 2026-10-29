#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek local-memory chat.

Thin entry point: the REPL lives in repl.py, chat logic in agent_core.py, and all
UI strings in i18n/<lang>.json. Run with --lang en_ca to switch languages.
"""

from repl import main

if __name__ == "__main__":
    raise SystemExit(main())
