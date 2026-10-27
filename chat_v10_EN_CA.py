#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek local-memory chat (English UI).

Thin entry point: all logic lives in agent_core.py and all UI strings live in
i18n/en_ca.json. Run with --lang zh_cn to switch languages at runtime.
"""

from agent_core import main

if __name__ == "__main__":
    raise SystemExit(main("en_ca"))