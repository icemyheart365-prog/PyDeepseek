#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek local-memory chat (Chinese UI).

Thin entry point: all logic lives in agent_core.py and all UI strings live in
i18n/zh_cn.json. Run with --lang en_ca to switch languages at runtime.
"""

from agent_core import main

if __name__ == "__main__":
    raise SystemExit(main("zh_cn"))