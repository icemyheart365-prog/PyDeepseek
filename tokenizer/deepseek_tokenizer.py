#!/usr/bin/env python3
"""Tokenizer sanity check (requires the optional `transformers` package).

Run from this directory:
    pip install transformers
    python deepseek_tokenizer.py
"""
import os

import transformers

CHAT_TOKENIZER_DIR = os.path.dirname(os.path.abspath(__file__))

tokenizer = transformers.AutoTokenizer.from_pretrained(
    CHAT_TOKENIZER_DIR, trust_remote_code=True
)

result = tokenizer.encode("Hello!")
print(result)
