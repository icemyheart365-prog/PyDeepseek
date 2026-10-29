"""Loading, validating and persisting config.json.

Kept free of UI concerns except for the two startup error strings, which are
still hard-coded Chinese pending the error-code pass (step 2).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import models
from paths import CONFIG_FILE

DEFAULT_BASE_URL = "https://api.deepseek.com"

# Values that look filled-in but are still templates.
PLACEHOLDER_KEYS = {"INPUT_YOUR_API_KEY_HERE", "YOUR_API_KEY", "sk-xxx"}

REQUIRED_FIELDS = ("api_key", "base_url", "model")


def load_json(path, default=None):
    """Read UTF-8 JSON, returning ``default`` when the file is absent."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"无法读取 {path.name}：{exc}") from exc


def save_json(path, data):
    """Write UTF-8 JSON atomically enough to survive a crash mid-write."""
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def validate_config(config):
    """Return (ok, problem) without raising; used by the wizard and startup."""
    if not isinstance(config, dict):
        return False, "top_level"
    for key in REQUIRED_FIELDS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            return False, key
    if config["api_key"].strip().upper() in {p.upper() for p in PLACEHOLDER_KEYS}:
        return False, "api_key"
    if not config["api_key"].strip().startswith("sk-"):
        return False, "api_key"
    return True, None


def load_config():
    """Load config.json, raising when it is absent or invalid."""
    config = load_json(CONFIG_FILE)
    ok, problem = validate_config(config)
    if not ok:
        if problem == "top_level":
            raise RuntimeError("config.json 必须是 JSON 对象。")
        raise RuntimeError(f"config.json 缺少有效的 {problem} 字段。")
    return config


def build_endpoint(base_url):
    """Normalise a base URL into the chat-completions endpoint."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def verify_api_key(api_key, base_url):
    """Send a minimal request; return (ok, reason).

    A one-token completion is the cheapest way to prove the key, the endpoint and
    the network all work before persisting anything.
    """
    payload = {
        "model": models.DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }
    request = urllib.request.Request(
        build_endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        return True, None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        return False, f"HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        return False, f"network error: {exc.reason}"
    except (OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
