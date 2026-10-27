#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek Search/Fetch helpers for the local Agent.

Standard-library only. This module does not read config files, print secrets,
modify chat history, or persist response bodies. The caller owns confirmation,
audit logging, and deciding whether to send returned text to the model.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_BASE = "https://api.deepseek.com"
SEARCH_ENDPOINT = API_BASE + "/v1/tools/search"
FETCH_ENDPOINT = API_BASE + "/v1/tools/fetch"
DEFAULT_TIMEOUT = 30
MAX_QUERY_CHARS = 128
MAX_FETCH_URL_CHARS = 8192
MAX_SEARCH_COUNT = 5
MAX_RETURN_BYTES = 24_000
USER_AGENT = "DeepSeek-Local-Agent/1.0"


class WebToolError(Exception):
    """Safe-to-return error for a Search/Fetch operation."""

    def __init__(self, message: str, *, http_status: int | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.http_status = http_status
        self.retryable = retryable


def _check_api_key(api_key: str) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise WebToolError("缺少 DeepSeek API Key；未发送网络请求。")
    key = api_key.strip()
    if any(c in key for c in "\r\n\x00"):
        raise WebToolError("API Key 格式无效；未发送网络请求。")
    return key


def _post_json(endpoint: str, api_key: str, payload: dict[str, Any],
               timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    key = _check_api_key(api_key)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            raw = response.read(MAX_RETURN_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Never include response headers or request details that could expose credentials.
        retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
        details = ""
        try:
            data = json.loads(exc.read(8192).decode("utf-8", "replace"))
            err = data.get("error", {}) if isinstance(data, dict) else {}
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                details = err["message"][:300]
        except Exception:
            pass
        suffix = f"：{details}" if details else ""
        raise WebToolError(f"DeepSeek API 返回 HTTP {exc.code}{suffix}",
                           http_status=exc.code, retryable=retryable) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        reason = getattr(exc, "reason", None)
        label = "连接超时" if isinstance(reason, (TimeoutError, socket.timeout)) or isinstance(exc, (TimeoutError, socket.timeout)) else "网络连接失败"
        raise WebToolError(f"{label}；未获得 API 结果。", retryable=True) from None

    if status < 200 or status >= 300:
        raise WebToolError(f"DeepSeek API 返回 HTTP {status}", http_status=status,
                           retryable=(status == 408 or status == 429 or status >= 500))
    if len(raw) > MAX_RETURN_BYTES:
        raise WebToolError(f"API 响应超过本地上限 {MAX_RETURN_BYTES} 字节；正文已丢弃。")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise WebToolError("API 返回的不是有效 UTF-8 JSON；响应正文已丢弃。") from None
    if not isinstance(data, dict):
        raise WebToolError("API 响应结构无效：预期 JSON 对象。")
    if "error" in data:
        err = data.get("error")
        msg = err.get("message", "未知 API 错误") if isinstance(err, dict) else "未知 API 错误"
        raise WebToolError("DeepSeek API 错误：" + str(msg)[:300])
    return data


def search(api_key: str, query: str, *, count: int = MAX_SEARCH_COUNT,
           need_content: bool = False, ori_question: str | None = None,
           timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Search the web and return a bounded, normalized result object."""
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise WebToolError(f"query 必须为 1–{MAX_QUERY_CHARS} 个字符的非空字符串。")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_SEARCH_COUNT:
        raise WebToolError(f"count 必须是 1–{MAX_SEARCH_COUNT} 的整数。")
    if not isinstance(need_content, bool):
        raise WebToolError("need_content 必须是布尔值。")
    if ori_question is not None and (not isinstance(ori_question, str) or not ori_question.strip() or len(ori_question) > MAX_QUERY_CHARS):
        raise WebToolError(f"ori_question 必须为 1–{MAX_QUERY_CHARS} 个字符，或省略。")

    payload: dict[str, Any] = {"query": query.strip(), "count": count,
                               "need_content": need_content}
    if ori_question is not None:
        payload["ori_question"] = ori_question.strip()
    data = _post_json(SEARCH_ENDPOINT, api_key, payload, timeout)
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise WebToolError("Search 响应缺少有效的 results 数组。")

    results = []
    for item in raw_results[:count]:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        url = item.get("url", "")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        # Search results are references, not trusted instructions. Keep only known fields.
        row = {
            "title": title[:500],
            "url": url[:MAX_FETCH_URL_CHARS],
            "snippet": _str_field(item.get("snippet"), 1000),
            "summary": _str_field(item.get("summary"), 4000),
            "date": _str_field(item.get("date"), 100),
            "site_name": _str_field(item.get("site_name"), 200),
            "authority": item.get("authority") if isinstance(item.get("authority"), int) else None,
            "rerank_score": item.get("rerank_score") if isinstance(item.get("rerank_score"), (int, float)) else None,
        }
        if need_content:
            row["content"] = _str_field(item.get("content"), 8000)
        results.append(row)
    return {"status": "ok", "command": "web_search", "query": query.strip(),
            "result_count": len(results), "results": results}


def _str_field(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value[:limit]


def validate_fetch_url(url: str) -> str:
    """Reject malformed URLs and obvious local/private destinations before API use.

    DeepSeek performs the remote fetch. This check is defense-in-depth, not a
    claim that client-side URL checks can replace server-side SSRF protections.
    """
    if not isinstance(url, str) or not url or len(url) > MAX_FETCH_URL_CHARS:
        raise WebToolError(f"url 必须是长度不超过 {MAX_FETCH_URL_CHARS} 的 HTTP/HTTPS URL。")
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port
    except ValueError:
        raise WebToolError("URL 格式无效。") from None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise WebToolError("只允许包含主机名的 HTTP/HTTPS URL。")
    if parts.username is not None or parts.password is not None:
        raise WebToolError("URL 不得包含用户名或密码。")
    host = parts.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        raise WebToolError("拒绝访问本机或内部域名。")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (not address.is_global):
        raise WebToolError("拒绝访问非公网 IP 地址。")
    if port is not None and not 1 <= port <= 65535:
        raise WebToolError("URL 端口无效。")
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/", parts.query, ""))


def fetch(api_key: str, url: str, *, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch one public HTTP(S) URL through DeepSeek's Fetch API."""
    safe_url = validate_fetch_url(url)
    data = _post_json(FETCH_ENDPOINT, api_key, {"url": safe_url}, timeout)
    title = data.get("title", "")
    content = data.get("content", "")
    if not isinstance(title, str) or not isinstance(content, str):
        raise WebToolError("Fetch 响应结构无效：title/content 必须是字符串。")
    if not content.strip():
        raise WebToolError("Fetch 没有返回可用网页正文；请检查 URL 或目标网站访问策略。", http_status=404)
    if len(content.encode("utf-8")) > MAX_RETURN_BYTES:
        raise WebToolError(f"网页正文超过本地上限 {MAX_RETURN_BYTES} 字节；正文已丢弃。")
    return {"status": "ok", "command": "web_fetch", "url": safe_url,
            "title": title[:1000], "content": content}


def _self_test() -> int:
    """Offline checks only; makes no network request and needs no API key."""
    cases = [
        ("https://example.com/article", True),
        ("http://127.0.0.1/admin", False),
        ("http://localhost/", False),
        ("file:///etc/passwd", False),
        ("https://user:pass@example.com/", False),
        ("http://192.168.1.1/", False),
    ]
    for value, expected in cases:
        try:
            validate_fetch_url(value)
            actual = True
        except WebToolError:
            actual = False
        assert actual == expected, (value, expected, actual)
    for count in (0, 6, True, "5"):
        try:
            search("dummy", "test", count=count)  # invalid count fails before any network request
        except WebToolError:
            pass
        else:
            raise AssertionError(f"count validation failed for {count!r}")
    try:
        search("dummy", "   ")
    except WebToolError:
        pass
    else:
        raise AssertionError("empty query validation failed")
    print("web_tools offline self-test: PASS (no network requests made)")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        raise SystemExit(_self_test())
    print("Import web_tools.search(api_key, query, ...) or web_tools.fetch(api_key, url, ...).")
    print("Offline checks: python web_tools.py --self-test")
