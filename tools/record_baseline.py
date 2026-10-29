"""Record a behaviour baseline for the API layer before it is refactored.

Records:
  1. build_prompt_view / build_command_context output, byte for byte
  2. _parse_agent_envelope on a table of legal, illegal and edge-case payloads
  3. the exact request body _api_request sends for a scripted conversation
  4. call_api's full message list and tool-round behaviour, with the network
     stubbed out so no request is ever made and no credit is spent

The output is JSON plus a few raw text blobs, written to tests/baseline/.
Run with:  python tools/record_baseline.py
"""
import io
import json
import os
import sys
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "baseline")


def write(name, payload):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    size = os.path.getsize(path)
    print("  wrote %-42s %6d bytes" % (name, size))


# ---------------------------------------------------------------- envelope table

ENVELOPE_CASES = [
    # (label, raw model output)
    ("output_minimal", '{"type":"output","text":"hello"}'),
    ("output_empty_text", '{"type":"output","text":""}'),
    ("output_unicode", '{"type":"output","text":"\\u4f60\\u597d\\uff0c\\u4e16\\u754c"}'),
    ("output_extra_field", '{"type":"output","text":"hi","extra":1}'),
    ("tool_call_minimal",
     '{"type":"tool_call","command":"ls","arguments":{},"reason":"listing"}'),
    ("tool_call_with_args",
     '{"type":"tool_call","command":"cat","arguments":{"path":"a.txt"},"reason":"read"}'),
    ("tool_call_missing_reason",
     '{"type":"tool_call","command":"ls","arguments":{}}'),
    ("tool_call_arguments_not_object",
     '{"type":"tool_call","command":"ls","arguments":[],"reason":"r"}'),
    ("tool_call_reason_not_string",
     '{"type":"tool_call","command":"ls","arguments":{},"reason":5}'),
    ("tool_call_unknown_field",
     '{"type":"tool_call","command":"ls","arguments":{},"reason":"r","extra":true}'),
    ("tool_call_missing_command",
     '{"type":"tool_call","command":"","arguments":{},"reason":"r"}'),
    ("unknown_type", '{"type":"other","text":"x"}'),
    ("not_json", 'this is plain text, not json'),
    ("json_array", '[1,2,3]'),
    ("json_string", '"just a string"'),
    ("output_not_object", '{"type":"output"}'),
    ("markdown_fenced", '```json\n{"type":"output","text":"hi"}\n```'),
    ("trailing_text", '{"type":"output","text":"hi"} and some extra'),
]


def record_envelope(core):
    print("[2] _parse_agent_envelope")
    results = {}
    for label, raw in ENVELOPE_CASES:
        try:
            parsed = core._parse_agent_envelope(raw)
            results[label] = {"ok": True, "result": parsed}
        except Exception as exc:
            results[label] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    write("envelope_cases.json", results)
    okc = sum(1 for v in results.values() if v["ok"])
    print("    %d parsed, %d rejected" % (okc, len(results) - okc))


# ---------------------------------------------------------------- request capture

def record_request_body(core):
    print("[3] _api_request request body")

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured = {}

    def fake_open(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(b'{"choices":[{"message":{"content":"{}"}}],"usage":{"total_tokens":1}}')

    real = core.urllib.request.urlopen
    core.urllib.request.urlopen = fake_open
    try:
        config = {"api_key": "sk-BASELINE-NOT-REAL", "base_url": "https://api.deepseek.com",
                  "model": "deepseek-flash"}
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        core._api_request(config, messages)
    finally:
        core.urllib.request.urlopen = real

    # Secret-free projection: the auth header is recorded only as a shape.
    headers = {k: ("<redacted>" if k.lower() == "authorization" else v)
               for k, v in captured["headers"].items()}
    write("request_body.json", {
        "url": captured["url"],
        "method": captured["method"],
        "timeout": captured["timeout"],
        "headers": headers,
        "body": captured["body"],
    })
    print("    captured %d request keys" % len(captured["body"]))


def record_call_api(core):
    print("[4] call_api message list (network + dispatcher + audit log stubbed)")

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_response_for(raw):
        return ('{"choices":[{"message":{"content":%s}}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
                % json.dumps(raw)).encode("utf-8")

    SCRIPT = {
        # label -> list of raw model outputs to return, one per request
        "output": ['{"type":"output","text":"done"}'],
        "tool_call": ['{"type":"tool_call","command":"ls","arguments":{},"reason":"look"}',
                      '{"type":"output","text":"after tool"}'],
        "bad_then_good": ['not json at all',
                          '{"type":"output","text":"repaired"}'],
    }

    config = {"api_key": "sk-BASELINE-NOT-REAL", "base_url": "https://api.deepseek.com",
              "model": "deepseek-flash"}
    base_memory = core.memory_store.new_memory()
    base_memory["messages"] = [{"role": "user", "content": "earlier"},
                               {"role": "assistant", "content": "reply"}]

    import agent_dispatcher
    real_open = core.urllib.request.urlopen
    real_dispatch = agent_dispatcher.dispatch_tool_call
    real_log = core._update_tool_log
    real_save_output = core._save_permitted_tool_output

    out = {}
    for label, script in SCRIPT.items():
        sent = []

        def fake_open(request, timeout=None, _sent=sent, _script=script):
            _sent.append(json.loads(request.data.decode("utf-8")))
            raw = _script[min(len(_sent) - 1, len(_script) - 1)]
            return FakeResponse(fake_response_for(raw))

        def fake_dispatch(envelope, base_dir):
            return {"status": "ok", "command": envelope.get("command"),
                    "stdout": "a.txt\nb.txt\n", "stderr": "", "exit_code": 0}

        def quiet(*a, **k):
            return None

        core.urllib.request.urlopen = fake_open
        agent_dispatcher.dispatch_tool_call = fake_dispatch
        core._update_tool_log = quiet
        core._save_permitted_tool_output = lambda *a, **k: None
        try:
            memory = dict(base_memory, messages=list(base_memory["messages"]))
            answer, usage = core.call_api(config, memory, "current question")
            out[label] = {
                "answer": answer,
                "usage": usage,
                "request_count": len(sent),
                "roles_per_request": [[m.get("role") for m in b.get("messages", [])]
                                      for b in sent],
                "first_request_messages": sent[0].get("messages") if sent else [],
            }
        except Exception as exc:
            out[label] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_count": len(sent),
                "roles_per_request": [[m.get("role") for m in b.get("messages", [])]
                                      for b in sent],
            }
        finally:
            core.urllib.request.urlopen = real_open
            agent_dispatcher.dispatch_tool_call = real_dispatch
            core._update_tool_log = real_log
            core._save_permitted_tool_output = real_save_output

    write("call_api_cases.json", out)
    for label, value in out.items():
        if "error_type" in value:
            print("    %-14s error: %s" % (label, value["error_type"]))
        else:
            print("    %-14s requests=%d answer=%r" % (label, value["request_count"], value["answer"]))


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import agent_core as core

    print("[1] prompt building")
    # Pin the translator: build_command_context reads the module-level `t`, so an
    # unstable language would make the baseline non-reproducible.
    core.t = core.Translator("zh_cn")
    command_data = core.load_command_definitions()
    view = core.build_prompt_view(command_data)
    context = core.build_command_context(command_data)
    write("prompt_view.txt", view)
    write("command_context.txt", context)
    try:
        from tokenizer import token_counter
        print("    prompt_view tokens:", token_counter(view))
        print("    command_context tokens:", token_counter(context))
    except Exception as exc:
        print("    (token count unavailable: %s)" % exc)

    record_envelope(core)
    record_request_body(core)
    record_call_api(core)
    print("done")


if __name__ == "__main__":
    main()
