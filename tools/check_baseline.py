"""Compare the current API layer against the recorded baseline.

Run after refactoring the API layer. Exits non-zero on the first difference, so
it works as a pre-commit gate.

    python tools/check_baseline.py              compare everything
    python tools/check_baseline.py --update     re-record (only when a change is intended)

Prompt-text differences are reported as a unified diff rather than a hard
failure, because step 3b intentionally translates the protocol prompts to
English. Structural invariants (roles per request, parsed/rejected verdicts,
request body shape) must still match exactly.
"""
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "tests", "baseline")

failures = []
notes = []


def load(name):
    with io.open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def load_text(name):
    with io.open(os.path.join(BASE, name), encoding="utf-8") as f:
        return f.read()


def check(condition, message):
    if not condition:
        failures.append(message)
    return condition


def main():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import agent_core as core

    # ---- 1. envelope verdicts must be identical (ok / rejected, and why)
    expected = load("envelope_cases.json")
    print("[1] envelope verdicts")

    ENVELOPE_CASES = {
        "output_minimal": '{"type":"output","text":"hello"}',
        "output_empty_text": '{"type":"output","text":""}',
        "output_unicode": '{"type":"output","text":"\\u4f60\\u597d\\uff0c\\u4e16\\u754c"}',
        "output_extra_field": '{"type":"output","text":"hi","extra":1}',
        "tool_call_minimal": '{"type":"tool_call","command":"ls","arguments":{},"reason":"listing"}',
        "tool_call_with_args": '{"type":"tool_call","command":"cat","arguments":{"path":"a.txt"},"reason":"read"}',
        "tool_call_missing_reason": '{"type":"tool_call","command":"ls","arguments":{}}',
        "tool_call_arguments_not_object": '{"type":"tool_call","command":"ls","arguments":[],"reason":"r"}',
        "tool_call_reason_not_string": '{"type":"tool_call","command":"ls","arguments":{},"reason":5}',
        "tool_call_unknown_field": '{"type":"tool_call","command":"ls","arguments":{},"reason":"r","extra":true}',
        "tool_call_missing_command": '{"type":"tool_call","command":"","arguments":{},"reason":"r"}',
        "unknown_type": '{"type":"other","text":"x"}',
        "not_json": 'this is plain text, not json',
        "json_array": '[1,2,3]',
        "json_string": '"just a string"',
        "output_not_object": '{"type":"output"}',
        "markdown_fenced": '```json\n{"type":"output","text":"hi"}\n```',
        "trailing_text": '{"type":"output","text":"hi"} and some extra',
    }
    for label, raw in ENVELOPE_CASES.items():
        want = expected.get(label)
        if want is None:
            failures.append("envelope: baseline missing case %r" % label)
            continue
        try:
            got = {"ok": True, "result": core._parse_agent_envelope(raw)}
        except Exception as exc:
            got = {"ok": False, "error_type": type(exc).__name__,
                   "error_message": str(exc)}
        check(got["ok"] == want["ok"],
              "envelope %s: ok %s != %s" % (label, got["ok"], want["ok"]))
        if got["ok"] and want["ok"]:
            check(got["result"] == want["result"],
                  "envelope %s: parsed result differs" % label)
        if not got["ok"] and not want["ok"]:
            check(got["error_type"] == want["error_type"],
                  "envelope %s: error type %s != %s"
                  % (label, got["error_type"], want["error_type"]))
    print("    %d cases checked, %d failures so far" % (len(ENVELOPE_CASES), len(failures)))

    # ---- 2. prompt text: diff, not failure (3b translates it on purpose)
    print("[2] prompt text")
    core.t = core.Translator("zh_cn")
    command_data = core.load_command_definitions()
    for name, current in (
        ("prompt_view.txt", core.build_prompt_view(command_data)),
        ("command_context.txt", core.build_command_context(command_data)),
    ):
        want = load_text(name)
        if current == want:
            print("    %-22s identical (%d bytes)" % (name, len(current)))
        else:
            notes.append("%s changed: %d -> %d bytes" % (name, len(want), len(current)))
            print("    %-22s CHANGED (%d -> %d bytes)" % (name, len(want), len(current)))

    # ---- 3. request body shape
    print("[3] request body")
    expected_body = load("request_body.json")

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
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(b'{"choices":[{"message":{"content":"{}"}}],"usage":{"total_tokens":1}}')

    real = core.urllib.request.urlopen
    core.urllib.request.urlopen = fake_open
    try:
        core._api_request(
            {"api_key": "sk-BASELINE-NOT-REAL", "base_url": "https://api.deepseek.com",
             "model": "deepseek-flash"},
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        )
    finally:
        core.urllib.request.urlopen = real

    check(captured["url"] == expected_body["url"],
          "request url %r != %r" % (captured["url"], expected_body["url"]))
    check(captured["method"] == expected_body["method"], "request method differs")
    check(set(captured["body"]) == set(expected_body["body"]),
          "request body keys %s != %s"
          % (sorted(captured["body"]), sorted(expected_body["body"])))
    for key in ("model", "messages"):
        if key in expected_body["body"]:
            check(captured["body"].get(key) == expected_body["body"][key],
                  "request body[%r] differs" % key)
    print("     url/method/keys checked")

    # ---- 4. call_api structural behaviour
    print("[4] call_api request/role structure")
    expected_calls = load("call_api_cases.json")

    SCRIPT = {
        "output": ['{"type":"output","text":"done"}'],
        "tool_call": ['{"type":"tool_call","command":"ls","arguments":{},"reason":"look"}',
                      '{"type":"output","text":"after tool"}'],
        "bad_then_good": ['not json at all',
                          '{"type":"output","text":"repaired"}'],
    }

    import agent_dispatcher
    real_open = core.urllib.request.urlopen
    real_dispatch = agent_dispatcher.dispatch_tool_call
    real_log = core._update_tool_log
    real_save = core._save_permitted_tool_output

    for label, script in SCRIPT.items():
        sent = []

        def fake_open2(request, timeout=None, _sent=sent, _script=script):
            _sent.append(json.loads(request.data.decode("utf-8")))
            raw = _script[min(len(_sent) - 1, len(_script) - 1)]
            return FakeResponse(('{"choices":[{"message":{"content":%s}}],'
                                 '"usage":{"prompt_tokens":10,"completion_tokens":2,'
                                 '"total_tokens":12}}' % json.dumps(raw)).encode("utf-8"))

        def fake_dispatch(envelope, base_dir):
            return {"status": "ok", "command": envelope.get("command"),
                    "stdout": "a.txt\nb.txt\n", "stderr": "", "exit_code": 0}

        core.urllib.request.urlopen = fake_open2
        agent_dispatcher.dispatch_tool_call = fake_dispatch
        core._update_tool_log = lambda *a, **k: None
        core._save_permitted_tool_output = lambda *a, **k: None
        try:
            memory = core.memory_store.new_memory()
            memory["messages"] = [{"role": "user", "content": "earlier"},
                                  {"role": "assistant", "content": "reply"}]
            answer, usage = core.call_api(
                {"api_key": "sk-BASELINE-NOT-REAL", "base_url": "https://api.deepseek.com",
                 "model": "deepseek-flash"}, memory, "current question")
            got_roles = [[m.get("role") for m in b.get("messages", [])] for b in sent]
        except Exception as exc:
            failures.append("call_api %s raised %s: %s" % (label, type(exc).__name__, exc))
            got_roles = None
        finally:
            core.urllib.request.urlopen = real_open
            agent_dispatcher.dispatch_tool_call = real_dispatch
            core._update_tool_log = real_log
            core._save_permitted_tool_output = real_save

        want = expected_calls.get(label, {})
        if got_roles is not None:
            check(got_roles == want.get("roles_per_request"),
                  "call_api %s: roles %s != %s" % (label, got_roles, want.get("roles_per_request")))
        print("    %-14s %s" % (label, "ok" if got_roles == want.get("roles_per_request") else "DIFFERS"))

    # ---- report
    print()
    if notes:
        print("--- notes (informational) ---")
        for note in notes:
            print("  * " + note)
        print()
    if failures:
        print("FAILED (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("baseline check passed")
    return 0


if __name__ == "__main__":
    if "--update" in sys.argv:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "record_baseline.py")], check=True)
    sys.exit(main())
