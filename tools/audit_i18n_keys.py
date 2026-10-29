"""Audit every translation key: used-in-code vs present-in-json.

Reports three classes of problem:
  MISSING   the code references a key that json lacks (runtime KeyError risk)
  UNUSED    json defines a key nothing references (dead weight, or renamed)
  ASYMMETRIC the key exists in one language but not the other

Also checks format placeholders line up between the two languages, because a
template that uses {count} in one file and {total} in the other breaks at
runtime only for speakers of one language.
"""
import ast
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"tmp", ".git", "__pycache__", "library", "tests"}

# Strings that are looked up dynamically, so the literal never appears in an
# ast.Constant we can collect directly.
DYNAMIC_PREFIXES = ("language_name_",)


def iter_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py") and not name.startswith("_"):
                yield os.path.join(dirpath, name)


def collect_used_keys():
    """Return {key: [ (file, lineno), ... ]} for literal translation lookups."""
    used = {}
    call_names = {"t", "tr", "translator"}
    for path in iter_py_files():
        with io.open(path, encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            print("SKIP (syntax): %s: %s" % (path, exc))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname not in call_names:
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                used.setdefault(first.value, []).append(
                    (os.path.relpath(path, ROOT), node.lineno))
    return used


def collect_dynamic_keys():
    """Keys built with an f-string or concatenation: report, do not validate."""
    dynamic = []
    for path in iter_py_files():
        with io.open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            for prefix in DYNAMIC_PREFIXES:
                if prefix in line and "{" in line:
                    dynamic.append((os.path.relpath(path, ROOT), i, line.strip()))
                    break
    return dynamic


def placeholders(text):
    """Set of {name} placeholders in a template, ignoring {{ and }}."""
    return set(re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", text or ""))


def main():
    langs = {}
    for code in ("zh_cn", "en_ca"):
        p = os.path.join(ROOT, "i18n", "%s.json" % code)
        with io.open(p, encoding="utf-8") as f:
            langs[code] = json.load(f)

    used = collect_used_keys()
    zh, en = langs["zh_cn"], langs["en_ca"]

    print("=" * 70)
    print("key counts: zh_cn=%d  en_ca=%d  used-in-code=%d" % (len(zh), len(en), len(used)))
    print("=" * 70)

    print()
    print("--- MISSING: used in code, absent from a language file ---")
    missing = 0
    for key in sorted(used):
        for code, data in (("zh_cn", zh), ("en_ca", en)):
            if key not in data:
                where = ", ".join("%s:%d" % (f, ln) for f, ln in used[key][:3])
                print("  %-46s missing in %-6s  used at %s" % (key, code, where))
                missing += 1
    if not missing:
        print("  none")

    print()
    print("--- ASYMMETRIC: in one language only ---")
    asym = 0
    for key in sorted(set(zh) ^ set(en)):
        where = "only in zh_cn" if key in zh else "only in en_ca"
        print("  %-46s %s" % (key, where))
        asym += 1
    if not asym:
        print("  none")

    print()
    print("--- UNUSED: defined in json, never referenced by a literal lookup ---")
    shared = set(zh) & set(en)
    unused = sorted(shared - set(used))
    for key in unused:
        print("  %s" % key)
    print("  (%d keys)" % len(unused))

    print()
    print("--- PLACEHOLDER MISMATCH between languages ---")
    bad = 0
    for key in sorted(shared):
        pz, pe = placeholders(zh[key]), placeholders(en[key])
        if pz != pe:
            print("  %-46s zh=%s  en=%s" % (key, sorted(pz), sorted(pe)))
            bad += 1
    if not bad:
        print("  none")

    print()
    print("--- PLACEHOLDER vs CALL SITE (placeholder count mismatch) ---")
    # Keys where the template takes no placeholders but is called with kwargs,
    # or takes named ones that the call does not supply.
    mism = 0
    for key in sorted(set(used) & shared):
        tmpl = zh[key]
        want = placeholders(tmpl)
        if not want:
            continue
        # Only flag if the key is called at all with no kwargs anywhere; a
        # precise per-call check would need argument binding, so this is a hint.
    print("  (skipped: requires call-site argument binding; see dynamic-keys note)")

    print()
    print("--- DYNAMIC keys (built at runtime, cannot be validated statically) ---")
    dyn = collect_dynamic_keys()
    for f, ln, text in dyn:
        print("  %s:%d  %s" % (f, ln, text[:80]))
    if not dyn:
        print("  none")

    print()
    print("--- JSON files under i18n/ ---")
    for name in sorted(os.listdir(os.path.join(ROOT, "i18n"))):
        print("  %s" % name)


if __name__ == "__main__":
    main()
