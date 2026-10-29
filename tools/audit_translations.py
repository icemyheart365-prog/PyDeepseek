"""Render every translation key with synthetic arguments and check the result.

translator.Translator.format swallows KeyError/IndexError/ValueError and returns
the raw template, so a misspelled placeholder leaks '{name}' straight into the
UI with no error anywhere. This audit renders each key with one unique sentinel
per placeholder and verifies:

  1. no placeholder survives rendering        (template.format actually ran)
  2. every sentinel appears in the output     (the argument name matched)
  3. no stray braces remain                   (escaped braces are balanced)
  4. both languages produce the same shape    (same set of sentinels)

Run:  python tools/audit_translations.py
Exit code is non-zero when any key fails.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from translator import SUPPORTED_LANGS, load_catalog  # noqa: E402

# {name} but not {{ or }}
PLACEHOLDER = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


def lone_brace(text):
    """Return a lone brace from text, or None.

    Doubled braces are legitimate format escapes, so strip them first; whatever
    is left is an unpaired brace that would confuse format().
    """
    stripped = text.replace("{{", "").replace("}}", "")
    match = re.search(r"[{}]", stripped)
    return match.group() if match else None


def sentinel_for(name):
    """A value that is unmistakable in the rendered output."""
    return "\u2039%s\u203a" % name


def render(template, names):
    args = {name: sentinel_for(name) for name in names}
    try:
        return template.format(**args)
    except Exception as exc:  # noqa: BLE001 - report whatever format raised
        return "<FORMAT-ERROR %s: %s>" % (type(exc).__name__, exc)


def main():
    catalogs = {lang: load_catalog(lang) for lang in SUPPORTED_LANGS}
    reference = "zh_cn"
    keys = sorted(catalogs[reference])

    failures = []
    doubled = []
    stats = {"no_placeholder": 0, "checked": 0}

    print("=" * 74)
    print("dynamic render audit: %d keys x %d languages" % (len(keys), len(catalogs)))
    print("=" * 74)

    for key in keys:
        shapes = {}
        for lang in SUPPORTED_LANGS:
            template = catalogs[lang].get(key)
            if template is None:
                failures.append((key, lang, "missing key"))
                continue
            names = sorted(set(PLACEHOLDER.findall(template)))
            shapes[lang] = tuple(names)

            if not names:
                stats["no_placeholder"] += 1
                # Translator short-circuits when no kwargs are passed, so the
                # template reaches the UI verbatim. Only an unpaired brace is a
                # defect; '{{' is a valid escape that simply stays doubled and is
                # reported separately as a doubled-brace note.
                if lone_brace(template):
                    failures.append((key, lang,
                                     "unpaired brace: %r" % template[:60]))
                elif "{{" in template or "}}" in template:
                    doubled.append((key, lang))
                continue

            stats["checked"] += 1
            rendered = render(template, names)

            if rendered.startswith("<FORMAT-ERROR"):
                failures.append((key, lang, rendered))
                continue
            if lone_brace(rendered):
                failures.append((key, lang,
                                 "placeholder survived: %r" % rendered[:80]))
                continue
            missing = [n for n in names if sentinel_for(n) not in rendered]
            if missing:
                failures.append((key, lang,
                                 "sentinel missing for %s: %r" % (missing, rendered[:80])))

        # Shape comparison: the two languages must consume the same arguments.
        if len(shapes) == len(SUPPORTED_LANGS):
            distinct = set(shapes.values())
            if len(distinct) > 1:
                failures.append((key, "shape",
                                 "placeholder sets differ: %s" % shapes))

    print()
    print("keys with placeholders : %d" % stats["checked"])
    print("keys without           : %d" % stats["no_placeholder"])
    print("failures               : %d" % len(failures))
    print("doubled-brace templates: %d" % len(doubled))
    print()

    if failures:
        print("--- failures ---")
        seen = set()
        for key, lang, message in failures:
            if (key, message) in seen:
                continue
            seen.add((key, message))
            print("  %-46s [%s] %s" % (key, lang, message))
        return 1

    print("all keys render cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
