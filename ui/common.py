"""Shared terminal UI primitives.

Three layers, deliberately separated:

``ask_choice``
    Pure protocol: render a prompt, loop until the answer is recognised, return
    the normalised value. Knows nothing about i18n beyond the strings it is
    handed.

``confirm``
    A yes/no question. The *question* is passed in already complete, so callers
    never compose it into a template. The old ``ask_yes_no(action)`` embedded a
    fragment into ``是否{action}？``, which silently produced doubled text when a
    caller passed a full question.

``preview_then_confirm``
    Show a block of output, then ask. Used by flows that print a cost estimate
    or a file list before the user decides.
"""
from __future__ import annotations

# The one accepted-answer table. Kept module-level so tests can assert on it.
YES = ("y", "yes")
NO = ("", "n", "no")


def ask_choice(prompt, accept, default, on_invalid=None):
    """Render ``prompt`` in a loop and return the value mapped by ``accept``.

    ``accept`` maps a lower-cased answer to the value returned. ``default`` is
    returned for a blank line. ``on_invalid`` is called (and should print) when
    the answer is not recognised; the loop then repeats.
    """
    while True:
        answer = input(prompt).strip().casefold()
        if answer in accept:
            return accept[answer]
        if not answer:
            return default
        if on_invalid is not None:
            on_invalid()


def confirm(question, default=False, invalid_message=None):
    """Ask a complete yes/no question.

    ``question`` must already be a finished sentence: callers pass the result of
    a translator lookup, never a fragment to be embedded in a template.
    """
    prompt = "%s [y/N] " % question
    accept = {answer: True for answer in YES}
    accept.update({answer: False for answer in NO})

    def _invalid():
        if invalid_message is not None:
            print(invalid_message)

    return ask_choice(prompt, accept, default, on_invalid=_invalid)


def preview_then_confirm(render, question, default=False, invalid_message=None):
    """Print a preview block, then ask for confirmation."""
    render()
    return confirm(question, default=default, invalid_message=invalid_message)
