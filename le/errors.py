"""Compile-time error and result types for the Logical English compiler.

Marking rule: a *gap* name in this module is
INTERPRETATION -- it is the standing label for a
feature the v1 compiler refuses rather than silently drops.  The refusal
itself (a raised CompileError) is MEASURED for whatever input the test battery
exercises.
"""

from __future__ import annotations

from dataclasses import dataclass


class CompileError(ValueError):
    """A loud, attributed refusal to compile a Logical English sentence.

    ``gap`` names the LE/LPS feature that is out of scope for v1.  Standing
    rule: state the gap explicitly, never silently
    drop it.  ``sentence`` and ``index`` locate the refusal in the source, so
    the message points at the input rather than at the parser's internals.
    """

    def __init__(
        self,
        message: str,
        *,
        sentence: str | None = None,
        gap: str | None = None,
        index: int | None = None,
    ) -> None:
        self.message = message
        self.gap = gap
        self.sentence = sentence
        self.index = index
        parts = [message]
        if gap:
            parts.append(f"gap: {gap}")
        if sentence is not None:
            where = f"sentence {index}" if index is not None else "sentence"
            parts.append(f"{where}: {sentence!r}")
        super().__init__(" -- ".join(parts))


@dataclass(frozen=True)
class CompileResult:
    """The two streams (plus the reactive schema) of a compilation.

    ``rules`` is datalog-dafsa rule text, ready for ``dl_load_rules`` (or the
    CLI ``query`` command).  ``facts`` is one dl fact string per element,
    ready for the ``add_fact`` path: dl_load_rules *rejects* inline facts
    ("rule has no body"), so the two streams are kept apart by construction.

    ``reactive`` is the third artefact: one :class:`le.reactive.ReactiveRule`
    per LPS reactive rule (``If ... then ...``), in document order.  Each
    reactive rule ALSO contributes its antecedent-detection rule to ``rules``
    (the ``le_antecedent_<i>`` head), so everything the engine loads still
    lives in ``rules``; ``reactive`` holds only the DRIVER's instructions (the
    consequent schema), never engine text.  The field is a defaulted tuple so
    a declarative-only compilation (and every existing consumer of ``.rules`` /
    ``.facts``) is unchanged.
    """

    rules: str
    facts: tuple[str, ...]
    reactive: tuple = ()

    @property
    def rule_lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.rules.splitlines() if line.strip())

    def facts_text(self) -> str:
        return "".join(f"{fact}\n" for fact in self.facts)
