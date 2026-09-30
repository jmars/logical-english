"""Shared fixtures: file paths and small inline lexicons."""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _import_dlb():
    """Return ``(dlb_module, error)`` for the engine binding, or ``(None, why)``.

    A plain import first; then along ``$DLB_PATH``, for a binding that lives
    outside ``sys.path``.  A namespace husk -- a directory named ``dlb``
    shadowing the real package (a dlb checkout's root is itself so named)
    imports "successfully" but carries no engine API, so it is dropped and
    re-imported along ``$DLB_PATH``.
    """
    error = None
    for attempt in ("plain", "DLB_PATH"):
        if attempt == "DLB_PATH":
            dlb_path = os.environ.get("DLB_PATH")
            if not dlb_path:
                break
            if dlb_path not in sys.path:
                sys.path.insert(0, dlb_path)
            sys.modules.pop("dlb", None)  # drop a husk so the import re-walks
        try:
            module = importlib.import_module("dlb")
            if not hasattr(module, "Db"):  # a husk, not the binding
                raise ImportError("module 'dlb' exposes no engine API (shadowed?)")
            return module, None
        except Exception as exc:  # pragma: no cover - environment dependent
            error = exc
    return None, error


dlb, _DLB_ERR = _import_dlb()
DLB_IMPORT_ERROR = _DLB_ERR

from le import Lexicon  # noqa: E402  (after the sys.path bootstrap)

EXAMPLES = os.path.join(ROOT, "examples")
LEXICON_INI = os.path.join(EXAMPLES, "lexicon.ini")
LEGAL_INI = LEXICON_INI  # the historical name used by the tests
LEGAL = Lexicon.load(LEXICON_INI)

# Kowalski 2020 example (1): the rock-paper-scissors reactive fixture.
RPS_INI = os.path.join(EXAMPLES, "rps.ini")
RPS_LE = os.path.join(EXAMPLES, "rps.le")
RPS = Lexicon.load(RPS_INI)
RPS_SENTENCE = (
    "If a player P1 plays a choice C1 and another player P2 plays a choice C2 "
    "and C1 beats C2 and it is not the case that RpsGame is over then P1 "
    "receives RpsPrize and it becomes the case that RpsGame is over."
)
RPS_RULE = (
    "le_antecedent_1(P1, C1, P2, C2) :- plays(P1, C1), plays(P2, C2), "
    "beats(C1, C2), !over(rpsgame), P2 != P1.\n"
)

# A minimal lexicon with a declared 'before' predicate template.
DAY_INI = """
[sorts]
day = day

[names]
Wednesday = day
Monday = day
Tuesday = day

[predicates]
<dayA> is before <dayB> | before | day, day
<a day> is a delivery day | delivery_day | day
"""
DAY = Lexicon.from_text(DAY_INI)

# Same sorts, but 'is before' / 'is after' / 'is on or after' are left to the
# built-in comparison copulas (nothing declares them here).
DAY_COPULA_INI = """
[sorts]
day = day

[names]
Wednesday = day
Monday = day
Tuesday = day

[predicates]
<a day> is a delivery day | delivery_day | day
<dayA> precedes <dayB> | precedes | day, day
"""
DAY_COPULA = Lexicon.from_text(DAY_COPULA_INI)
