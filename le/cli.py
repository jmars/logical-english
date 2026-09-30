"""``le-compile`` -- compile a Logical English file and print dl output.

    python -m le FILE [--lexicon PATH] [--facts] [--json] [--check] [--reactive]
                      [--render]

``--lexicon`` defaults to the input path with its suffix replaced by ``.ini``.

``--check`` runs :func:`validate_dl`, a stdlib structural check of the emitted
rule text (shape, balanced parentheses, body atoms, per-predicate arity
consistency).  It does NOT load the engine: the dlb binding is imported only
by ``le/driver.py`` (lazily) and by ``tests/test_engine_roundtrip.py`` /
``tests/test_reactive_engine.py`` (which skip when it is absent), so
engine-level verification lives there.

``--reactive`` prints the JSON driver manifest (``rules``, ``facts``, the
per-rule ``reactive`` schema and the lexicon's declared ``actions``).  Without
it the JSON output keeps exactly its two keys, so existing consumers are
unchanged.  There is no ``--run-events``: running needs an open database, so
the driver is exercised through the API (and the tests), not the CLI.

``--render`` prints the REVERSE direction (le.render) instead of the dl: the
compiled streams read back as Logical English sentences.  Both streams
render -- the rules stream as ``conclusion if conditions.`` sentences (one
per rule line), the fact stream as ground sentences -- because the bridge
is bidirectional as a whole; the dl streams stay exactly what they were
(this flag only prints the LE reading, it never changes a compilation).  A
reactive document's artifact rules (``le_antecedent_*``) have no LE sentence
behind them -- their source sentence is carried by the schema in
``--reactive`` -- so such a document renders its FACTS only.  A rendering
refusal (a predicate with no declared template, an ambiguity with no
escape, a symbol with no declared surface) is a loud CompileError with its
``GAP_RENDER_*`` gap, same as every compile error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .errors import CompileError
from .lexicon import Lexicon
from .lower import compile_text

_PRED_ATOM = re.compile(r"([a-z][A-Za-z0-9_]*)\s*\((.*)\)\s*\Z", re.S)
_OPERATOR_ATOM = re.compile(
    r"[A-Za-z0-9_]+\s*(=|!=|<=|>=|<|>)\s*[A-Za-z0-9_+\-*/%() ]+\Z"
)


def _split_top_commas(text: str):
    """Split on commas that are not inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _check_predicate_atom(atom: str, lineno: int, arities: dict, problems: list,
                          exempt=frozenset()):
    match = _PRED_ATOM.match(atom)
    if not match:
        problems.append(f"line {lineno}: malformed body atom {atom!r}")
        return
    pred, args = match.group(1), match.group(2).strip()
    if "(" in args or ")" in args:
        problems.append(f"line {lineno}: nested parentheses in {atom!r}")
        return
    arity = 0 if not args else len(_split_top_commas(args))
    if pred in arities and arities[pred] != arity and pred not in exempt:
        problems.append(
            f"line {lineno}: predicate '{pred}' used with arity {arity} and "
            f"{arities[pred]}"
        )
    else:
        arities[pred] = arity


def validate_dl(rules_text: str, exempt=frozenset()) -> tuple:
    """Return a tuple of problems with ``rules_text`` (empty tuple = OK).

    A structural check of the emitted rule text only: it does NOT load the
    engine (dlb is imported only by tests/test_engine_roundtrip.py).

    ``exempt`` names predicates that ONE predicate may carry with several
    arities.  The engine's variadic relations are the only such predicates and
    the lexicon's ``[meta]`` reifying relations are the only variadic ones
    (``states(<subject>, <pred>, <args...>)`` -- one column count per mentioned
    predicate), so the CLI passes ``lexicon.meta_preds``.  The default empty
    set keeps the one-arity-per-predicate rule for everything else.
    """
    problems = []
    arities: dict = {}
    lines = rules_text.splitlines()
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.endswith("."):
            problems.append(f"line {lineno}: does not end with '.'")
            continue
        body_text = stripped[:-1]
        if body_text.count("(") != body_text.count(")"):
            problems.append(f"line {lineno}: unbalanced parentheses")
            continue
        if body_text.count(":-") != 1:
            problems.append(f"line {lineno}: expected exactly one ':-'")
            continue
        head, body = (part.strip() for part in body_text.split(":-", 1))
        if head.startswith("!"):
            problems.append(f"line {lineno}: the head is negated")
        _check_predicate_atom(head, lineno, arities, problems, exempt)
        atoms = [a.strip() for a in _split_top_commas(body)]
        if not atoms or all(not a for a in atoms):
            problems.append(f"line {lineno}: empty body")
            continue
        positives = 0
        for atom in atoms:
            if not atom:
                problems.append(f"line {lineno}: empty body atom")
                continue
            negated = atom.startswith("!")
            core = atom[1:].strip() if negated else atom
            if _PRED_ATOM.match(core):
                _check_predicate_atom(core, lineno, arities, problems, exempt)
                if not negated:
                    positives += 1
            elif _OPERATOR_ATOM.match(core):
                continue
            else:
                problems.append(f"line {lineno}: malformed body atom {atom!r}")
        if not positives:
            problems.append(
                f"line {lineno}: no positive predicate atom in the body"
            )
    return tuple(problems)


def reactive_manifest(result, lexicon: Lexicon) -> list:
    """The per-rule driver schema of a compilation, as plain JSON values."""
    return [
        {
            "index": rule.index,
            "antecedent_pred": rule.antecedent_pred,
            "head_vars": list(rule.head_vars),
            "event_preds": list(rule.event_preds),
            "consequents": [
                {
                    "kind": consequent.kind,
                    "pred": consequent.pred,
                    "args": [arg.text for arg in consequent.args],
                }
                for consequent in rule.consequents
            ],
        }
        for rule in result.reactive
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="le-compile",
        description="Compile Logical English into datalog-dafsa rules + facts.",
    )
    parser.add_argument("file", help="Logical English source file")
    parser.add_argument(
        "--lexicon",
        default=None,
        help="lexicon file (default: the input path with a .ini suffix, else "
        "lexicon.ini beside the input)",
    )
    parser.add_argument(
        "--facts", action="store_true", help="also print the fact stream"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print {'rules': ..., 'facts': [...]} instead of raw text",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="structurally check the emitted rules and report the result",
    )
    parser.add_argument(
        "--reactive",
        action="store_true",
        help="print the JSON driver manifest (rules, facts, the reactive "
        "schema and the declared actions)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="print the compiled streams read back as Logical English "
        "(le.render) instead of the dl",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    lexicon_path = args.lexicon
    if lexicon_path is None:
        sibling = os.path.splitext(args.file)[0] + ".ini"
        beside = os.path.join(os.path.dirname(os.path.abspath(args.file)), "lexicon.ini")
        lexicon_path = sibling if os.path.exists(sibling) else beside
    try:
        lexicon = Lexicon.load(lexicon_path)
    except OSError as exc:
        print(f"le-compile: cannot read lexicon: {exc}", file=sys.stderr)
        return 1
    try:
        with open(args.file, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"le-compile: cannot read input: {exc}", file=sys.stderr)
        return 1

    try:
        result = compile_text(text, lexicon)
    except CompileError as exc:
        print(f"le-compile: {exc}", file=sys.stderr)
        return 1

    if args.check:
        problems = validate_dl(result.rules, lexicon.meta_preds)
        for problem in problems:
            print(f"le-compile: check: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(
            f"check: OK ({len(result.rule_lines)} rule(s), "
            f"{len(result.facts)} fact(s))",
            file=sys.stderr,
        )

    if args.json or args.reactive:
        payload = {"rules": result.rules, "facts": list(result.facts)}
        if args.reactive:
            payload["reactive"] = reactive_manifest(result, lexicon)
            payload["actions"] = list(
                dict.fromkeys(tpl.pred for tpl in lexicon.actions)
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.render:
        from .render import render_facts, render_rule

        # The reactive artifact rules (le_antecedent_*) have no LE sentence
        # behind them -- their source sentence is carried by the schema in
        # --reactive -- so a document that compiled reactive rules renders
        # its FACTS only.  Everything else renders both streams.
        antecedent_preds = {rule.antecedent_pred for rule in result.reactive}
        for line in result.rule_lines:
            if line.split("(", 1)[0].strip() in antecedent_preds:
                continue
            print(render_rule(line, lexicon))
        for sentence in render_facts(result.facts, lexicon):
            print(sentence)
        return 0

    sys.stdout.write(result.rules)
    if args.facts:
        sys.stdout.write(result.facts_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
