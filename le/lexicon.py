"""The declared, CLOSED noun/predicate set that drives the compiler.

Why this module is load-bearing (INTERPRETATION): LE's grammar is not the
binding constraint -- the vocabulary is.  Unconstrained, a generator invents
predicate names per sentence; invented predicates carry no semantics, so the
compilation yields nothing checkable and the vocabulary drifts every turn.  A
predicate that is not declared here is therefore a hard CompileError naming
the ontology-extension gap: state the gap explicitly; never silently drop.

File format (INI-ish; this parser is deliberately tiny):

    # comments and blank lines
    [sorts]
    transaction = transaction      # noun surface -> sort symbol (identity ok)
    day                            # bare noun = identity mapping

    [names]
    AcmeTransaction = transaction  # proper noun -> its declared sort
    Wednesday = day

    [predicates]
    <a transaction> commences on <a day> | commences | transaction, day
    <dayA> is the day before <dayB>      | before    | day, day

    [actions]
    <a player> plays <a choice>          | plays     | player, choice

    [meta]
    <a confirmation> of <a transaction> states that <atom>
                                         | states    | confirmation, transaction, _

A template is literal words interleaved with ``<slot>`` markers.  Slot markers
are *names for slots*; the declared argument sorts on the right of the ``|``
(comma separated, in slot order, ``_`` = any sort) are what type-checks a
compilation.  Comparison copulas (``is before``, ``is on or after``,
``is <N> days before``) are built in, NOT declared here -- a lexicon predicate
may not be a comparison operator.

``[names]`` declares the proper nouns (symbol constants) each with their sort.
Without it a constant would carry no sort, so a wrong-sort proper noun would
slip into a slot unchecked -- a silent misread.  A proper noun used in a
sentence but not declared here is therefore a hard error, and a declared one
type-checks exactly like a variable (its sort must equal the slot's).

``[actions]`` declares the EXTERNAL EVENTS (the reactive vocabulary): a
predicate declared here is an action the world performs and the driver
CONSUMES (deletes) once a rule that read it has fired.  Membership is the
single switch that separates "static knowledge / state" (``[predicates]``),
which merely holds, from "event" (``[actions]``), which is consumed.  The same
predicate may not be declared in both sections -- that is a loud error rather
than a guess about which reading was meant.

``[meta]`` declares the META-LEVEL verb phrases -- the embedding of an
object-level sentence inside a higher-order sentence (Kowalski 2020, paper
example 2: "a confirmation of the transaction states that the transaction is
governed by IsdaAgreement").  One entry is

    <subject slots> <verb phrase> that <atom> | pred | <subject sorts>, _

i.e. every slot of a normal template EXCEPT the last, which must be the
single ``<atom>`` slot: the MENTIONED object-level sentence.  The verb phrase
is the run of literal words immediately before that slot; its last word must
be the literal ``that`` (which is reserved everywhere else in the lexicon,
because it is also the LE relative-clause marker).  ``pred`` names the
REIFYING relation the mention lowers to -- the paper's "is governed by" is
``governs`` USED as a predicate in the conclusion and MENTIONED as a term
inside ``states(...)``, so the mention becomes

    states(<subject terms>, <the mentioned template's pred>, <its arguments>)

a flat, function-free row (the target engine has no compound terms).  Only
the ``pred`` column names the mentioned predicate: the mention's own ARGUMENT
sorts come from the template it mentions, so the trailing ``_`` slot sort
types nothing.  Because different mentions have different arities, the
relation is declared VARIADIC in the engine (see ``le.cli.validate_dl``,
which exempts it from the one-arity-per-predicate rule).

An object-level predicate is mentioned by its DECLARED template, so a meta
condition is matched with the same closed lexicon as everything else: a
mention of an undeclared predicate is refused, never invented.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .errors import CompileError

# Fixed article/ordinal table (adjective/ordinal table is FIXED).
ORDINALS = {"first": 1, "second": 2, "third": 3}
ARTICLES = ("a", "an", "the")

COMPARISON_OPS = frozenset({"<", "<=", ">", ">=", "=", "!="})

_IDENT = re.compile(r"[a-z][a-z0-9_]*\Z")
# Explicit LE variable names are legal-text style: a letter followed by digits
# (P1, C2, T57).  The shape is deliberately disjoint from proper nouns
# (letters only) so the two can never be confused.
_EXPLICIT_NAME = re.compile(r"[A-Za-z][0-9]+\Z")
_SLOT_RE = re.compile(r"<([^<>]*)>")
_AUTO_VAR = re.compile(r"V[0-9]+\Z")

# Words that can never be read as a proper-noun constant inside a slot.
_RESERVED = frozenset(
    {
        "a", "an", "the", "if", "and", "then", "when", "it", "that", "not",
        "case", "other", "another", "of", "is", "are", "or", "before",
        "after", "days", "becomes", "states", "which", "as",
    }
)

GAP_LEXICON = "ontology extension (the declared predicate set is closed)"
GAP_AMBIGUOUS_TEMPLATE = "ambiguous template match"
GAP_IDENTIFIER = (
    "declared predicate names must be dl identifiers (lowercase-initial: the "
    "engine parses an uppercase head as a variable)"
)
GAP_UNDECLARED_NAME = (
    "a proper noun whose sort is not declared in the lexicon's [names] "
    "section (ontology typing)"
)
GAP_RESERVED_WORD = (
    "a reserved connective in a template ('and' / 'if' are part of the LE "
    "sentence grammar)"
)
GAP_ACTION_OVERLAP = (
    "a predicate declared in both [predicates] and [actions] -- static "
    "knowledge/state and a consumable external event are different readings "
    "of the same template"
)
GAP_META_SHAPE = (
    "the meta-level 'states that' embedding -- the form is '<subject noun "
    "phrases> <verb phrase> that <a DECLARED predicate atom>', mentioning an "
    "object-level sentence (paper example 2)"
)
GAP_META_ARITY = (
    "a meta-level mention whose reified relation would need more than the "
    "engine's 8 columns (subject slots + 1 predicate column + the mentioned "
    "arguments)"
)
GAP_META_VARIABLE_PRED = (
    "a meta-level mention of a VARIABLE predicate -- the mentioned predicate "
    "must be a declared template, whose arity fixes the reified row's shape"
)


def is_explicit_name(token: str) -> bool:
    return bool(_EXPLICIT_NAME.match(token))


def is_auto_var(token: str) -> bool:
    return bool(_AUTO_VAR.match(token))


def is_proper_noun(token: str) -> bool:
    """A capitalised, letters-only token: a proper-noun symbol constant."""
    return token.isalpha() and token[0].isupper()


@dataclass(frozen=True)
class NP:
    """A noun phrase as read off the token stream.

    ``art`` is one of ``a``, ``the``, ``another``, ``the_other`` or ``None``
    (bare ordinal).  ``proper`` is set for a symbol constant; ``name`` is set
    for an explicit LE variable name.  ``bare`` marks a common noun with no
    determiner -- parsed so that binding can refuse it loudly rather than the
    template matcher silently failing.
    """

    art: str | None
    ordinal: str | None
    noun: str | None
    name: str | None
    proper: str | None
    text: str
    bare: bool = False


@dataclass(frozen=True)
class Template:
    """A declared predicate template: literal words plus ``<slot>`` markers."""

    elements: tuple  # ('lit', word) | ('slot', marker)
    pred: str
    sorts: tuple
    source: str

    @property
    def slots(self) -> tuple:
        return tuple(v for k, v in self.elements if k == "slot")

    @property
    def text(self) -> str:
        return " ".join(
            f"<{v}>" if k == "slot" else v for k, v in self.elements
        )


@dataclass(frozen=True)
class Match:
    """The result of matching a whole atom against the lexicon."""

    kind: str  # 'template' | 'cmp' | 'arith'
    text: str
    template: Template | None = None
    nps: tuple = ()
    op: str = ""
    left: NP | None = None
    right: NP | None = None
    n: int | None = None


def _parse_template(text: str, line_no: int) -> tuple:
    elements = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i)
            if j < 0:
                raise CompileError(
                    f"line {line_no}: unterminated '<' slot marker in template"
                )
            marker = text[i + 1 : j].strip()
            if not marker:
                raise CompileError(f"line {line_no}: empty '<>' slot marker")
            elements.append(("slot", marker))
            i = j + 1
            if i < len(text) and not text[i].isspace():
                raise CompileError(
                    f"line {line_no}: slot marker must be separated by spaces"
                )
        else:
            j = text.find("<", i)
            if j < 0:
                j = len(text)
            for word in text[i:j].split():
                elements.append(("lit", word.lower()))
            i = j
    return tuple(elements)


def _parse_template_line(line: str, line_no: int, section: str) -> Template:
    """Parse one ``<template> | pred | sort, sort`` declaration line."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 3:
        raise CompileError(
            f"line {line_no}: [{section}] entry needs "
            "'<template> | pred | sort, sort' (3 '|' fields)"
        )
    tpl_text, pred, sort_text = parts
    if pred in COMPARISON_OPS:
        raise CompileError(
            f"line {line_no}: comparison operator '{pred}' is "
            "built in and cannot be declared in the lexicon",
            gap=GAP_IDENTIFIER,
        )
    if not _IDENT.match(pred):
        raise CompileError(
            f"line {line_no}: predicate '{pred}' is not a dl identifier",
            gap=GAP_IDENTIFIER,
        )
    sorts_list = tuple(s.strip() for s in sort_text.split(",") if s.strip())
    elements = _parse_template(tpl_text, line_no)
    for kind, value in elements:
        # 'then' is reserved too: it separates a reactive rule's two halves, so
        # a template containing it could not be told apart from the split.
        # 'that' is reserved by the relative-clause test in le.lower -- the one
        # legal occurrence is the final word of a [meta] verb phrase, checked
        # (with the rest of the [meta] shape) below.
        if kind == "lit" and (
            value in ("and", "if", "then")
            or (value == "that" and section != "meta")
        ):
            raise CompileError(
                f"line {line_no}: '{value}' is reserved by the LE "
                "sentence grammar and cannot appear in a template",
                gap=GAP_RESERVED_WORD,
            )
    slots = [v for k, v in elements if k == "slot"]
    if not slots:
        raise CompileError(
            f"line {line_no}: template '{tpl_text}' declares no <slot> marker"
        )
    if len(slots) != len(sorts_list):
        raise CompileError(
            f"line {line_no}: template '{tpl_text}' has {len(slots)} slot(s) "
            f"but {len(sorts_list)} declared sort(s)"
        )
    if section == "meta":
        _check_meta_template(elements, sorts_list, tpl_text, line_no)
    return Template(elements, pred, sorts_list, tpl_text)


def _check_meta_template(elements, sorts_list, text: str, line_no: int) -> None:
    """Enforce the ``[meta]`` shape: ``<subject slots> <verb phrase> that <atom>``.

    The trailing ``<atom>`` slot is the MENTIONED object-level sentence; the
    literal run before it is the meta verb phrase and must end in ``that``.
    A slot (rather than a fixed verb phrase) in that position would make the
    mentioned PREDICATE a variable -- the row's shape would then depend on a
    value at run time, which a function-free engine cannot express -- so it is
    refused by name (GAP_META_VARIABLE_PRED).
    """
    if not elements or elements[-1][0] != "slot":
        raise CompileError(
            f"line {line_no}: [meta] template '{text}' must END with the "
            "<atom> slot -- the mentioned object-level sentence",
            gap=GAP_META_SHAPE,
        )
    if sorts_list[-1] != "_":
        raise CompileError(
            f"line {line_no}: [meta] template '{text}': the trailing <atom> "
            f"slot declares sort '{sorts_list[-1]}'; it must be '_' (the "
            "mentioned atom is typed by the template it mentions)",
            gap=GAP_META_SHAPE,
        )
    words = []
    for kind, value in reversed(elements[:-1]):
        if kind != "lit":
            break
        words.append(value)
    connective = tuple(reversed(words))
    if not connective:
        raise CompileError(
            f"line {line_no}: [meta] template '{text}' has no fixed verb "
            "phrase before the <atom> slot; the mentioned predicate must be a "
            "declared template, not a variable",
            gap=GAP_META_VARIABLE_PRED,
        )
    if connective[-1] != "that":
        raise CompileError(
            f"line {line_no}: [meta] template '{text}': the verb phrase "
            f"before the <atom> slot ends in '{connective[-1]}'; it must end "
            "in the literal 'that'",
            gap=GAP_META_SHAPE,
        )
    for position, (kind, value) in enumerate(elements):
        if kind == "lit" and value == "that" and position != len(elements) - 2:
            raise CompileError(
                f"line {line_no}: [meta] template '{text}': 'that' may only be "
                "the last word of the verb phrase before the <atom> slot",
                gap=GAP_RESERVED_WORD,
            )


class Lexicon:
    """The declared sorts, proper nouns, predicate, action and meta templates."""

    def __init__(
        self,
        sorts: dict,
        templates: tuple,
        name: str | None = None,
        names: dict | None = None,
        actions: tuple = (),
        meta: tuple = (),
        render_canonical: dict | None = None,
    ):
        self.sorts = dict(sorts)
        self.templates = tuple(templates)
        self.actions = tuple(actions)
        self.meta = tuple(meta)
        self.name = name
        self.names = dict(names or {})
        # The reverse renderer's canonical-template marking ([render], an
        # ADDITIVE section: absent = every pre-[render] lexicon, unchanged).
        self.render_canonical = dict(render_canonical or {})
        # The predicates an action template can be performed as (the driver's
        # consumable event vocabulary).
        self.action_preds = frozenset(tpl.pred for tpl in self.actions)
        # The REIFYING relations a meta condition lowers to.  These rows carry
        # one column per mention of a different arity, so the engine declares
        # them variadic and the CLI's arity check exempts them.
        self.meta_preds = frozenset(tpl.pred for tpl in self.meta)
        # A literal word can never be read as a proper-noun constant; the meta
        # verb phrase words ('states', 'that', any further declared one) are
        # therefore reserved alongside the predicate/action literals.
        self.literals = frozenset(
            v
            for tpl in self.all_templates + self.meta
            for k, v in tpl.elements
            if k == "lit"
        )

    @property
    def all_templates(self) -> tuple:
        """Every template a sentence's ATOM may match, predicates first.

        ``[meta]`` templates are deliberately NOT here: a meta condition is
        tried only after :func:`match_atom` has failed for the whole span, so
        every non-meta sentence takes exactly the path it took before the
        meta embedding existed.
        """
        return self.templates + self.actions

    def is_action(self, pred: str) -> bool:
        """True when ``pred`` is declared in the lexicon's ``[actions]``."""
        return pred in self.action_preds

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path) -> "Lexicon":
        with open(os.fspath(path), encoding="utf-8") as handle:
            return cls.from_text(handle.read(), name=os.fspath(path))

    @classmethod
    def from_text(cls, text: str, name: str | None = None) -> "Lexicon":
        sorts: dict = {}
        names: dict = {}
        render_canonical: dict = {}
        raw_templates: list = []
        action_templates: list = []
        meta_templates: list = []
        section = None
        for line_no, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                if section not in (
                    "sorts", "names", "predicates", "actions", "meta",
                    "render",
                ):
                    raise CompileError(
                        f"line {line_no}: unknown section '[{section}]' "
                        "(expected [sorts], [names], [predicates], [actions], "
                        "[meta] or [render])"
                    )
                continue
            if section is None:
                raise CompileError(
                    f"line {line_no}: content before any section header"
                )
            if section in ("sorts", "names"):
                if "=" not in line:
                    if section == "names":
                        raise CompileError(
                            f"line {line_no}: [names] entry needs "
                            "'Name = sort'"
                        )
                    noun = sort = line.strip()
                else:
                    noun, _, sort = line.partition("=")
                    noun, sort = noun.strip(), sort.strip()
                if not noun or not sort:
                    raise CompileError(
                        f"line {line_no}: malformed [{section}] entry"
                    )
                if section == "names":
                    if not is_proper_noun(noun):
                        raise CompileError(
                            f"line {line_no}: [names] entry '{noun}' is not a "
                            "proper noun (a capitalised, letters-only token)"
                        )
                    if noun in names and names[noun] != sort:
                        raise CompileError(
                            f"line {line_no}: proper noun '{noun}' declared "
                            f"twice with different sorts ('{names[noun]}' and "
                            f"'{sort}')"
                        )
                    names[noun] = sort
                    continue
                if noun in sorts and sorts[noun] != sort:
                    raise CompileError(
                        f"line {line_no}: noun '{noun}' declared twice with "
                        f"different sorts ('{sorts[noun]}' and '{sort}')"
                    )
                sorts[noun] = sort
            elif section == "render":
                # ADDITIVE: the reverse renderer's canonical-template
                # marking.  Old inis have no [render] section and parse
                # byte-identically without it; a line here only narrows the
                # reverse renderer's ambiguity error into a canonical pick.
                if "=" not in line:
                    raise CompileError(
                        f"line {line_no}: [render] entry needs "
                        "'pred = <template>'"
                    )
                pred, _, template_text = line.partition("=")
                render_canonical[pred.strip()] = template_text.strip()
            else:
                template = _parse_template_line(line, line_no, section)
                if section == "actions":
                    action_templates.append(template)
                elif section == "meta":
                    meta_templates.append(template)
                else:
                    raw_templates.append(template)

        if not sorts:
            raise CompileError("lexicon declares no [sorts]")
        if not raw_templates:
            raise CompileError("lexicon declares no [predicates]")

        declared = set(sorts.values())
        for tpl in raw_templates + action_templates + meta_templates:
            for want in tpl.sorts:
                if want == "_":
                    continue
                if want not in declared:
                    raise CompileError(
                        f"template '{tpl.text}' declares sort '{want}', which "
                        "no [sorts] entry maps to"
                    )
        for proper, sort in names.items():
            if sort not in declared:
                raise CompileError(
                    f"[names] entry '{proper}' declares sort '{sort}', which "
                    "no [sorts] entry maps to"
                )

        declared_sections = (
            ("predicates", raw_templates),
            ("actions", action_templates),
            ("meta", meta_templates),
        )
        seen: dict = {}
        declared_in: dict = {}
        for section_name, templates in declared_sections:
            for tpl in templates:
                key = (tpl.text, tpl.pred)
                if key in seen:
                    if seen[key] != section_name:
                        raise CompileError(
                            f"template '{tpl.text}' is declared in both "
                            f"[{seen[key]}] and [{section_name}]",
                            gap=GAP_ACTION_OVERLAP,
                        )
                    raise CompileError(
                        f"template '{tpl.text}' is declared twice",
                        gap=GAP_AMBIGUOUS_TEMPLATE,
                    )
                seen[key] = section_name
                previous = declared_in.get(tpl.pred)
                if previous is not None and previous != section_name:
                    raise CompileError(
                        f"predicate '{tpl.pred}' is declared in both "
                        f"[{previous}] and [{section_name}]",
                        gap=GAP_ACTION_OVERLAP,
                    )
                declared_in[tpl.pred] = section_name

        for pred, template_text in render_canonical.items():
            matches = [
                t
                for t in raw_templates + action_templates
                if t.pred == pred and t.source == template_text
            ]
            if not matches:
                declared = (
                    "/".join(sorted({t.source for t in raw_templates + action_templates if t.pred == pred}))
                    or "none"
                )
                raise CompileError(
                    f"the [render] canonical template '{template_text}' for "
                    f"'{pred}' is not a declared template for that predicate "
                    f"(declared: {declared})"
                )
        return cls(
            sorts,
            tuple(raw_templates),
            name=name,
            names=names,
            actions=tuple(action_templates),
            meta=tuple(meta_templates),
            render_canonical=render_canonical,
        )

    # -- queries -----------------------------------------------------------

    def sort_of(self, noun: str) -> str | None:
        return self.sorts.get(noun)

    def sort_of_name(self, proper: str) -> str | None:
        """The declared sort of a proper noun, or None if undeclared."""
        return self.names.get(proper)

    def nouns(self) -> tuple:
        return tuple(sorted(self.sorts))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def parse_np_span(span, lexicon: Lexicon) -> NP | None:
    """Read ``span`` as exactly one noun phrase, or return None."""
    n = len(span)
    if n < 1 or n > 4:
        return None
    if n == 1:
        token = span[0]
        low = token.lower()
        if low in lexicon.sorts:
            return NP(None, None, low, None, None, token, bare=True)
        if is_explicit_name(token):
            return NP(None, None, None, token, None, token)
        if (
            is_proper_noun(token)
            and low not in lexicon.literals
            and low not in _RESERVED
        ):
            return NP(None, None, None, None, token, token)
        return None

    i = 0
    art: str | None = None
    ordinal: str | None = None
    name: str | None = None
    low0 = span[0].lower()
    if low0 in ("a", "an"):
        art = "a"
        i = 1
    elif low0 == "the":
        art = "the"
        i = 1
        if i < n and span[i].lower() == "other":
            art = "the_other"
            i += 1
    elif low0 == "another":
        art = "another"
        i = 1
    if i < n and span[i].lower() in ORDINALS:
        ordinal = span[i].lower()
        i += 1
    if i < n and span[i].lower() in lexicon.sorts:
        noun = span[i].lower()
        i += 1
    else:
        return None
    if i < n and is_explicit_name(span[i]):
        name = span[i]
        i += 1
    if i != n:
        return None
    if art is None and ordinal is None:
        return None
    return NP(art, ordinal, noun, name, None, " ".join(span))


def match_template(template: Template, tokens, lexicon: Lexicon) -> list:
    """All ways ``tokens`` can be read as ``template`` (list of NP lists)."""
    results: list = []
    elements = template.elements

    def walk(ei: int, ti: int, acc: list) -> None:
        if ei == len(elements):
            if ti == len(tokens):
                results.append(list(acc))
            return
        kind, value = elements[ei]
        if kind == "lit":
            if ti < len(tokens) and tokens[ti].lower() == value:
                walk(ei + 1, ti + 1, acc)
            return
        for end in range(ti + 1, min(len(tokens), ti + 4) + 1):
            np = parse_np_span(tokens[ti:end], lexicon)
            if np is None:
                continue
            acc.append(np)
            walk(ei + 1, end, acc)
            acc.pop()

    walk(0, 0, [])
    return results


def _single_np(span, lexicon: Lexicon) -> NP | None:
    return parse_np_span(span, lexicon)


# Longest phrase first, so 'is on or after' wins over 'is after'.
COMPARISON_PHRASES = (
    (("is", "on", "or", "after"), ">="),
    (("is", "on", "or", "before"), "<="),
    (("is", "before"), "<"),
    (("is", "after"), ">"),
)


def match_comparison(tokens, lexicon: Lexicon) -> Match | None:
    """Match a built-in comparison copula spanning the whole token list."""
    n = len(tokens)
    low = [t.lower() for t in tokens]
    text = " ".join(tokens)

    # '<A> is <N> days before <B>'  ->  arithmetic form.
    for i in range(1, n):
        if low[i] != "is" or i + 3 >= n:
            continue
        if low[i + 2] != "days" or low[i + 3] != "before":
            continue
        if not tokens[i + 1].isdigit():
            continue
        left = _single_np(tokens[:i], lexicon)
        right = _single_np(tokens[i + 4 :], lexicon)
        if left and right:
            return Match(
                kind="arith",
                text=text,
                left=left,
                right=right,
                n=int(tokens[i + 1]),
            )

    for phrase, op in COMPARISON_PHRASES:
        length = len(phrase)
        for i in range(1, n - length):
            if tuple(low[i : i + length]) != phrase:
                continue
            left = _single_np(tokens[:i], lexicon)
            right = _single_np(tokens[i + length :], lexicon)
            if left and right:
                return Match(kind="cmp", text=text, op=op, left=left, right=right)
    return None


def match_atom(lexicon: Lexicon, tokens) -> Match | None:
    """Match a whole atom against the lexicon.

    Raises CompileError when two or more declared templates match the same
    span (a silent wrong longest-match is worse than a
    refusal), and when the span matches no template and no built-in copula
    (the closed-lexicon gap).
    """
    if not tokens:
        return None
    found = []
    for tpl in lexicon.all_templates:
        results = match_template(tpl, tokens, lexicon)
        if not results:
            continue
        if len(results) > 1:
            raise CompileError(
                f"template '<{tpl.text}>' matches '{' '.join(tokens)}' in "
                f"{len(results)} different ways",
                gap=GAP_AMBIGUOUS_TEMPLATE,
            )
        found.append((tpl, results[0]))
    if len(found) > 1:
        names = " / ".join(f"<{t.text}>" for t, _ in found)
        raise CompileError(
            f"{names} all match '{' '.join(tokens)}'",
            gap=GAP_AMBIGUOUS_TEMPLATE,
        )
    if found:
        tpl, nps = found[0]
        return Match(
            kind="template",
            text=" ".join(tokens),
            template=tpl,
            nps=tuple(nps),
        )
    return match_comparison(tokens, lexicon)


# ---------------------------------------------------------------------------
# Meta conditions: '<subject noun phrases> <verb phrase> that <atom>'
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaMatch:
    """A meta condition read off the token stream.

    ``nps`` are the SUBJECT noun phrases (the ``[meta]`` template's own slots,
    in slot order); ``mention`` is the token span after the connective -- the
    MENTIONED object-level sentence, which the lowering reduces to terms.
    """

    template: Template
    connective: tuple
    nps: tuple
    mention: tuple


def meta_parts(tpl: Template) -> tuple:
    """``(subject_template, connective, subject_sorts)`` of a ``[meta]`` template.

    The connective is the run of literal words immediately before the trailing
    ``<atom>`` slot -- validated at load time to be non-empty and to end in
    ``that``.  The subject template is everything before it, so the SUBJECT is
    matched (and its slots type-checked) by exactly the machinery every other
    template uses.
    """
    words = []
    for kind, value in reversed(tpl.elements[:-1]):
        if kind != "lit":
            break
        words.append(value)
    connective = tuple(reversed(words))
    elements = tpl.elements[: len(tpl.elements) - 1 - len(connective)]
    text = " ".join(f"<{v}>" if k == "slot" else v for k, v in elements)
    return (
        Template(elements, tpl.pred, tpl.sorts[:-1], text),
        connective,
        tpl.sorts[:-1],
    )


def match_meta(lexicon: Lexicon, tokens) -> MetaMatch | None:
    """Match a meta condition, or return None when no ``[meta]`` verb occurs.

    None means the token span contains no declared meta verb phrase at all, so
    the caller's ordinary "no declared predicate template matches" refusal is
    the right diagnosis.  A declared verb phrase whose SUBJECT does not match
    its ``[meta]`` template, or one that matches it in more than one way, is a
    loud CompileError instead of a silent pick.
    """
    if not lexicon.meta or not tokens:
        return None
    low = [t.lower() for t in tokens]
    found: list = []
    seen_verb = False
    for tpl in lexicon.meta:
        subject, connective, _sorts = meta_parts(tpl)
        length = len(connective)
        for i in range(len(low) - length + 1):
            if tuple(low[i : i + length]) != connective:
                continue
            seen_verb = True
            results = match_template(subject, tokens[:i], lexicon)
            if len(results) > 1:
                raise CompileError(
                    f"template '<{subject.text}>' matches "
                    f"'{' '.join(tokens[:i])}' in {len(results)} different ways",
                    gap=GAP_AMBIGUOUS_TEMPLATE,
                )
            if not results:
                continue
            found.append(
                MetaMatch(tpl, connective, tuple(results[0]), tuple(tokens[i + length :]))
            )
    if len(found) > 1:
        names = " / ".join(
            f"<{m.template.text}>" for m in found
        )
        raise CompileError(
            f"{names} all match the meta condition '{' '.join(tokens)}'",
            gap=GAP_AMBIGUOUS_TEMPLATE,
        )
    if found:
        return found[0]
    if seen_verb:
        forms = " / ".join(f"<{meta_parts(t)[0].text}>" for t in lexicon.meta)
        raise CompileError(
            f"'{' '.join(tokens)}' contains a declared [meta] verb phrase, but "
            f"the words before it match no declared [meta] subject form "
            f"({forms})",
            gap=GAP_META_SHAPE,
        )
    return None
