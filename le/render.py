"""Reverse rendering: the compiler's own Datalog output -> Logical English.

WHY THIS MODULE: the decided requirement is a BIDIRECTIONAL bridge --
"derived facts are readable back in Logical English" -- and only the forward
half (:func:`le.lower.compile_text`) existed.  This
module renders the compiler's own output shapes back into the Logical English
sentences they encode:

* :func:`render_facts` -- ground ``pred(args).`` fact lines (the same shape
  :func:`le.driver.intern_facts` reads) -> one LE sentence per line;
* :func:`render_rule` -- one ``head :- body.`` rule line -> one LE
  ``conclusion if condition and condition.`` sentence;
* :func:`render_result` -- convenience over :attr:`CompileResult.facts`;
* :func:`render_db` -- ``(pred, cols)`` rows read out of a live dlb
  database (the engine's u32 columns) -> one LE sentence per row.  This is
  the derived-facts leg: the DERIVED tuples come back as sentences.  It
  lives here (not in :mod:`le.driver`) and imports dlb lazily inside the
  function -- exactly the driver's pattern -- so ``import le`` stays
  stdlib-only.

THE INVERSE-BINDING ALGORITHM (render_rule).  The compiler binds variables
as it walks the sentence -- conclusion first, then conditions in source
order -- and numbers them ``V1, V2, ...`` with a counter shared across the
WHOLE document.  The renderer inverts that walk on the compiled rule.
For every emission in the example corpus the rendered sentence recompiles
BYTE-IDENTICALLY through the real compiler (MEASURED); for the
out-of-order class below it
recompiles ALPHA-EQUIVALENTLY (MEASURED, and pinned by
test_negation_introduced_variable_renders_alpha_equivalently).

1. WALK ORDER = head, then the body atoms in their dl order.  The compiler
   emits bodies kind-sorted (positive atoms, then negations, then
   comparisons; ``le.lower._render_clause``), which is NOT always the
   source order: binding follows SOURCE order, so a negated or compared
   condition can bind a variable that an earlier-EMITTED positive atom
   re-mentions (grounding is a set check, le.lower._check_clause), putting
   first occurrences in the dl text out of numbering order.  The renderer
   therefore never reads ordinals off the text order:
2. PER SLOT: a variable's FIRST occurrence in the walk renders ``a
   <noun>``, a LATER one ``the <noun>``; the ORDINAL word comes from the
   variable's NUMBER -- the compiler's counter is sequential, so a
   variable's rank among the clause's same-sort auto variables (by V-number)
   IS its ``(sort, ordinal)`` binding key.  ``a <noun>`` for rank 1, ``a
   first/second/third <noun>`` for ranks 2-4.  For an emission whose
   numbers ascend in the dl text this is the plain inverse walk and the
   rendered sentence recompiles BYTE-IDENTICALLY; for an emission whose
   negation/cmp bound a variable before an earlier-emitted positive atom
   re-mentions it, the sentence renders with the same (sort, ordinal) keys
   but meets them in text order, so it recompiles to an ALPHA-EQUIVALENT
   rule (a consistent renaming of the auto variables -- same atoms, same
   constraints), never a different meaning.  A variable whose text is an
   explicit LE name (``P1``) renders ``a player P1`` at its introduction and
   the bare name ``P1`` at references.  The NOUN is the canonical
   first-declared noun of the slot's DECLARED SORT -- slot markers are slot
   NAMES, not article+noun (MEASURED: 5 of 27 markers in examples/lexicon.ini
   are not parseable noun phrases).  A constant reverse-folds through
   ``[names]`` (or the caller's explicit ``names=`` display map); an integer
   renders as digits.
3. ``!pred(...)`` -> ``it is not the case that <atom>``; comparisons invert
   through the ``COMPARISON_PHRASES`` table (``>=`` -> ``is on or after``).
4. A ``states(...)`` row (meta) inverts through the ``[meta]`` template: the
   subject columns fill the meta template's subject slots, the predicate
   column names the MENTIONED template, and the remaining columns re-render
   through it.  A subject slot whose variable was FOLDED into the mention
   (the compiler drops a subject column that repeats a mentioned variable)
   is recovered by sort-matching against the mention's variables -- or, on
   a fully ground DERIVED row (a meta-head rule evaluated by the engine),
   against the mention VALUES (one sort-consistent reading required).

CONTRACT (render_rule): the input is the compiler's own emission.  The
compiler's counter is shared across the whole DOCUMENT, so a rule's
variables need not start at ``V1`` -- but within one clause they are
consecutive and the numbers ascend along the source binding order, and the
number-derived ranks above make the inverse well-defined WITHOUT reading
the text order (which the kind sort may reorder).  What is refused is what
the compiler could not emit: a body outside the kind order, a variable
only the conclusion mentions (GAP_EXISTENTIAL_HEAD in the forward
direction), a name used as two variables (GAP_RENDER_ORDER).

RENDER-ONLY CASES (display works, the round-trip does not):

* INTEGERS render as digits.  The forward direction refuses digits in a
  slot (MEASURED: ``Rock beats 3`` -> the closed-lexicon gap), so an
  int-carrying row displays but does not re-round-trip.
* ``another`` / ``the other`` are information-lost in dl (both lower to a
  fresh variable; MEASURED on day_chain), so the CANONICAL inverse is the
  ordinal form -- ``another day`` renders back as ``a first day``.  The
  rendered sentence is LE the compiler accepts and recompiles to the same
  dl, but it is not the original surface wording.

SHARED-PREDICATE AMBIGUITY AND ITS ESCAPES.  MEASURED:
examples/lexicon.ini declares two templates for ``before``.  The reverse
renderer never picks silently: a shared predicate is a loud
GAP_RENDER_AMBIGUOUS_TEMPLATE unless (in override order) the caller passes
``template=`` (every entry point: render_rule, render_facts,
render_result, render_db -- one call, one pick), a ``canonical=`` map, or
the lexicon's ADDITIVE ``[render]`` section marks one template canonical::

    [render]
    before = <dayA> is before <dayB>

(the value is the template's ini text, ``|``-fields before the first pipe).
Old inis -- no ``[render]`` section -- parse byte-identically and keep the
loud behaviour.

Every refusal is a loud :class:`~le.errors.CompileError` carrying a
``GAP_RENDER_*`` gap (the module's discipline: never guess a template, never
invent a proper noun, never silently skip a predicate).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from itertools import combinations, product

from .errors import CompileError
from .lexicon import (
    COMPARISON_PHRASES,
    Lexicon,
    Template,
    is_auto_var,
    is_explicit_name,
    is_proper_noun,
    match_atom,
    meta_parts,
)

# -- gaps (each names a feature the reverse renderer refuses, never guesses) --

GAP_RENDER_NO_TEMPLATE = (
    "reverse render: a dl predicate no declared template matches (the closed "
    "lexicon binds the reverse direction too -- le_antecedent_* reactive "
    "artifacts, w_* witness predicates and engine internals are not "
    "LE-readable)"
)
GAP_RENDER_AMBIGUOUS_TEMPLATE = (
    "reverse render: several declared templates share the predicate, so the "
    "surface wording is undecidable from the dl alone (pass template= to "
    "pick one)"
)
GAP_RENDER_UNDECLARED_SYMBOL = (
    "reverse render: a symbol constant with no declared surface -- refusing "
    "to invent a proper noun (pass names={symbol: surface} to declare one)"
)
GAP_RENDER_NAME_COLLISION = (
    "reverse render: several declared surfaces fold onto the same dl symbol"
)
GAP_RENDER_UNSORTED_SLOT = (
    "reverse render: a variable whose sort no declared slot determines, so "
    "its noun phrase cannot be chosen"
)
GAP_RENDER_ARITY = (
    "reverse render: the row's arity disagrees with the declared template"
)
GAP_RENDER_TYPE_CLASH = (
    "reverse render: ontology typing (a constant's declared sort disagrees "
    "with the slot's)"
)
GAP_RENDER_NOT_GROUND = "reverse render: a fact line containing a variable"
GAP_RENDER_ORDER = (
    "reverse render: a rule outside the compiler's own emission shape (body "
    "not kind-sorted, or variable numbering inconsistent with any source "
    "binding order the kind sort allows, or a variable only the conclusion "
    "mentions)"
)
GAP_RENDER_ARITH = (
    "reverse render: an arithmetic producer (L = R - N) in a body -- the "
    "event-calculus '<N> days before' form is a declared gap"
)
GAP_RENDER_CMP_OP = (
    "reverse render: a comparison operator with no inverse copula phrase"
)
GAP_RENDER_PHRASE_COLLISION = (
    "reverse render: the rendered wording would re-read as a DIFFERENT atom "
    "(a declared template and a built-in comparison copula can share a "
    "wording, and templates win in the forward matcher), so the sentence "
    "would silently recompile into another meaning"
)
GAP_RENDER_META_SHAPE = (
    "reverse render: a meta-level row that fits no declared [meta] template "
    "(subject columns + 1 predicate column + the mentioned arguments)"
)
GAP_RENDER_SHAPE = (
    "reverse render: a line outside the compiler's emitted dl shape"
)
GAP_RENDER_BINDING = (
    "reverse render: more distinct variables of one sort than the "
    "article/ordinal table can introduce (a, a first, a second, a third)"
)
GAP_RENDER_OVERRIDE = (
    "reverse render: the template= override names a predicate this input "
    "does not use"
)
GAP_RENDER_NO_DLB = (
    "reverse render: the dlb engine binding is not importable (put the "
    "binding's package directory on sys.path, or set DLB_PATH)"
)

# -- dl shapes (the compiler's own emission; le.driver.intern_facts reads the
# fact shape verbatim) --

_FACT_RE = re.compile(r"([a-z][A-Za-z0-9_]*)\((.*)\)\.\s*\Z")
_PRED_RE = re.compile(r"([a-z][A-Za-z0-9_]*)\((.*)\)\Z")
_ARITH_RE = re.compile(r"\S+\s*=\s*\S+\s*-\s*[0-9]+")
_CMP_RE = re.compile(r"(\S+)\s*(<=|>=|!=|<|>|=)\s*(\S+)")
_VAR_RE = re.compile(r"[A-Z][A-Za-z0-9_]*\Z")
_INT_RE = re.compile(r"[0-9]+\Z")
_SYMBOL_RE = re.compile(r"[a-z][a-z0-9_]*\Z")

# The inverse of le.lexicon.COMPARISON_PHRASES: op -> copula phrase.  Only
# the four ordering copulas have an inverse; '=' only occurs inside the
# (refused) arithmetic producer and '!=' only inside reactive artifacts.
_CMP_PHRASE = {op: " ".join(phrase) for phrase, op in COMPARISON_PHRASES}

# Intro index within a sort -> the ordinal word of its binding key (a
# variable's k-th introduction binds the (sort, word) key; k=1 is the bare
# 'a <noun>' key).  Four keys per sort, as the forward binding allows.
_ORDINAL_WORD = (None, "first", "second", "third")

# Body kinds in the compiler's emission order (le.lower._render_clause).
_KIND_RANK = {"pos": 0, "neg": 1, "cmp": 2}


@dataclass(frozen=True)
class _Term:
    """One dl term as read off a compiled line."""

    kind: str  # 'var' | 'sym' | 'int'
    text: str  # the dl text ('V1', 'acmetransaction', '"Foo Bar"', '3')
    inner: str  # quoted constants: the unescaped surface; else == text


def _term(token: str) -> _Term:
    if _INT_RE.match(token):
        return _Term("int", token, token)
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        inner = (
            token[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        )
        return _Term("sym", token, inner)
    if _VAR_RE.match(token):
        return _Term("var", token, token)
    return _Term("sym", token, token)


@dataclass
class _Atom:
    kind: str  # 'tpl' | 'cmp'
    neg: bool
    pred: str
    args: tuple  # of _Term ('tpl')
    op: str
    left: _Term | None
    right: _Term | None
    text: str


@dataclass
class _MetaRow:
    """A resolved ``states(...)`` row: the meta template plus its reading.

    ``slots`` holds one entry per ``[meta]`` subject slot in slot order:
    the prefix term that fills it, or the mention variable it was folded
    onto (the compiler drops a subject column that repeats a mentioned
    variable; the sort-matched variable recovers it).
    """

    template: Template
    connective: tuple
    subject_sorts: tuple
    slots: tuple  # of _Term
    mention_tpl: Template
    mention_args: tuple  # of _Term


def _split_commas(text: str) -> list:
    """Split ``text`` on top-level commas (quote- and paren-aware)."""
    parts, cur, depth, i, in_quote = [], [], 0, 0, False
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(text):
                cur.append(ch)
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_quote = False
            cur.append(ch)
        elif ch == '"':
            in_quote = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_fact(line: str) -> tuple:
    match = _FACT_RE.match(line.strip())
    if not match:
        raise CompileError(
            f"not a compiled dl fact line: {line.strip()!r} (expected "
            "'pred(args).')",
            gap=GAP_RENDER_SHAPE,
        )
    pred = match.group(1)
    args = tuple(_term(a) for a in _split_commas(match.group(2)))
    return pred, args


def _parse_atom(text: str) -> _Atom:
    core = text.strip()
    neg = core.startswith("!")
    if neg:
        core = core[1:].strip()
        if not core:
            raise CompileError(
                f"a bare '!' in the rule body: {text.strip()!r}",
                gap=GAP_RENDER_SHAPE,
            )
    match = _PRED_RE.match(core)
    if match:
        args = tuple(_term(a) for a in _split_commas(match.group(2)))
        return _Atom("tpl", neg, match.group(1), args, "", None, None, core)
    if _ARITH_RE.match(core):
        raise CompileError(
            f"the arithmetic producer '{core}' cannot be rendered back (the "
            "'<N> days before' form is a declared gap)",
            gap=GAP_RENDER_ARITH,
        )
    cmp_match = _CMP_RE.match(core)
    if cmp_match:
        if neg:
            raise CompileError(
                f"the negated comparison '!{core}' is not a compiler-emitted "
                "shape",
                gap=GAP_RENDER_SHAPE,
            )
        return _Atom(
            "cmp", False, "", (),
            cmp_match.group(2),
            _term(cmp_match.group(1)), _term(cmp_match.group(3)), core,
        )
    raise CompileError(
        f"not a compiled dl atom: {text.strip()!r} (expected 'pred(args)', "
        "'!pred(args)' or 'L <op> R')",
        gap=GAP_RENDER_SHAPE,
    )


def _parse_rule(rule_text: str) -> tuple:
    line = rule_text.strip()
    if not line.endswith("."):
        raise CompileError(
            f"the rule line does not end in '.': {line!r}",
            gap=GAP_RENDER_SHAPE,
        )
    core = line[:-1].strip()
    if ":-" not in core:
        raise CompileError(
            f"the rule line has no ':-' (facts never enter the rules "
            f"stream): {line!r}",
            gap=GAP_RENDER_SHAPE,
        )
    head_text, body_text = core.split(":-", 1)
    head = _parse_atom(head_text)
    if head.kind != "tpl" or head.neg:
        raise CompileError(
            f"the rule head is not a positive predicate atom: "
            f"{head_text.strip()!r}",
            gap=GAP_RENDER_SHAPE,
        )
    body = [_parse_atom(a) for a in _split_commas(body_text)]
    if not body:
        raise CompileError(
            f"the rule has no body atoms: {line!r}",
            gap=GAP_RENDER_SHAPE,
        )
    return head, body


def _cap(sentence: str) -> str:
    if sentence[:1].isdigit():
        # A digit-leading sentence ('3 Happens on Wednesday.') is not a
        # capitalisation artefact -- it is an int in a subject slot, which
        # the forward compiler refuses (render-only), so it is refused here
        # rather than mangled.
        raise CompileError(
            f"the rendered sentence starts with a digit ('{sentence}'): an "
            "integer in a subject slot is render-only in an argument, and a "
            "sentence cannot begin with one",
            gap=GAP_RENDER_SHAPE,
        )
    for i, ch in enumerate(sentence):
        if ch.isalpha():
            return sentence[:i] + ch.upper() + sentence[i + 1 :]
    return sentence


def _fold_key(surface: str) -> str:
    """The dl constant a proper noun folds to (le.binding._constant)."""
    low = surface.lower()
    if _SYMBOL_RE.match(low):
        return low
    return '"' + surface.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _Walk:
    """Per-sentence inverse binding: variable -> (sort, intro index).

    The intro index is the variable's per-sort RANK BY NUMBER (see
    _collect_auto_vars) -- not its walk position: the compiler's counter is
    sequential, so a variable's number already encodes its ``(sort,
    ordinal)`` binding key even when the kind-sorted dl text mentions a later
    variable before an earlier one.
    """

    def __init__(self, renderer: "_Renderer") -> None:
        self.r = renderer
        self.info: dict = {}  # var text -> (sort, k)
        self.count: dict = {}  # sort -> introductions so far

    def var_np(self, term: _Term, slot_sort: str | None) -> str:
        var = term.text
        known = self.info.get(var)
        if known is not None:
            sort, k = known
            if slot_sort is not None and slot_sort != "_" and slot_sort != sort:
                raise CompileError(
                    f"the variable '{var}' occupies a slot declaring sort "
                    f"'{slot_sort}' but was bound as sort '{sort}'",
                    gap=GAP_RENDER_TYPE_CLASH,
                )
            return self._ref(var, sort, k)
        sort = None
        if slot_sort is not None and slot_sort != "_":
            sort = slot_sort
        else:
            sort = self.r.var_sorts.get(var)
        if sort is None:
            raise CompileError(
                f"the variable '{var}' appears in no slot that declares a "
                "sort, so its noun phrase cannot be chosen",
                gap=GAP_RENDER_UNSORTED_SLOT,
            )
        declared = self.r.var_sorts.get(var)
        if declared is not None and declared != sort:
            raise CompileError(
                f"the variable '{var}' is sort '{declared}' elsewhere but "
                f"sort '{sort}' here",
                gap=GAP_RENDER_TYPE_CLASH,
            )
        if self._is_named(var):
            # An explicit name introduces with 'a <noun> <name>' and refers
            # by the bare name; it takes no ordinal key.
            self.info[var] = (sort, None)
            noun = self.r.noun_of(sort)
            article = "an" if noun[0] in "aeiou" else "a"
            return f"{article} {noun} {var}"
        if not is_auto_var(var):
            raise CompileError(
                f"'{var}' is neither an auto variable (V1, V2, ...) nor an "
                "explicit LE name (a letter followed by digits), so it is "
                "not a compiler-emitted term",
                gap=GAP_RENDER_SHAPE,
            )
        k = self.r.rank_of(var, sort)
        if k > len(_ORDINAL_WORD):
            raise CompileError(
                f"sort '{sort}' needs {k} distinct variables, but the "
                "article/ordinal table introduces at most four per sort "
                "per clause (a, a first, a second, a third)",
                gap=GAP_RENDER_BINDING,
            )
        self.count[sort] = self.count.get(sort, 0) + 1
        self.info[var] = (sort, k)
        noun = self.r.noun_of(sort)
        ordinal = _ORDINAL_WORD[k - 1]
        # The article agrees with the word it precedes: every ordinal word
        # starts with a consonant, so an ordinal-carrying introduction is
        # always 'a'; the bare introduction agrees with the noun.
        article = "a" if ordinal else ("an" if noun[0] in "aeiou" else "a")
        words = [article]
        if ordinal:
            words.append(ordinal)
        words.append(noun)
        return " ".join(words)

    def _ref(self, var: str, sort: str, k: int) -> str:
        if self._is_named(var):
            return var
        words = ["the"]
        ordinal = _ORDINAL_WORD[k - 1]
        if ordinal:
            words.append(ordinal)
        words.append(self.r.noun_of(sort))
        return " ".join(words)

    @staticmethod
    def _is_named(var: str) -> bool:
        # V-names are the auto-generated ones (a user may not write them);
        # every other letter+digits shape is an explicit LE name.
        return not is_auto_var(var) and is_explicit_name(var)


class _Renderer:
    def __init__(self, lexicon, names=None, template=None, canonical=None) -> None:
        if not isinstance(lexicon, Lexicon):
            raise TypeError("lexicon must be a le.Lexicon")
        if template is not None and not isinstance(template, Template):
            raise TypeError("template must be a le.lexicon.Template")
        self.lex = lexicon
        self.override = template
        self._override_used = False
        # The lexicon's [render] canonical marking (ini escape; the argument
        # form -- template= -- remains the per-call escape and wins here only
        # when the ini names no template for the predicate).
        self._ini_canonical = dict(
            canonical if canonical is not None else lexicon.render_canonical
        )
        self.var_sorts: dict = {}
        # Auto variables seen while resolving the rule's atoms -- and each
        # one's binding rank by number -- filled by _prescan/
        # _collect_auto_vars before any walk.
        self._numbers: set = set()
        self._var_rank: dict = {}
        # Canonical noun per sort: the FIRST declared noun mapping to it (a
        # non-injective noun->sort map is legal, and this pick is display-
        # only -- same sort, same type-check).
        self._sort_noun: dict = {}
        for noun, sort in lexicon.sorts.items():
            self._sort_noun.setdefault(sort, noun)
        # Reverse-fold index: dl constant -> declared surfaces.  [names]
        # entries key on the fold of the surface; caller display entries key
        # on the dl symbol the caller names (an explicit, caller-owned
        # extension of [names], never an invention by the renderer).
        self._surfaces: dict = {}
        for proper in lexicon.names:
            self._add_surface(_fold_key(proper), proper)
        for symbol, surface in (names or {}).items():
            if not is_proper_noun(surface):
                raise CompileError(
                    f"the display map entry {symbol!r}: {surface!r} is not a "
                    "proper noun (a capitalised, letters-only token)",
                    gap=GAP_RENDER_UNDECLARED_SYMBOL,
                )
            self._add_surface(symbol, surface)
            if len(symbol) >= 2 and symbol.startswith('"') and symbol.endswith('"'):
                self._add_surface(symbol[1:-1], surface)

    def _add_surface(self, key: str, surface: str) -> None:
        held = self._surfaces.setdefault(key, [])
        if surface not in held:
            held.append(surface)

    def noun_of(self, sort: str) -> str:
        noun = self._sort_noun.get(sort)
        if noun is None:
            raise CompileError(
                f"sort '{sort}' has no declared noun, so no noun phrase can "
                "be built for it",
                gap=GAP_RENDER_UNSORTED_SLOT,
            )
        return noun

    # -- constants ---------------------------------------------------------

    def const_np(self, term: _Term, slot_sort: str | None) -> str:
        if term.kind == "int":
            # Render-only: the forward compiler refuses digits in a slot, so
            # this displays but does not re-round-trip (module docstring).
            return term.text
        keys = (term.text, term.inner) if term.text != term.inner else (term.text,)
        found: list = []
        for key in keys:
            for surface in self._surfaces.get(key, ()):
                if surface not in found:
                    found.append(surface)
        if not found:
            raise CompileError(
                f"the symbol constant '{term.text}' has no declared surface; "
                "the renderer does not invent proper nouns (add it to the "
                "lexicon's [names] section or pass names={symbol: surface})",
                gap=GAP_RENDER_UNDECLARED_SYMBOL,
            )
        if len(found) > 1:
            raise CompileError(
                f"the symbol constant '{term.text}' could be any of "
                f"{', '.join(sorted(found))}",
                gap=GAP_RENDER_NAME_COLLISION,
            )
        surface = found[0]
        if slot_sort is not None and slot_sort != "_":
            declared = self.lex.names.get(surface)
            if declared is not None and declared != slot_sort:
                raise CompileError(
                    f"the constant '{surface}' is declared sort "
                    f"'{declared}' but the slot declares '{slot_sort}'",
                    gap=GAP_RENDER_TYPE_CLASH,
                )
        return surface

    # -- templates ---------------------------------------------------------

    def _templates_for(self, pred: str, arity: int, what: str) -> Template:
        cands = [t for t in self.lex.all_templates if t.pred == pred]
        if self.override is not None and self.override.pred == pred:
            cands = [self.override]
            self._override_used = True
        if not cands:
            raise CompileError(
                f"{what}: no declared template has the predicate '{pred}'",
                gap=GAP_RENDER_NO_TEMPLATE,
            )
        right = [t for t in cands if len(t.sorts) == arity]
        if not right:
            declared = "/".join(str(len(t.sorts)) for t in cands)
            raise CompileError(
                f"{what}: '{pred}' is declared with {declared} argument(s) "
                f"but the row has {arity}",
                gap=GAP_RENDER_ARITY,
            )
        if len(right) > 1:
            # The lexicon's [render] section may mark ONE of them canonical
            # (the ini escape; template= is the per-call escape).  Still
            # ambiguous without either: no silent first-declared pick.
            canonical_text = self._ini_canonical.get(pred)
            if canonical_text is not None:
                marked = [t for t in right if t.source == canonical_text]
                if marked:
                    return marked[0]
            listed = " / ".join(f"<{t.text}>" for t in right)
            raise CompileError(
                f"{what}: {listed} all declare the predicate '{pred}'",
                gap=GAP_RENDER_AMBIGUOUS_TEMPLATE,
            )
        return right[0]

    def rank_of(self, var: str, sort: str) -> int:
        """The variable's per-sort binding rank (see _collect_auto_vars)."""
        rank = self._var_rank.get(sort, {}).get(var)
        if rank is None:
            raise CompileError(
                f"the variable '{var}' of sort '{sort}' is not among the "
                "rule's auto variables (an inconsistent walk)",
                gap=GAP_RENDER_SHAPE,
            )
        return rank

    # -- rendering ---------------------------------------------------------

    def _np_of(self, term: _Term, slot_sort: str | None, walk: _Walk) -> str:
        if term.kind == "var":
            return walk.var_np(term, slot_sort)
        return self.const_np(term, slot_sort)

    def _fill(self, tpl: Template, nps: list) -> str:
        out, i = [], 0
        for kind, value in tpl.elements:
            if kind == "lit":
                out.append(value)
            else:
                out.append(nps[i])
                i += 1
        return " ".join(out)

    def render_tpl_atom(self, atom: _Atom, walk: _Walk) -> str:
        tpl = self._templates_for(atom.pred, len(atom.args), f"atom '{atom.text}'")
        nps = [
            self._np_of(term, sort, walk)
            for term, sort in zip(atom.args, tpl.sorts)
        ]
        text = self._fill(tpl, nps)
        if not any(t.kind == "int" for t in atom.args):
            # An int-bearing atom is render-only by construction (the forward
            # parser refuses a bare digit), so there is no re-read to guard.
            self._verify_tpl_phrase(text, tpl, atom.text)
        return f"it is not the case that {text}" if atom.neg else text

    def render_cmp_atom(self, atom: _Atom, walk: _Walk) -> str:
        phrase = _CMP_PHRASE.get(atom.op)
        if phrase is None:
            raise CompileError(
                f"the comparison operator '{atom.op}' has no inverse copula "
                "phrase",
                gap=GAP_RENDER_CMP_OP,
            )
        left = self._np_of(atom.left, None, walk)
        right = self._np_of(atom.right, None, walk)
        text = f"{left} {phrase} {right}"
        if atom.left.kind != "int" and atom.right.kind != "int":
            self._verify_cmp_phrase(text, atom.op, atom.text)
        return text

    # -- wording verification ----------------------------------------------

    def _verify_tpl_phrase(self, text: str, tpl: Template, where: str) -> None:
        """The rendered span must re-read as the SAME template atom.

        The forward matcher tries declared templates before the built-in
        copulas, so a template whose wording collides with a comparison
        copula ('<dayA> is before <dayB>' vs 'is before') would swallow the
        rendered comparison -- or a second template the same wording would
        take the span -- and the sentence would recompile into a different
        dl.  That is a silent meaning change, so it is loud here.
        """
        try:
            match = match_atom(self.lex, text.split())
        except CompileError as exc:
            raise CompileError(
                f"the rendered atom '{text}' (from '{where}') is ambiguous "
                f"to the forward matcher: {exc.message}",
                gap=GAP_RENDER_PHRASE_COLLISION,
            ) from None
        if (
            match is None
            or match.kind != "template"
            or match.template.pred != tpl.pred
            or len(match.nps) != len(tpl.sorts)
        ):
            raise CompileError(
                f"the rendered atom '{text}' (from '{where}') does not "
                "re-read as the template that rendered it",
                gap=GAP_RENDER_PHRASE_COLLISION,
            )

    def _verify_cmp_phrase(self, text: str, op: str, where: str) -> None:
        """The rendered comparison span must re-read as the SAME comparison."""
        try:
            match = match_atom(self.lex, text.split())
        except CompileError as exc:
            raise CompileError(
                f"the rendered comparison '{text}' (from '{where}') is "
                f"ambiguous to the forward matcher: {exc.message}",
                gap=GAP_RENDER_PHRASE_COLLISION,
            ) from None
        if match is None or match.kind != "cmp" or match.op != op:
            raise CompileError(
                f"the rendered comparison '{text}' (from '{where}') does "
                "not re-read as the comparison '{op}'",
                gap=GAP_RENDER_PHRASE_COLLISION,
            )

    def render_meta_atom(self, row: _MetaRow, walk: _Walk, negated: bool) -> str:
        subject_tpl, _, _ = meta_parts(row.template)
        nps = [
            self._np_of(term, sort, walk)
            for term, sort in zip(row.slots, row.subject_sorts)
        ]
        mention_nps = [
            self._np_of(term, sort, walk)
            for term, sort in zip(row.mention_args, row.mention_tpl.sorts)
        ]
        text = "{} {} {}".format(
            self._fill(subject_tpl, nps),
            " ".join(row.connective),
            self._fill(row.mention_tpl, mention_nps),
        )
        return f"it is not the case that {text}" if negated else text

    # -- meta rows ---------------------------------------------------------

    def _meta_shape(self, atom: _Atom) -> tuple:
        """Resolve a meta row's TEMPLATES: which ``[meta]`` template, which
        prefix is the subject, which template the mention re-renders through.

        Template-level only, so it is stable before the variable sorts are
        pre-scanned (the fold pass below needs those sorts).
        """
        args = atom.args
        candidates = []
        for tpl in self.lex.meta:
            subject_tpl, connective, subject_sorts = meta_parts(tpl)
            slots = len(subject_tpl.slots)
            for prefix_len in range(0, min(slots, len(args) - 1) + 1):
                sym = args[prefix_len]
                if sym.kind != "sym":
                    continue
                mention = self._mention_template(
                    sym.text, len(args) - prefix_len - 1, atom.text
                )
                if mention is None:
                    continue
                candidates.append((tpl, connective, subject_sorts, prefix_len, mention))
        if not candidates:
            raise CompileError(
                f"the meta row '{atom.text}' fits no declared [meta] "
                "template (subject columns + 1 predicate column + the "
                "mentioned arguments)",
                gap=GAP_RENDER_META_SHAPE,
            )
        if len(candidates) > 1:
            listed = " / ".join(f"<{c[0].text}>" for c in candidates)
            raise CompileError(
                f"the meta row '{atom.text}' fits {listed}",
                gap=GAP_RENDER_AMBIGUOUS_TEMPLATE,
            )
        return candidates[0]

    def resolve_meta(self, atom: _Atom) -> _MetaRow:
        """Resolve a ``states(...)`` row, including its folded subject slots.

        The compiler DROPS a subject column whose variable also appears in
        the mention (le.lower._build_meta_atom folds it away), so the row can
        carry fewer subject columns than the template has subject slots.  The
        fold keys on VARIABLE identity -- a constant subject is ALWAYS a
        column -- so a RULE row recovers a folded slot from the mention's
        variables (sort-matched, more than one consistent reading loud).

        A DERIVED row (the fact leg: a rule with a meta head evaluated
        by the engine) is fully ground.  Its mention holds no variables, so
        the variable recovery has nothing to pool; the folded slot is
        recovered from the mention VALUES by the same sort-match, one
        reading required.  That is the inverse of the evaluation itself: the
        head's folded variable took the mention variable's value, so the
        value that fills the subject slot IS a mention value of the right
        sort.
        """
        tpl, connective, subject_sorts, prefix_len, mention = self._meta_shape(atom)
        args = atom.args
        prefix = args[:prefix_len]
        mention_args = args[prefix_len + 1 :]
        slots = len(subject_sorts)
        folded_count = slots - prefix_len

        mention_vars = [
            (i, t) for i, t in enumerate(mention_args) if t.kind == "var"
        ]
        # Value recovery ONLY on a fully ground row: with a variable present
        # the fold is variable-keyed by construction, and pooling values
        # beside it could read a constant subject as folded -- exactly what
        # the forward direction refuses (a constant subject is a column).
        ground_row = not mention_vars
        readings: list = []
        for folded in combinations(range(slots), folded_count):
            open_slots = [i for i in range(slots) if i not in folded]
            if not self._prefix_fits(prefix, open_slots, subject_sorts):
                continue
            per_slot = []
            ok = True
            for slot in folded:
                want = subject_sorts[slot]
                if ground_row:
                    cands = [
                        (i, t)
                        for i, t in enumerate(mention_args)
                        if t.kind != "var"
                        and (want == "_" or self._declared_sort_of(t) == want)
                    ]
                else:
                    cands = [
                        (i, t)
                        for i, t in mention_vars
                        if want == "_" or self.var_sorts.get(t.text, want) == want
                    ]
                if not cands:
                    ok = False
                    break
                per_slot.append(cands)
            if not ok:
                continue
            for combo in product(*per_slot):
                if len({i for i, _ in combo}) != len(combo):
                    continue
                slot_terms = [None] * slots
                for term, slot in zip(prefix, open_slots):
                    slot_terms[slot] = term
                for (i, term), slot in zip(combo, folded):
                    slot_terms[slot] = term
                readings.append(slot_terms)
        # Readings that place the same terms in the same slots are one
        # reading (two mention columns can hold the same variable).
        unique: dict = {}
        for slot_terms in readings:
            key = tuple(t.text if t is not None else None for t in slot_terms)
            unique.setdefault(key, slot_terms)
        if not unique:
            raise CompileError(
                f"the meta row '{atom.text}': no assignment of its subject "
                "columns (and the mention's variables) fits the declared "
                "subject sorts",
                gap=GAP_RENDER_META_SHAPE,
            )
        if len(unique) > 1:
            raise CompileError(
                f"the meta row '{atom.text}' has {len(unique)} consistent "
                "subject-column readings",
                gap=GAP_RENDER_AMBIGUOUS_TEMPLATE,
            )
        slot_terms = next(iter(unique.values()))
        if any(t is None for t in slot_terms):
            raise CompileError(
                f"the meta row '{atom.text}' has too few subject columns",
                gap=GAP_RENDER_META_SHAPE,
            )
        return _MetaRow(
            template=tpl,
            connective=connective,
            subject_sorts=subject_sorts,
            slots=tuple(slot_terms),
            mention_tpl=mention,
            mention_args=mention_args,
        )

    def _prefix_fits(self, prefix, open_slots, subject_sorts) -> bool:
        for term, slot in zip(prefix, open_slots):
            want = subject_sorts[slot]
            if want == "_":
                continue
            if term.kind == "var":
                have = self.var_sorts.get(term.text)
            elif term.kind == "sym":
                have = self._declared_sort_of(term)
            else:
                have = None
            if have is not None and have != want:
                return False
        return True

    def _declared_sort_of(self, term: _Term) -> str | None:
        for key in (term.text, term.inner):
            for surface in self._surfaces.get(key, ()):
                declared = self.lex.names.get(surface)
                if declared is not None:
                    return declared
        return None

    def _mention_template(self, pred: str, arity: int, where: str) -> Template | None:
        cands = [t for t in self.lex.all_templates if t.pred == pred]
        if not cands:
            return None
        if self.override is not None and self.override.pred == pred:
            cands = [self.override]
            self._override_used = True
        right = [t for t in cands if len(t.sorts) == arity]
        if not right:
            return None
        if len(right) > 1:
            # Same canonical escape as _templates_for (the ini [render]
            # marking); still loud when it names none.
            canonical_text = self._ini_canonical.get(pred)
            if canonical_text is not None:
                marked = [t for t in right if t.source == canonical_text]
                if marked:
                    return marked[0]
            listed = " / ".join(f"<{t.text}>" for t in right)
            raise CompileError(
                f"the mentioned predicate in '{where}': {listed} all declare "
                f"'{pred}'",
                gap=GAP_RENDER_AMBIGUOUS_TEMPLATE,
            )
        return right[0]

    # -- sentence assembly -------------------------------------------------

    def _prescan(self, head: _Atom, body: list) -> None:
        """Record each variable's declared sort and its binding rank.

        The RANK is the variable's position by NUMBER among the clause's
        auto variables -- the counter is sequential, so that is the source
        binding order, and it survives the kind sort that reorders the
        emitted text.  Meta SUBJECT slots are skipped for the sorts: which
        term fills them is decided by the fold enumeration, and the walk
        itself takes the sort from the slot at noun-phrase time.

        A duplicate V-number did not come from the compiler and is refused
        here; one walk order consistent with the ascending numbers is proven
        later, per sort, by the walk itself.
        """
        def note(term: _Term, sort: str | None) -> None:
            if term.kind != "var":
                return
            if sort is not None and sort != "_":
                previous = self.var_sorts.get(term.text)
                if previous is not None and previous != sort:
                    raise CompileError(
                        f"the variable '{term.text}' is sort '{previous}' in "
                        f"one slot and '{sort}' in another",
                        gap=GAP_RENDER_TYPE_CLASH,
                    )
                self.var_sorts[term.text] = sort
            if is_auto_var(term.text):
                # Texts, for the rank map: a meta SUBJECT slot's variable (or
                # a comparison operand) carries no slot sort here, but it is
                # still a variable of this clause and its number still ranks.
                self._numbers.add(term.text)

        for atom in (head, *body):
            if atom.kind == "cmp":
                note(atom.left, None)
                note(atom.right, None)
                continue
            if atom.pred in self.lex.meta_preds:
                shape = self._meta_shape(atom)
                mention = shape[4]
                for term in atom.args:
                    note(term, None)
                for term, sort in zip(
                    atom.args[shape[3] + 1 :], mention.sorts
                ):
                    note(term, sort)
                continue
            tpl = self._templates_for(atom.pred, len(atom.args), f"atom '{atom.text}'")
            for term, sort in zip(atom.args, tpl.sorts):
                note(term, sort)

    def _check_body_order(self, body: list) -> None:
        kinds = [
            "cmp" if a.kind == "cmp" else ("neg" if a.neg else "pos")
            for a in body
        ]
        for i in range(1, len(kinds)):
            if _KIND_RANK[kinds[i]] < _KIND_RANK[kinds[i - 1]]:
                raise CompileError(
                    f"the body is not in the compiler's emission order "
                    f"(positive atoms, then negations, then comparisons): "
                    f"'{kinds[i]}' follows '{kinds[i - 1]}'",
                    gap=GAP_RENDER_ORDER,
                )

    def _check_head_vars(self, head: _Atom, body: list) -> None:
        """A variable that only the conclusion mentions is not an emission.

        The forward compiler refuses it (GAP_EXISTENTIAL_HEAD,
        le.lower._check_clause: the engine has no existential head), so the
        renderer refuses it too -- the module contract is that it accepts
        the compiler's emission, and rendering this shape would produce a
        sentence the compiler itself rejects on recompile.
        """
        body_vars = {
            t.text
            for a in body
            for t in (
                a.args if a.kind == "tpl" else
                (t for t in (a.left, a.right) if t is not None)
            )
            if t.kind == "var"
        }
        for t in head.args:
            if t.kind == "var" and t.text not in body_vars:
                raise CompileError(
                    f"the variable '{t.text}' occurs only in the conclusion "
                    "(the forward compiler refuses the same shape with "
                    "GAP_EXISTENTIAL_HEAD)",
                    gap=GAP_RENDER_ORDER,
                )

    def _collect_auto_vars(self, head, body, metas) -> None:
        """Build the per-sort number->rank map after the sorts are known.

        Runs once per rule, AFTER the meta rows have resolved (a subject
        slot's variable may be typed by nothing but its resolved slot).  The
        counter is sequential, so within one sort the ascending numbers ARE
        the source introductions in order -- rank 1 = the ``a <noun>`` key,
        rank k = the (k-1)th ordinal key.  The ranks are injective by
        construction, so no count or gap check is needed here: the walk's
        per-sort introduction count (one NP per binding key, and the binder's
        own repeat-a refusal at recompile) is what makes the rendered
        sentence mean this clause.
        """
        sorts = dict(self.var_sorts)
        for row in metas.values():
            for term, sort in zip(row.slots, row.subject_sorts):
                if term.kind == "var" and sorts.get(term.text) is None:
                    sorts[term.text] = sort
        for var in sorted(self._numbers, key=lambda t: int(t[1:])):
            sort = sorts[var] if sorts.get(var) is not None else "_"
            ranks = self._var_rank.setdefault(sort, {})
            if var in ranks.values():
                raise CompileError(
                    f"the variable '{var}' is ranked twice for sort "
                    f"'{sort}' -- not a compiler emission",
                    gap=GAP_RENDER_ORDER,
                )
            ranks[var] = len(ranks) + 1


def render_facts(
    facts,
    lexicon,
    names=None,
    *,
    template: Template | None = None,
    canonical=None,
) -> tuple:
    """Render ground ``pred(args).`` fact lines as Logical English sentences.

    ``facts`` is an iterable of compiled fact lines (the shape
    :attr:`le.errors.CompileResult.facts` produces and
    :func:`le.driver.intern_facts` reads).  ``names`` is an optional
    caller-owned display map ``{dl symbol: proper-noun surface}`` extending
    the lexicon's ``[names]`` section -- the renderer never invents a proper
    noun, so a symbol with neither is a loud gap.  A shared predicate (one
    several templates declare) is a loud ambiguity unless the caller passes
    ``template=`` (a per-call escape, mirroring :func:`render_rule` -- one
    fact stream, one pick), a ``canonical=`` map, or the lexicon's
    ADDITIVE ``[render]`` section marks one template canonical.

    Raises :class:`~le.errors.CompileError` (never a guess or a silent
    skip) for a predicate with no declared template, a shared-predicate
    ambiguity, an undeclared symbol constant, a variable in a fact line, or
    a line outside the compiled shape.
    """
    renderer = _Renderer(
        lexicon, names=names, template=template, canonical=canonical
    )
    out = []
    for line in facts:
        pred, args = _parse_fact(line)
        if any(t.kind == "var" for t in args):
            raise CompileError(
                f"the fact line contains a variable: {line.strip()!r}",
                gap=GAP_RENDER_NOT_GROUND,
            )
        walk = _Walk(renderer)
        atom = _Atom("tpl", False, pred, args, "", None, None, line.strip()[:-1])
        if pred in renderer.lex.meta_preds:
            row = renderer.resolve_meta(atom)
            text = renderer.render_meta_atom(row, walk, False)
        else:
            text = renderer.render_tpl_atom(atom, walk)
        out.append(_cap(text) + ".")
    if renderer.override is not None and not renderer._override_used:
        raise CompileError(
            f"the template= override names the predicate "
            f"'{renderer.override.pred}', which this fact stream does not "
            "use",
            gap=GAP_RENDER_OVERRIDE,
        )
    return tuple(out)


def render_result(
    result,
    lexicon,
    names=None,
    *,
    template: Template | None = None,
    canonical=None,
) -> tuple:
    """Render :attr:`CompileResult.facts <le.errors.CompileResult>` as LE.

    A convenience over :func:`render_facts` for a whole compilation's fact
    stream (the derived facts leg, read back in Logical English).  ``template=``
    mirrors the same escape there: one fact stream, one pick.
    """
    return render_facts(
        result.facts, lexicon, names=names, template=template, canonical=canonical
    )


def render_rule(
    rule_text: str,
    lexicon,
    *,
    template: Template | None = None,
    canonical: "dict | None" = None,
) -> str:
    """Render one compiled rule line as one Logical English sentence.

    ``rule_text`` is a single ``head :- body.`` line of the compiler's rules
    stream.  ``template`` optionally names the template for a predicate that
    several templates share (MEASURED: examples/lexicon.ini's ``before``
    has two) -- without it such a predicate is a loud ambiguity, unless the
    lexicon's ``[render]`` section (or the ``canonical`` map, which overrides
    it) marks one canonical.
    """
    renderer = _Renderer(lexicon, template=template, canonical=canonical)
    head, body = _parse_rule(rule_text)
    renderer._check_body_order(body)
    renderer._check_head_vars(head, body)
    metas: dict = {}
    for atom in (head, *body):
        if atom.kind == "tpl" and atom.pred in renderer.lex.meta_preds:
            renderer._meta_shape(atom)  # templates resolve without var sorts
    renderer._prescan(head, body)
    # The fold pass reads the pre-scanned sorts, so meta rows resolve their
    # subject columns only now.
    for atom in (head, *body):
        if atom.kind == "tpl" and atom.pred in renderer.lex.meta_preds:
            metas[atom.text] = renderer.resolve_meta(atom)
    renderer._collect_auto_vars(head, body, metas)
    walk = _Walk(renderer)
    parts = []
    for atom in (head, *body):
        row = metas.get(atom.text)
        if row is not None:
            parts.append(renderer.render_meta_atom(row, walk, atom.neg))
        elif atom.kind == "cmp":
            parts.append(renderer.render_cmp_atom(atom, walk))
        else:
            parts.append(renderer.render_tpl_atom(atom, walk))
    if renderer.override is not None and not renderer._override_used:
        raise CompileError(
            f"the template= override names the predicate "
            f"'{renderer.override.pred}', which this rule does not use",
            gap=GAP_RENDER_OVERRIDE,
        )
    conclusion, conditions = parts[0], parts[1:]
    return _cap(f"{conclusion} if {' and '.join(conditions)}.")


# -- the live-database path (slice 3) ----------------------------------------
#
# dlb is imported LAZILY inside _load_dlb (the le.driver pattern) so `import
# le` stays stdlib-only; its absence is a RuntimeError naming the path, and
# every test skips rather than fails without it.


def _load_dlb():
    try:  # pragma: no cover - environment dependent
        import dlb
    except Exception as first_exc:  # pragma: no cover - environment dependent
        dlb_path = os.environ.get("DLB_PATH")
        if not dlb_path:
            raise RuntimeError(
                "render_db needs the datalog engine binding: put the binding's "
                "package directory on sys.path or set DLB_PATH "
                "(import dlb failed: %s)" % (first_exc,)
            ) from first_exc
        sys.path.insert(0, dlb_path)
        try:
            import dlb
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "render_db needs the datalog engine binding: put the binding's "
                "package directory on sys.path or set DLB_PATH "
                "(import dlb failed: %s)" % (exc,)
            ) from exc
    return dlb


def render_db(
    db,
    rows,
    lexicon,
    names=None,
    *,
    template: Template | None = None,
    canonical=None,
) -> tuple:
    """Render ``(pred, cols)`` rows read out of a live dlb database as LE.

    This is the derived-facts leg: the engine's DERIVED tuples, read back in
    Logical English.  ``rows`` is an iterable of ``(pred, cols)`` pairs in
    the engine's own column shape -- u32 values, an interned symbol id
    (``db.intern``; ids are 1-based, dl.h) or a raw integer.  A common
    caller is ``((pred, tuple(row)) for row in db.query(pred,
    collect=True))``.

    The column->term step is the WHOLE difference from :func:`render_facts`:
    ``db.sym_of(col)`` de-interns a symbol id, and a column with no symbol
    (or 0, which no interned id can hold) is a raw integer rendered as
    digits -- render-only, per the module docstring.  Everything after that
    step IS render_facts: same templates, same inverse binding, same loud
    gaps, so a rendered db row re-round-trips exactly when the same tuple
    written as a fact line would.  Unknown relations are simply never
    asked for (the caller names the preds); a requested pred with no
    declared template stays a loud GAP_RENDER_NO_TEMPLATE.

    ``db`` is type-checked against the dlb module found on sys.path (put
    the binding's package directory on it, or set ``DLB_PATH``; see
    :func:`le.driver`'s docstring).  ``template=``
    mirrors the render_facts escape (one call, one pick for a shared
    predicate).
    """
    dlb = _load_dlb()
    if not isinstance(db, dlb.Db):
        raise TypeError(
            "db must be a dlb.Db opened by the caller (the renderer never "
            "opens a database itself)"
        )
    lines = []
    for pred, cols in rows:
        cols = tuple(cols)
        if not cols:
            raise CompileError(
                f"the db row ('{pred}', {cols!r}) has no columns",
                gap=GAP_RENDER_SHAPE,
            )
        tokens = []
        for col in cols:
            if not isinstance(col, int) or isinstance(col, bool):
                raise CompileError(
                    f"the db row ('{pred}', {cols!r}) carries a non-u32 "
                    f"column {col!r} (engine columns are u32 ids or ints)",
                    gap=GAP_RENDER_SHAPE,
                )
            if col <= 0:
                tokens.append(str(col))
                continue
            symbol = db.sym_of(col)
            tokens.append(symbol if symbol is not None else str(col))
        lines.append(f"{pred}({', '.join(tokens)}).")
    return render_facts(
        lines, lexicon, names=names, template=template, canonical=canonical
    )
