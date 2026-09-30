"""Lowering: Logical English text -> datalog-dafsa text + facts.

Two streams by construction (MEASURED, plan node fact (1)): ``dl_load_rules``
rejects inline facts ("rule has no body") even when the relation is
pre-declared, so facts never enter the rules text -- they come back as a
separate list for the ``add_fact`` path.

Body ordering is decided by the COMPILER, never by the source order: positive
atom(s) first (they bind every variable), then arithmetic producers in
dependency order (each producer's operand must be bound by a PRECEDING atom),
then negated atoms, then comparisons.  The engine requires a variable in a
negated atom or a comparison to be bound by a positive body atom first
(stratified negation; MEASURED fact (4)), so trusting the input order would
compile programs the engine rejects.

Every feature outside the declarative subset raises CompileError naming the
gap (state the gap explicitly; never silently drop) -- see the GAP_*
constants below.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .binding import BindingState, Counter, Term
from .errors import CompileError, CompileResult
from .lexicon import (
    GAP_META_ARITY,
    GAP_META_SHAPE,
    GAP_META_VARIABLE_PRED,
    Lexicon,
    match_atom,
    match_comparison,
    match_meta,
    meta_parts,
)

# ``GAP_META_SHAPE`` / ``GAP_META_ARITY`` / ``GAP_META_VARIABLE_PRED`` are
# defined in le.lexicon (where the ``[meta]`` declaration is checked) and
# imported here so that a caller reading the lowering finds every gap a meta
# condition can raise in one place.

# -- gaps (each names a feature the v1 subset refuses) ----------------------

GAP_REACTIVE_FORM = (
    "a malformed LPS reactive rule -- the form is 'If <antecedent> then "
    "<consequent>.' (paper example 1)"
)
GAP_CONSEQUENT_NOT_ACTION = (
    "a reactive consequent performed as an action must be a template declared "
    "in the lexicon's [actions] section (a [predicates] template is static "
    "knowledge or state, not something a rule can perform)"
)
GAP_CONSEQUENT_SHAPE = (
    "a reactive consequent must be an action atom, 'it becomes the case that "
    "<atom>' or 'it becomes the case that it is not the case that <atom>', "
    "using variables bound in the antecedent (or constants)"
)
GAP_FLUENT_UPDATE = (
    "'it becomes the case that' outside a reactive consequent -- LPS fluent "
    "update / event calculus (paper example 3)"
)
GAP_RELATIVE_CLAUSE = (
    "relative clauses ('that' / 'which' / 'as of which', paper example 6)"
)
GAP_TEMPORAL_WHEN = "'when' conditions (paper example 3); the clause form uses 'if'"
GAP_THE_FUNCTIONAL = (
    "'the <noun>' used as a functional expression ('the day before the day "
    "before Wednesday'); v1 reads 'the' only as a back-reference to one "
    "variable -- paper example 6"
)
GAP_PLURALS = "plurals and quantifiers ('all'/'some'/...; LE requires singular nouns)"
GAP_CONJUNCTIVE_CONCLUSION = (
    "conjunctive conclusions -- LPS action sequences (paper example 1)"
)
GAP_CLOSED_LEXICON = "ontology extension (the declared predicate set is closed)"
GAP_TYPE_CLASH = "ontology typing (a slot's sort disagrees with the declaration)"
GAP_EXISTENTIAL_HEAD = (
    "existential head variables -- skolemization (paper examples 4 and 5)"
)
GAP_NEG_BINDING = (
    "a negated atom whose variables are not bound by a positive condition "
    "(stratified negation)"
)
GAP_UNGROUNDED_CMP = (
    "a comparison whose operands are not bound by a positive condition "
    "(the engine rejects an ungrounded comparison operand)"
)
GAP_NEG_ONLY = (
    "a clause with no positive condition (unstratified negation; the engine "
    "rejects a body that is only negation)"
)
GAP_SYMBOL_ORDER = (
    "ordering comparison on a symbol constant (the engine rejects a symbol "
    "constant in '<', '<=', '>', '>='); declare a predicate template instead"
)
GAP_NEGATED_CMP = "negated comparisons"
GAP_SORT_COMPARE = "a comparison between values of two different sorts"
GAP_TEMPORAL_ARITH = (
    "'<N> days before' as a predicate argument -- the event-calculus arithmetic "
    "form (paper example 3); the produced variable, its operand and the order "
    "of the producers must be consistent (one producer per result variable, no "
    "cycles)"
)
GAP_FACT_GROUND = "facts must be ground (no variables in a fact)"
GAP_FACT_SHAPE = "facts are positive predicate atoms only"
GAP_HEAD_SHAPE = "the conclusion of a clause must be a positive predicate atom"
GAP_TERMINATOR = "a sentence must end with '.'"

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_ALLOWED_CHARS = re.compile(r"[A-Za-z0-9_,.\s]+\Z")
_UNSUPPORTED = re.compile(r"[^A-Za-z0-9_,.\s]")

_NEG_PREFIX = ("it", "is", "not", "the", "case", "that")
_QUANTIFIERS = frozenset(
    {
        "all", "every", "each", "some", "any", "no", "both", "most",
        "several", "many", "either", "neither",
    }
)
# Fixed copula tokens that must not be read as plural nouns.
_COPULA_WORDS = frozenset({"days"})


@dataclass
class _Atom:
    kind: str  # 'pos' | 'neg' | 'cmp' | 'arith'
    pred: str = ""
    terms: tuple = ()
    op: str = ""
    left: Term | None = None
    right: Term | None = None
    n: int | None = None
    text: str = ""


# -- sentence splitting -----------------------------------------------------


def _tokenize(text: str) -> list:
    return _WORD.findall(text)


def _sentences(text: str):
    """Yield ``(index, raw, tokens)`` for each sentence in ``text``."""
    lines = []
    for line in text.splitlines():
        cut = line.find("#")
        lines.append(line[:cut] if cut >= 0 else line)
    cleaned = "\n".join(lines)
    chunks = cleaned.split(".")
    tail = chunks.pop()
    if tail.strip():
        raise CompileError(
            f"final sentence does not end with '.': {tail.strip()!r}",
            gap=GAP_TERMINATOR,
        )
    out = []
    index = 0
    for chunk in chunks:
        index += 1
        if not chunk.strip():
            continue
        raw = chunk.strip()
        if not _ALLOWED_CHARS.match(chunk):
            bad = sorted(set(_UNSUPPORTED.findall(chunk)))
            raise CompileError(
                f"unsupported character(s) {bad} in sentence",
                sentence=raw,
                index=index,
            )
        out.append((index, raw, _tokenize(chunk)))
    return out


def _split_and(tokens, *, sentence, index) -> list:
    parts: list = [[]]
    for token in tokens:
        if token.lower() == "and":
            parts.append([])
        else:
            parts[-1].append(token)
    if any(not part for part in parts):
        raise CompileError(
            "empty condition (a stray 'and')", sentence=sentence, index=index
        )
    return parts


def _meta_masked(tokens, lexicon: Lexicon) -> set:
    """Indices of the tokens that form a declared ``[meta]`` verb phrase.

    ``states that`` is a legal part of a meta condition, so the relative-clause
    scan must not read its ``that`` as a relative pronoun (and
    :func:`_build_meta_atom` uses the same test to notice a *mention of a
    mention*).  Masking is exact: only the declared connective of a declared
    ``[meta]`` template is hidden, so every other ``that`` (the paper's example
    6, a user's relative clause) is still refused by name.
    """
    masked: set = set()
    low = [t.lower() for t in tokens]
    for tpl in lexicon.meta:
        connective = meta_parts(tpl)[1]
        length = len(connective)
        for i in range(len(low) - length + 1):
            if tuple(low[i : i + length]) == connective:
                masked.update(range(i, i + length))
    return masked


def _sentence_gap_check(tokens, lexicon: Lexicon, *, sentence, index) -> None:
    """Refuse the declarative forms v1 does not lower.

    A sentence-initial ``If`` no longer reaches here: :func:`compile_text`
    routes it to :mod:`le.reactive`.  ``it becomes the case that`` is still
    refused in DECLARATIVE position (it is only legal as the prefix of a
    reactive consequent, which the reactive parser strips before calling this).
    """
    low = [t.lower() for t in tokens]
    joined = " " + " ".join(low) + " "
    for needle, gap in (
        (" it becomes the case that ", GAP_FLUENT_UPDATE),
        (" when ", GAP_TEMPORAL_WHEN),
    ):
        if needle in joined:
            raise CompileError(
                f"{sentence!r} uses '{needle.strip()}', which v1 does not compile",
                gap=gap,
                sentence=sentence,
                index=index,
            )
    masked = _meta_masked(tokens, lexicon)
    kept = " " + " ".join(t for i, t in enumerate(low) if i not in masked) + " "
    stripped = kept.replace(" it is not the case that ", " ")
    for needle in (" that ", " which ", " as of which "):
        if needle in stripped:
            raise CompileError(
                f"{sentence!r} uses the relative '{needle.strip()}' form, "
                "which v1 does not compile",
                gap=GAP_RELATIVE_CLAUSE,
                sentence=sentence,
                index=index,
            )
    quantifier = sorted(_QUANTIFIERS.intersection(low))
    if quantifier:
        raise CompileError(
            f"{sentence!r} uses the quantifier '{quantifier[0]}'; LE quantifies "
            "by article only",
            gap=GAP_PLURALS,
            sentence=sentence,
            index=index,
        )


def _plural_check(tokens, lexicon: Lexicon, *, sentence, index) -> None:
    for i, token in enumerate(tokens):
        low = token.lower()
        if low in _COPULA_WORDS:
            # '<N> days before' belongs to the copula table, not to a noun.
            if (
                i > 0
                and tokens[i - 1].isdigit()
                and i + 1 < len(tokens)
                and tokens[i + 1].lower() == "before"
            ):
                continue
        if low in lexicon.sorts:
            continue
        stem = None
        if low.endswith("es") and low[:-2] in lexicon.sorts:
            stem = low[:-2]
        elif low.endswith("s") and low[:-1] in lexicon.sorts:
            stem = low[:-1]
        if stem is not None:
            raise CompileError(
                f"'{token}' is plural (singular '{stem}' is declared); LE uses "
                "singular nouns and articles only",
                gap=GAP_PLURALS,
                sentence=sentence,
                index=index,
            )


# -- atom building ----------------------------------------------------------


def _repeated_definite(tokens, lexicon: Lexicon):
    """The head noun of a second 'the <noun>' in a token run, if any.

    A functional reading ('the day before the day before Wednesday') has no
    template, so the closed-lexicon error would hide the real cause; naming it
    points at the paper's example 6 extension instead (plan risk (f)).
    """
    seen = {}
    for i, token in enumerate(tokens):
        if token.lower() != "the" or i + 1 >= len(tokens):
            continue
        noun = tokens[i + 1].lower()
        if noun == "other" and i + 2 < len(tokens):
            noun = tokens[i + 2].lower()
        if noun not in lexicon.sorts:
            continue
        seen[noun] = seen.get(noun, 0) + 1
        if seen[noun] > 1:
            return noun
    return None


def _template_terms(template, nps, state, *, label, sentence, index) -> list:
    """Resolve a template match's noun phrases to terms, type-checking slots.

    Shared by ordinary atoms and by the arguments of a MENTIONED atom, so a
    meta mention is typed by the template it mentions -- the same sorts, the
    same messages -- rather than by a parallel implementation that could drift.
    """
    terms = []
    for i, np in enumerate(nps):
        term = state.resolve_np(np, sentence=sentence, index=index)
        want = template.sorts[i]
        if term.sort is not None and want != "_" and want != term.sort:
            raise CompileError(
                f"in '{label}': slot {i + 1} of '{template.pred}' "
                f"declares sort '{want}' but '{np.text}' is sort "
                f"'{term.sort}'",
                gap=GAP_TYPE_CLASH,
                sentence=sentence,
                index=index,
            )
        terms.append(term)
    return terms


# The engine's per-atom arity cap (MEASURED: 8 columns load and derive, 9 are
# refused with 'arity must be 0..8').
META_MAX_COLUMNS = 8


def _build_meta_atom(meta, state, *, negated, in_head, text, sentence, index):
    """Lower a meta condition to an ORDINARY reified atom.

    The USE/MENTION distinction is the whole point, and it is structural: the
    mentioned sentence is reduced to TERMS here and NEVER becomes an _Atom of
    the outer clause.  ``a confirmation of the transaction states that the
    transaction is governed by IsdaAgreement`` becomes

        states(V2, governs, V1, isdaagreement)

    -- a plain positive/negative atom whose second column is the mentioned
    predicate as a symbol, so the object-level ``governs`` of the conclusion is
    USED and this one is MENTIONED (paper paragraph 18).  Compiling the mention
    as a body atom instead would silently invert the semantics, which is why
    the mention is matched and reduced before any atom is built.
    """
    template = meta.template
    subject_tpl, connective, _sorts = meta_parts(template)
    subject_terms = _template_terms(
        subject_tpl, meta.nps, state, label=text, sentence=sentence, index=index
    )
    mention = list(meta.mention)
    phrase = " ".join(connective)
    if not mention:
        raise CompileError(
            f"'{text}' has nothing after '{phrase}': a meta condition mentions "
            "an object-level sentence ('<subject> states that <atom>')",
            gap=GAP_META_SHAPE,
            sentence=sentence,
            index=index,
        )
    if tuple(t.lower() for t in mention[: len(_NEG_PREFIX)]) == _NEG_PREFIX:
        raise CompileError(
            f"the mentioned sentence '{' '.join(mention)}' is negated; a meta "
            "condition mentions a POSITIVE object-level atom in v1",
            gap=GAP_META_SHAPE,
            sentence=sentence,
            index=index,
        )
    if _meta_masked(mention, state.lexicon):
        raise CompileError(
            f"the mentioned sentence '{' '.join(mention)}' is itself a meta "
            "condition; a mention is an OBJECT-level atom (nesting is not "
            "admitted in v1)",
            gap=GAP_META_SHAPE,
            sentence=sentence,
            index=index,
        )
    matched = match_atom(state.lexicon, mention)
    if matched is None or matched.kind != "template":
        if match_comparison(mention, state.lexicon) is not None:
            raise CompileError(
                f"the mentioned sentence '{' '.join(mention)}' is a comparison, "
                "not a predicate atom: a meta condition mentions a DECLARED "
                "predicate template (the paper's 'is governed by' is used as a "
                "predicate in the conclusion and mentioned as a term here)",
                gap=GAP_META_SHAPE,
                sentence=sentence,
                index=index,
            )
        raise CompileError(
            f"the mentioned predicate in '{' '.join(mention)}' is not declared: "
            "the closed lexicon holds INSIDE a meta condition too (an invented "
            "predicate would carry no semantics)",
            gap=GAP_CLOSED_LEXICON,
            sentence=sentence,
            index=index,
        )
    mention_tpl = matched.template
    mention_terms = _template_terms(
        mention_tpl, matched.nps, state,
        label=matched.text, sentence=sentence, index=index,
    )
    # A subject VARIABLE the mention already carries is not repeated in the
    # subject prefix: the paper's 'a confirmation of the transaction states
    # that the transaction is governed by IsdaAgreement' must reify as
    # states(Conf, governs(T, A)) -- one column per DISTINCT entity -- and not
    # states(Conf, T, governs, T, A), which would split the same variable over
    # two columns of the paper's own example (see the README).  The fold keys
    # on VARIABLE identity (a variable's text IS the entity it stands for) and
    # never on a constant: folding a constant subject would let two different
    # [meta] templates reify onto the same row, so a fact posted through one
    # would be silently read by a rule compiled through the other.  A constant
    # subject is therefore ALWAYS a column.
    mentioned = {term.text for term in mention_terms if term.kind == "var"}
    prefix = [
        term
        for term in subject_terms
        if not (term.kind == "var" and term.text in mentioned)
    ]
    # The engine's per-atom cap is the row's bound: the FOLDED columns of this
    # instantiation -- the subject prefix + 1 predicate column + the mentioned
    # arguments -- must fit the 8 (MEASURED: 8 columns load and derive, 9 are
    # refused with 'arity must be 0..8').  The check therefore runs on the row
    # that is actually emitted, AFTER the mention's terms are resolved, so a
    # mention that carries a subject term back still fits; folding only ever
    # removes columns, so this is the tighter check.
    columns = len(prefix) + 1 + len(mention_terms)
    if columns > META_MAX_COLUMNS:
        detail = (
            f"{len(subject_tpl.slots)} subject slot(s) + 1 predicate column + "
            f"{len(mention_tpl.sorts)} argument(s) of the mentioned "
            f"'{mention_tpl.pred}'"
        )
        folded = len(subject_terms) - len(prefix)
        if folded:
            detail += f", less {folded} folded subject term(s)"
        raise CompileError(
            f"the meta relation would need {columns} columns ({detail}), but "
            f"the engine's arity cap is {META_MAX_COLUMNS}",
            gap=GAP_META_ARITY,
            sentence=sentence,
            index=index,
        )
    if in_head and negated:
        raise CompileError(
            "a conclusion cannot be negated (dl has no negative heads)",
            gap=GAP_HEAD_SHAPE,
            sentence=sentence,
            index=index,
        )
    # The mentioned predicate's own name is the reified symbol: a bare dl
    # symbol (predicate names are lower-case identifiers by construction) with
    # no sort, so it can only ever occupy a meta relation column -- never a
    # typed slot or a comparison.
    symbol = Term("const", mention_tpl.pred, None, mention_tpl.pred)
    return _Atom(
        kind="neg" if negated else "pos",
        pred=template.pred,
        terms=tuple(prefix) + (symbol,) + tuple(mention_terms),
        text=text,
    )


def _build_atom(tokens, state: BindingState, *, in_head: bool, sentence, index):
    tokens = list(tokens)
    if not tokens:
        raise CompileError("empty condition", sentence=sentence, index=index)
    negated = False
    if tuple(t.lower() for t in tokens[:6]) == _NEG_PREFIX:
        negated = True
        tokens = tokens[6:]
        if not tokens:
            raise CompileError(
                "'it is not the case that' with nothing after it",
                sentence=sentence,
                index=index,
            )

    match = match_atom(state.lexicon, tokens)
    if match is None:
        # A meta condition is tried ONLY here, after the ordinary matcher has
        # failed for the whole span: every non-meta sentence therefore takes
        # exactly the path it took before the meta embedding existed.
        meta = match_meta(state.lexicon, tokens)
        if meta is not None:
            return _build_meta_atom(
                meta, state, negated=negated, in_head=in_head,
                text=" ".join(tokens), sentence=sentence, index=index,
            )
        repeated = _repeated_definite(tokens, state.lexicon)
        if repeated is not None and not negated:
            raise CompileError(
                f"'{' '.join(tokens)}' uses 'the {repeated}' more than once",
                gap=GAP_THE_FUNCTIONAL,
                sentence=sentence,
                index=index,
            )
        raise CompileError(
            f"no declared predicate template matches '{' '.join(tokens)}'",
            gap=GAP_CLOSED_LEXICON,
            sentence=sentence,
            index=index,
        )

    if match.kind == "template":
        template = match.template
        terms = _template_terms(
            template, match.nps, state,
            label=match.text, sentence=sentence, index=index,
        )
        if in_head and negated:
            raise CompileError(
                "a conclusion cannot be negated (dl has no negative heads)",
                gap=GAP_HEAD_SHAPE,
                sentence=sentence,
                index=index,
            )
        return _Atom(
            kind="neg" if negated else "pos",
            pred=template.pred,
            terms=tuple(terms),
            text=match.text,
        )

    if negated:
        raise CompileError(
            f"negating '{match.text}' is not compiled",
            gap=GAP_NEGATED_CMP,
            sentence=sentence,
            index=index,
        )

    left = state.resolve_np(match.left, sentence=sentence, index=index)
    right = state.resolve_np(match.right, sentence=sentence, index=index)

    if match.kind == "cmp":
        for term in (left, right):
            if term.kind == "const":
                raise CompileError(
                    f"'{match.text}' compares the symbol constant "
                    f"'{term.surface}' with an ordering operator",
                    gap=GAP_SYMBOL_ORDER,
                    sentence=sentence,
                    index=index,
                )
        if left.sort is not None and right.sort is not None and left.sort != right.sort:
            raise CompileError(
                f"'{match.text}' compares sort '{left.sort}' with sort "
                f"'{right.sort}'",
                gap=GAP_SORT_COMPARE,
                sentence=sentence,
                index=index,
            )
        return _Atom(kind="cmp", op=match.op, left=left, right=right, text=match.text)

    # '<N> days before': an arithmetic producer, so its subject must be a
    # fresh variable and its object a variable carrying the numeric date.
    if left.kind != "var":
        raise CompileError(
            f"'{match.text}': the subject of '<N> days before' must be a "
            f"variable, not the constant '{left.surface}'",
            gap=GAP_TEMPORAL_ARITH,
            sentence=sentence,
            index=index,
        )
    if right.kind != "var":
        raise CompileError(
            f"'{match.text}': '{right.surface}' has no numeric value for the "
            "'- N' arithmetic",
            gap=GAP_TEMPORAL_ARITH,
            sentence=sentence,
            index=index,
        )
    if left.sort is not None and right.sort is not None and left.sort != right.sort:
        raise CompileError(
            f"'{match.text}' mixes sort '{left.sort}' with sort '{right.sort}'",
            gap=GAP_SORT_COMPARE,
            sentence=sentence,
            index=index,
        )
    return _Atom(kind="arith", left=left, right=right, n=match.n, text=match.text)


# -- assembly ---------------------------------------------------------------


def _order_arith(arith: list, pos_vars: set, *, sentence, index) -> list:
    """Order arithmetic producers ``result = operand - N`` by dependency.

    Every producer must have exactly one definition (a second one for the same
    result variable is a silent overwrite the engine happily loads and answers
    with the wrong rows) and its operand must be bound by a preceding atom --
    a positive condition or a producer emitted before it.  A set of producers
    that never becomes emittable is a cycle or a self-reference; both are loud
    errors rather than emissions the engine rejects.
    """
    producers: dict = {}
    for atom in arith:
        if atom.left.text in producers:
            first = producers[atom.left.text]
            raise CompileError(
                f"the arithmetic result '{atom.left.text}' is produced twice "
                f"in one clause ('{first.text}' and '{atom.text}'); the engine "
                "would silently answer with the rows of one of them",
                gap=GAP_TEMPORAL_ARITH,
                sentence=sentence,
                index=index,
            )
        if atom.left.text in pos_vars:
            raise CompileError(
                f"the arithmetic result '{atom.left.text}' is already bound "
                "by a positive condition",
                gap=GAP_TEMPORAL_ARITH,
                sentence=sentence,
                index=index,
            )
        producers[atom.left.text] = atom

    available = set(pos_vars)
    pending = list(arith)
    ordered: list = []
    while pending:
        for position, atom in enumerate(pending):
            if atom.right.text in available:
                ordered.append(atom)
                available.add(atom.left.text)
                pending.pop(position)
                break
        else:
            stuck = ", ".join(f"'{a.text}'" for a in pending)
            raise CompileError(
                f"the arithmetic condition(s) {stuck} form a cycle (including "
                "a self-reference) or depend on a variable that no positive "
                "condition binds",
                gap=GAP_TEMPORAL_ARITH,
                sentence=sentence,
                index=index,
            )
    return ordered


def _check_clause(head: _Atom, atoms, *, sentence, index) -> list:
    """Check one clause body; return its arithmetic atoms in producer order."""
    positive = [a for a in atoms if a.kind == "pos"]
    pos_vars = {
        t.text for a in positive for t in a.terms if t.kind == "var"
    }
    arith = [a for a in atoms if a.kind == "arith"]
    # Ordering runs before the body-shape check: a body of arithmetic alone is
    # an arithmetic dependency error, not a 'no positive condition' one.
    ordered_arith = _order_arith(arith, pos_vars, sentence=sentence, index=index)
    if not positive:
        raise CompileError(
            "this clause has no positive condition", gap=GAP_NEG_ONLY,
            sentence=sentence, index=index,
        )
    arith_vars = {a.left.text for a in arith}
    # A head variable may be arithmetic-produced (the engine accepts it:
    # MEASURED 'delivery_day(V1) :- before(V2,V3), V1 = V3 - 3.').
    bound = pos_vars | arith_vars

    head_vars = {t.text for t in head.terms if t.kind == "var"}
    missing = sorted(head_vars - bound)
    if missing:
        raise CompileError(
            f"variable(s) {', '.join(missing)} occur only in the conclusion",
            gap=GAP_EXISTENTIAL_HEAD,
            sentence=sentence,
            index=index,
        )
    for atom in atoms:
        if atom.kind == "neg":
            # Only a POSITIVE atom grounds a negation: the engine's
            # 'unsafe negation' check does not count an arithmetic producer.
            unbound = sorted(
                {t.text for t in atom.terms if t.kind == "var"} - pos_vars
            )
            if unbound:
                raise CompileError(
                    f"the negated atom uses variable(s) {', '.join(unbound)} "
                    "that no positive condition binds",
                    gap=GAP_NEG_BINDING,
                    sentence=sentence,
                    index=index,
                )
        elif atom.kind == "cmp":
            # A comparison MAY be grounded by arithmetic (the engine accepts
            # it; MEASURED), and comparisons are emitted after the producers.
            unbound = sorted(
                {t.text for t in (atom.left, atom.right) if t.kind == "var"} - bound
            )
            if unbound:
                raise CompileError(
                    f"the comparison uses ungrounded variable(s) "
                    f"{', '.join(unbound)}",
                    gap=GAP_UNGROUNDED_CMP,
                    sentence=sentence,
                    index=index,
                )
    return ordered_arith


def _render_atom(atom: _Atom) -> str:
    if atom.kind in ("pos", "neg"):
        args = ", ".join(t.text for t in atom.terms)
        return f"{'!' if atom.kind == 'neg' else ''}{atom.pred}({args})"
    if atom.kind == "cmp":
        return f"{atom.left.text} {atom.op} {atom.right.text}"
    return f"{atom.left.text} = {atom.right.text} - {atom.n}"


def _render_clause(head: _Atom, atoms: list, ordered_arith: list) -> str:
    ordered = (
        [a for a in atoms if a.kind == "pos"]
        + list(ordered_arith)
        + [a for a in atoms if a.kind == "neg"]
        + [a for a in atoms if a.kind == "cmp"]
    )
    body = ", ".join(_render_atom(a) for a in ordered)
    return f"{_render_atom(head)} :- {body}.\n"


def _compile_clause(
    conclusion, conditions, lexicon, counter, consts, *, raw, index
) -> str:
    low = [t.lower() for t in conclusion]
    if "and" in low and match_atom(lexicon, conclusion) is None:
        raise CompileError(
            "the conclusion contains 'and'",
            gap=GAP_CONJUNCTIVE_CONCLUSION,
            sentence=raw,
            index=index,
        )
    state = BindingState(lexicon, counter, consts)
    head = _build_atom(conclusion, state, in_head=True, sentence=raw, index=index)
    if head.kind != "pos":
        raise CompileError(
            "the conclusion is not a positive predicate atom",
            gap=GAP_HEAD_SHAPE,
            sentence=raw,
            index=index,
        )
    parts = _split_and(conditions, sentence=raw, index=index)
    atoms = [
        _build_atom(p, state, in_head=False, sentence=raw, index=index) for p in parts
    ]
    ordered_arith = _check_clause(head, atoms, sentence=raw, index=index)
    return _render_clause(head, atoms, ordered_arith)


def _compile_fact(tokens, lexicon, counter, consts, *, raw, index) -> list:
    state = BindingState(lexicon, counter, consts)
    parts = _split_and(tokens, sentence=raw, index=index)
    facts = []
    for part in parts:
        atom = _build_atom(part, state, in_head=False, sentence=raw, index=index)
        if atom.kind != "pos":
            raise CompileError(
                "a fact must be a positive predicate atom",
                gap=GAP_FACT_SHAPE,
                sentence=raw,
                index=index,
            )
        for term in atom.terms:
            if term.kind == "var":
                raise CompileError(
                    f"the fact contains the variable {term.text} (from "
                    f"'{term.surface}')",
                    gap=GAP_FACT_GROUND,
                    sentence=raw,
                    index=index,
                )
        args = ", ".join(t.text for t in atom.terms)
        facts.append(f"{atom.pred}({args}).")
    return facts


# -- public API -------------------------------------------------------------


def compile_text(text: str, lexicon) -> CompileResult:
    """Compile Logical English ``text`` against ``lexicon``.

    ``lexicon`` is a :class:`~le.lexicon.Lexicon` or a path to a lexicon file.
    Returns a :class:`~le.errors.CompileResult`; raises
    :class:`~le.errors.CompileError` for anything outside the v1 subset.
    """
    if isinstance(lexicon, (str, os.PathLike)):
        if isinstance(lexicon, str) and (
            "\n" in lexicon or lexicon.lstrip().startswith("[")
        ):
            raise TypeError(
                "lexicon looks like lexicon text, not a path; pass "
                "Lexicon.from_text(text) instead"
            )
        lexicon = Lexicon.load(lexicon)
    if not isinstance(lexicon, Lexicon):
        raise TypeError("lexicon must be a le.Lexicon or a path to a lexicon file")

    counter = Counter()
    # One fold map per document: two distinct proper nouns must not fold onto
    # the same dl symbol even when they sit in different sentences.
    consts: dict = {}
    rules: list = []
    facts: list = []
    reactive: list = []
    for index, raw, tokens in _sentences(text):
        if not tokens:
            continue
        if tokens[0].lower() == "if":
            # A sentence-initial 'If' is the LPS reactive form; reactive.py
            # imports this module, so the import is function-level here to
            # keep the dependency acyclic.
            from .reactive import parse_reactive

            _plural_check(tokens, lexicon, sentence=raw, index=index)
            rule_index = len(reactive) + 1
            rule_text, rule = parse_reactive(
                tokens, lexicon, counter, consts,
                sentence=raw, index=index, rule_index=rule_index,
            )
            rules.append(rule_text)
            reactive.append(rule)
            continue
        _sentence_gap_check(tokens, lexicon, sentence=raw, index=index)
        _plural_check(tokens, lexicon, sentence=raw, index=index)
        split = next(
            (i for i, t in enumerate(tokens) if t.lower() == "if"), None
        )
        if split is None:
            facts.extend(
                _compile_fact(
                    tokens, lexicon, counter, consts, raw=raw, index=index
                )
            )
        else:
            rules.append(
                _compile_clause(
                    tokens[:split], tokens[split + 1 :], lexicon, counter,
                    consts, raw=raw, index=index,
                )
            )
    return CompileResult("".join(rules), tuple(facts), tuple(reactive))
