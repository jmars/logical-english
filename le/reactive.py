"""LPS reactive rules: ``If <antecedent> then <consequent>.``

The model (Kowalski 2020, "Logical English", LPOP 2020 position paper,
https://www.doc.ic.ac.uk/~rak/papers/LPOP.pdf,
example (1)): "Reactive rules in LPS represent goals that are made true by
making their consequents true whenever their antecedents become true.
Consequents can be made true either deliberately by performing actions or
fortuitously by observing external events."

Compilation of one reactive rule (document order index ``i``) produces TWO
artefacts, which is what the two-stream discipline of this package requires:

* the ANTECEDENT DETECTION RULE, into ``CompileResult.rules`` so that
  everything the engine loads still lives in the rule stream:

      le_antecedent_<i>(<head vars>) :- <antecedent body>.

  Its body is lowered exactly like a declarative clause body (positives, then
  ordered arithmetic producers, then negations, then comparisons), so it is
  ordinary dl text the engine loads and queries; ``db.query`` asks the engine
  whether the antecedent holds, and the row carries the substitution the
  driver needs.
* the CONSEQUENT SCHEMA, into ``CompileResult.reactive``: a
  :class:`ReactiveRule` the :mod:`le.driver` executes.  A consequent is never
  engine text -- it is a driver instruction (record an action, add or delete a
  state fact), because a least-fixpoint engine has no state transition.

Consequent forms (v1):

* ``<action atom>`` -- the predicate must be declared in the lexicon's
  ``[actions]`` section (loud GAP_CONSEQUENT_NOT_ACTION otherwise): a
  deliberate action, which the driver RECORDS and reports.  The driver does
  not write the action's own fact: performing it is the world's job (the
  host application's, unwired here), so a recorded action is an
  output, not a state change.
* ``it becomes the case that <atom>`` -- a fluent update: the driver adds the
  fact (the predicate may be any declared template).
* ``it becomes the case that it is not the case that <atom>`` -- the driver
  deletes the fact.

Every consequent term must be a variable bound in the ANTECEDENT or a declared
constant: a consequent that introduces a term of its own would be
existentially quantified, and v1 refuses to guess that reading
(GAP_CONSEQUENT_SHAPE).

Two deliberate refinements of the shared binding rules, both confined to the
reactive ANTECEDENT (see the ``reactive mode`` note in le/binding.py): the
distinctness guards ``another <noun>`` implies, and the paper's own
"a choice C1 ... a choice C2" (a second named binding of a bound sort).
"""

from __future__ import annotations

from dataclasses import dataclass

from .binding import BindingState, Term
from .errors import CompileError
from .lexicon import is_explicit_name, is_auto_var
from .lower import (
    GAP_CONSEQUENT_NOT_ACTION,
    GAP_CONSEQUENT_SHAPE,
    GAP_REACTIVE_FORM,
    _Atom,
    _build_atom,
    _check_clause,
    _render_clause,
    _sentence_gap_check,
    _split_and,
)

# 'it becomes the case that' / 'it becomes the case that it is not the case
# that' -- the consequent prefixes (longest first).
_BECOMES = ("it", "becomes", "the", "case", "that")
_BECOMES_NOT = _BECOMES + ("it", "is", "not", "the", "case", "that")

# The dl term a var-less antecedent head is padded with: MEASURED, the engine
# rejects a 0-arity rule head ("head arity 0 for variadic"), so a reactive rule
# whose antecedent binds no variable still needs one head column.
ARITY_PAD = "0"


@dataclass(frozen=True)
class Arg:
    """One consequent/event argument: a bound variable or a constant."""

    kind: str  # 'var' | 'const'
    text: str


@dataclass(frozen=True)
class Consequent:
    """One consequent part: an action, a fluent add or a fluent delete."""

    kind: str  # 'action' | 'add' | 'delete'
    pred: str
    args: tuple  # tuple of Arg, in slot order


@dataclass(frozen=True)
class ReactiveRule:
    """The driver's instructions for one reactive rule.

    ``head_vars`` is the column order of the ``le_antecedent_<index>``
    relation, so a row IS the antecedent's substitution: explicit names in
    first-occurrence order, then the auto-generated ``V``-names in ascending
    order (both source-order derived, so the compiled text is deterministic
    and independent of the hash seed).
    """

    index: int
    head_vars: tuple
    antecedent_pred: str
    event_atoms: tuple  # ((pred, (Arg, ...)), ...) in antecedent order
    consequents: tuple  # (Consequent, ...) in consequent order

    @property
    def event_preds(self) -> tuple:
        """The declared action predicates this antecedent reads (consumed)."""
        out: list = []
        for pred, _args in self.event_atoms:
            if pred not in out:
                out.append(pred)
        return tuple(out)


def _vars_of(atom: _Atom) -> list:
    if atom.kind in ("pos", "neg"):
        return [t.text for t in atom.terms if t.kind == "var"]
    return [t.text for t in (atom.left, atom.right) if t.kind == "var"]


def _bound_vars(atoms) -> list:
    """The antecedent's variables in source order (first appearance).

    Only POSITIVE atoms and arithmetic producers can bind a variable, so the
    head carries exactly those: a variable that appears only inside a negated
    atom is not bound at all, and :func:`le.lower._check_clause` must be free
    to say so (GAP_NEG_BINDING) rather than the synthetic head hiding it.
    """
    variables: list = []
    for atom in atoms:
        if atom.kind not in ("pos", "arith"):
            continue
        for name in _vars_of(atom):
            if name not in variables:
                variables.append(name)
    return variables


def _head_order(variables) -> list:
    """Canonical head order: explicit names first, then V-names ascending.

    ``V1`` matches the explicit-name shape too, so auto-generated names are
    classified FIRST (an explicitly written ``V1`` is refused upstream with
    GAP_NAME_AUTO, so the two classes never overlap in practice).
    """
    autos = sorted(
        (v for v in variables if is_auto_var(v)), key=lambda v: int(v[1:])
    )
    names = [v for v in variables if not is_auto_var(v)]
    for name in names:
        if not is_explicit_name(name):
            # A head variable outside either shape is a compiler bug, not a
            # user-visible condition; fail loud rather than drop a column.
            raise AssertionError(f"unclassifiable head variable {name!r}")
    return names + autos


def _consequent(tokens, state, lexicon, head_vars, *, sentence, index):
    low = [t.lower() for t in tokens]
    kind = "action"
    rest = tokens
    if tuple(low[: len(_BECOMES_NOT)]) == _BECOMES_NOT:
        kind, rest = "delete", tokens[len(_BECOMES_NOT) :]
    elif tuple(low[: len(_BECOMES)]) == _BECOMES:
        kind, rest = "add", tokens[len(_BECOMES) :]
    if not rest:
        raise CompileError(
            f"the consequent part '{' '.join(tokens)}' has no predicate after "
            "it",
            gap=GAP_CONSEQUENT_SHAPE,
            sentence=sentence,
            index=index,
        )
    _sentence_gap_check(rest, lexicon, sentence=sentence, index=index)
    atom = _build_atom(rest, state, in_head=False, sentence=sentence, index=index)
    if atom.kind != "pos":
        raise CompileError(
            f"the consequent part '{' '.join(tokens)}' is not a positive "
            "predicate atom",
            gap=GAP_CONSEQUENT_SHAPE,
            sentence=sentence,
            index=index,
        )
    if kind == "action" and not lexicon.is_action(atom.pred):
        raise CompileError(
            f"the consequent '{atom.text}' uses '{atom.pred}', which is not "
            "declared in the lexicon's [actions] section: a consequent the "
            "rule performs must be an external event (a [predicates] template "
            "is static knowledge or state)",
            gap=GAP_CONSEQUENT_NOT_ACTION,
            sentence=sentence,
            index=index,
        )
    args = tuple(Arg(t.kind, t.text) for t in atom.terms)
    for arg in args:
        if arg.kind == "var" and arg.text not in head_vars:
            surface = next(
                (t.surface for t in atom.terms if t.text == arg.text), arg.text
            )
            raise CompileError(
                f"the consequent part '{' '.join(tokens)}' introduces the new "
                f"variable {arg.text} (from '{surface}'); a reactive "
                "consequent may only use variables bound in the antecedent or "
                "declared constants",
                gap=GAP_CONSEQUENT_SHAPE,
                sentence=sentence,
                index=index,
            )
    return Consequent(kind, atom.pred, args)


def parse_reactive(
    tokens, lexicon, counter, consts, *, sentence, index, rule_index=1
):
    """Parse one reactive sentence into ``(antecedent rule text, schema)``."""
    low = [t.lower() for t in tokens]
    if "then" not in low:
        raise CompileError(
            f"{sentence!r} begins with 'If' but has no 'then': a reactive rule "
            "is 'If <antecedent> then <consequent>.'",
            gap=GAP_REACTIVE_FORM,
            sentence=sentence,
            index=index,
        )
    # 'then' is reserved by the lexicon, so the first one is the split.
    then_at = low.index("then")
    antecedent = tokens[1:then_at]
    consequent = tokens[then_at + 1 :]
    if not antecedent or not consequent:
        raise CompileError(
            f"{sentence!r} has an empty antecedent or consequent",
            gap=GAP_REACTIVE_FORM,
            sentence=sentence,
            index=index,
        )
    for part, word, what in (
        (antecedent, "if", "antecedent"),
        (consequent, "then", "consequent"),
    ):
        if any(t.lower() == word for t in part):
            raise CompileError(
                f"{sentence!r} has a second '{word}' inside its {what}",
                gap=GAP_REACTIVE_FORM,
                sentence=sentence,
                index=index,
            )
    # The antecedent is a condition list: the declarative gap checks apply to
    # it unchanged ('it becomes the case that' in an antecedent is a fluent
    # update in a body -- still a loud gap).
    _sentence_gap_check(antecedent, lexicon, sentence=sentence, index=index)

    state = BindingState(lexicon, counter, consts, reactive=True)
    atoms = [
        _build_atom(part, state, in_head=False, sentence=sentence, index=index)
        for part in _split_and(antecedent, sentence=sentence, index=index)
    ]
    # 'another <noun>' claimed a distinct entity: emit its '!=' guards.  They
    # are comparison atoms, so they render after the negations.
    for sort, new, old in state.distinct_pairs:
        atoms.append(
            _Atom(
                kind="cmp",
                op="!=",
                left=Term("var", new, sort, new),
                right=Term("var", old, sort, old),
                text=f"{new} != {old}",
            )
        )

    variables: list = _bound_vars(atoms)
    head_vars = _head_order(variables)

    antecedent_pred = f"le_antecedent_{rule_index}"
    if head_vars:
        head = _Atom(
            kind="pos",
            pred=antecedent_pred,
            terms=tuple(Term("var", v, None, v) for v in head_vars),
        )
    else:
        head = _Atom(
            kind="pos",
            pred=antecedent_pred,
            terms=(Term("const", ARITY_PAD, None, ARITY_PAD),),
        )

    consequents = tuple(
        _consequent(part, state, lexicon, set(head_vars), sentence=sentence,
                    index=index)
        for part in _split_and(consequent, sentence=sentence, index=index)
    )
    event_atoms = tuple(
        (atom.pred, tuple(Arg(t.kind, t.text) for t in atom.terms))
        for atom in atoms
        if atom.kind == "pos" and lexicon.is_action(atom.pred)
    )

    ordered_arith = _check_clause(head, atoms, sentence=sentence, index=index)
    rule_text = _render_clause(head, atoms, ordered_arith)
    rule = ReactiveRule(
        index=rule_index,
        head_vars=tuple(head_vars),
        antecedent_pred=antecedent_pred,
        event_atoms=event_atoms,
        consequents=consequents,
    )
    return rule_text, rule
