"""The reactive-rule driver: an edge-triggered step function over dlb facts.

WHY A DRIVER.  dlb is a least-fixpoint datalog engine: it has no state
transition, no actions and no abducibles, so an LPS reactive rule ("a goal made
true by making its consequent true whenever its antecedent becomes true") can
only be executed by holding the current state as FACTS and stepping it.  This
module is that step function; it is the only module in ``le`` that imports
``dlb``, and it does so lazily inside a function (``import le`` stays
stdlib-only).  The binding is found by a plain ``import dlb``; a binding
living outside ``sys.path`` can be pointed at with the ``DLB_PATH``
environment variable (the directory that contains the ``dlb`` package -- the
engine tests bootstrap it that way, and :func:`le.render.render_db` honours
it through the same convention).

THE STEP (``submit``), and why it is NOT one transaction.  MEASURED on this
host: a read (``query`` / ``lookup`` / ``count``) inside an OPEN transaction
does NOT see that transaction's buffered writes.  A reaction round therefore
has to read the committed state, so a step is:

* phase 1 -- ONE transaction: the caller's external events (idempotent adds);
* phase 2 -- one transaction PER ROUND: every rule's antecedent is queried by
  the engine (``le_antecedent_<i>``, the rule stream's part of a reactive
  rule), the rows that are NEW since the previous observation fire, the
  consequents are applied, the antecedent's event facts are CONSUMED, and the
  round commits.  Rounds repeat until a round fires nothing (quiescence).

A naive reading of this feature is "one transaction per step"; that is not
executable against this engine -- reads cannot see an open transaction's
writes -- so a step is a sequence of atomic rounds plus a COMPENSATING
RESTORE on abort (below).  Each round is still all-or-nothing.

A ROUND IS TWO-PHASE TOO, and that is what stops one rule from starving
another of a shared event.  Within a round the driver FIRST queries every
rule's antecedent against the state the round STARTED from, collecting the
complete firing plan, and only THEN applies that plan -- the consequent
mutations and the event consumption together, in the round's single
transaction.  No consequence is applied while the plan is being collected, so
two rules that read the SAME ``[actions]`` predicate both fire on the round's
event; neither can consume it out from under the other (a consume-before-
evaluate order would be the engine's own read-your-writes problem, which this
engine does not have: MEASURED, a read cannot see an open transaction's
buffered writes).

EVENTS ARE CONSUMED WHEN PROCESSED.  Consumption is a state mutation like any
other: a rule whose antecedent was false at the moment the event was consumed
never had a true antecedent, and a guard that clears LATER -- a later round of
the same step, or a later step after the world changes -- does not resurrect
the event, because there is no second edge and the event fact is gone.  This
is the LPS reading of an external event as a one-off occurrence (the paper's
"whenever their antecedents become true"); an alternative reading that keeps an
event until every rule reading it has fired would need a per-rule copy of each
event and is NOT implemented (gap g9).

EDGE, NOT LEVEL.  The driver keeps, per rule, the row set of the PREVIOUS
OBSERVATION (the previous round; for round 1, the previous step's last
observation, empty before the first step).  A row fires only when it is not in
that set, i.e. when the antecedent BECAME true between two observations.  The
antecedent's event facts are also consumed (deleted), so a held event cannot
re-fire.  Because consumption and mutation share one transaction, the two
mechanisms cannot interleave (MEASURED: add+delete of one fact inside one
transaction commits last-wins, so the atomically-consuming round is exact).

TERMINATION.  No internal loop can run forever:

* a (rule, row) pair fires at most once per step -- a second firing would mean
  the antecedent went false and true again inside one step, i.e. the reaction
  is oscillating: the step ABORTS loudly (:class:`StepError`).  The check runs
  BEFORE the offending round commits;
* the rounds that FIRE are hard-bounded (``round_bound``, default 100) for
  genuinely growing cascades: a reaction that has not quiesced after that many
  firing rounds ABORTS loudly naming the bound, while a cascade that settles in
  exactly ``round_bound`` firing rounds completes -- the final, quiet
  observation round is not a firing round, so it is not charged to the bound;
* ``run`` is event-driven with an optional ``max_steps``; a truncation raises
  ``ValueError`` instead of silently dropping batches.

On abort the driver ROLLS BACK: an already-committed round cannot be undone by
the engine, so the driver keeps an undo log of every state-changing mutation
(the inverse of each add/delete, recorded once per actual change of the fact's
presence) and applies it in reverse in one final transaction.  That log spans
the caller's own submitted events (phase 1) as well, so an aborted step restores
the whole step -- the submitted events included -- and nothing partial is left
behind.  Under dlb's sole-writer contract (this driver is that writer) the
restore is exact.

v1 LIMITS, stated rather than hidden: the driver is not wired into a host
application (an action consequent is RECORDED and returned as an
:class:`ActionRecord`, whose shape mirrors ``ActionCall(predicate, args)``;
applying it to the world is a one-line adapter for a later task), no event
history is retained (only the current state plus the per-rule
previous-observation sets), and any
fact of a declared action predicate is accepted as an external event -- there
is no check that the event really happened (no world model is in scope).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

from .lexicon import Lexicon
from .reactive import ReactiveRule

DEFAULT_ROUND_BOUND = 100

_INT = re.compile(r"[0-9]+\Z")


class StepError(RuntimeError):
    """A step that could not complete: the reaction did not terminate."""


@dataclass(frozen=True)
class FireRecord:
    """One firing: the rule that fired and the antecedent row that fired it."""

    rule_index: int
    row: tuple


@dataclass(frozen=True)
class ActionRecord:
    """One performed action: a predicate and its interned column values.

    Mirrors the common ``ActionCall(predicate, args)`` shape WITHOUT
    importing it (``le`` stays stdlib-only), so wiring the driver's
    actions into the world is a one-line adapter.
    """

    pred: str
    cols: tuple


@dataclass(frozen=True)
class StepResult:
    """What one step did: firings, rounds, state changes, actions.

    ``added`` / ``deleted`` hold ``(pred, cols)`` pairs, one entry per CHANGE of
    a fact's presence made by the step itself -- including the antecedent event
    facts a firing CONSUMED.  A fact moved twice (two rules consuming the same
    event fact, or an add and a delete of it inside one round) is reported once
    per change, never twice for one change.  The caller's submitted events are
    its own input and are not echoed.  ``rounds`` counts the rounds that fired
    (0 when the step was quiet).
    """

    fired: tuple  # (FireRecord, ...)
    rounds: int
    added: tuple  # ((pred, cols), ...)
    deleted: tuple
    actions: tuple  # (ActionRecord, ...)


def intern_facts(db, facts) -> list:
    """Turn compiled fact text (``"pred(a, b)."``) into ``(pred, cols)`` pairs.

    The same reading of a constant the compiler uses: a run of digits is an
    integer column, a double-quoted token is a symbol, and anything else is a
    bare symbol.  Accepts :attr:`le.errors.CompileResult.facts` directly.
    """
    out = []
    for text in facts:
        match = re.match(r"([a-z][A-Za-z0-9_]*)\((.*)\)\.\s*\Z", text)
        if not match:
            raise ValueError(f"not a dl fact: {text!r}")
        pred = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        out.append((pred, tuple(_col(db, a) for a in args)))
    return out


def _col(db, token: str) -> int:
    if _INT.match(token):
        return int(token)
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return db.intern(token[1:-1])
    return db.intern(token)


def _load_dlb():
    try:  # pragma: no cover - environment dependent
        import dlb
    except Exception as first_exc:  # pragma: no cover - environment dependent
        dlb_path = os.environ.get("DLB_PATH")
        if not dlb_path:
            raise RuntimeError(
                "the reactive driver needs the datalog engine binding: put the "
                "binding's package directory on sys.path or set DLB_PATH "
                "(import dlb failed: %s)" % (first_exc,)
            ) from first_exc
        sys.path.insert(0, dlb_path)
        try:
            import dlb
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the reactive driver needs the datalog engine binding: put the "
                "binding's package directory on sys.path or set DLB_PATH "
                "(import dlb failed: %s)" % (exc,)
            ) from exc
    return dlb


class ReactiveDriver:
    """Executes compiled reactive rules over a caller-supplied ``dlb.Db``."""

    def __init__(
        self,
        db,
        rules_text: str,
        reactive_rules: "tuple[ReactiveRule, ...]",
        *,
        relations: dict,
        action_preds,
        variadic=(),
        round_bound: int = DEFAULT_ROUND_BOUND,
    ) -> None:
        if round_bound < 1:
            raise ValueError("round_bound must be at least 1")
        self.db = db
        self.rules_text = rules_text
        self.reactive_rules = tuple(reactive_rules)
        self.relations = dict(relations)
        self.action_preds = frozenset(action_preds)
        # The predicates declared with the engine's VARIADIC relation form (one
        # column count per atom, MEASURED): the lexicon's [meta] reifying
        # relations, which hold mentions of predicates of different arities.
        self.variadic = frozenset(variadic)
        self.round_bound = round_bound
        self._entered = False
        # Per-rule row set of the previous observation (edge detection).
        self._prev: dict = {}

    @classmethod
    def from_compile(
        cls, db, result, lexicon: Lexicon, *, round_bound: int = DEFAULT_ROUND_BOUND
    ) -> "ReactiveDriver":
        """Build a driver from a :class:`CompileResult` and its lexicon."""
        if not isinstance(lexicon, Lexicon):
            raise TypeError(
                "lexicon must be a le.Lexicon (the driver needs every "
                "predicate's declared arity to declare the relations)"
            )
        relations: dict = {}
        for template in lexicon.all_templates:
            arity = len(template.sorts)
            previous = relations.setdefault(template.pred, arity)
            if previous != arity:
                raise ValueError(
                    f"predicate '{template.pred}' is declared with arity "
                    f"{previous} and {arity}"
                )
        # A [meta] reifying relation is VARIADIC (the engine's one-relation-many-
        # arities form) and is not an all_templates predicate, so the driver --
        # which has the lexicon -- declares it itself.  It stays loud: a caller
        # whose engine cannot hold the relation still gets enter()'s error.
        return cls(
            db,
            result.rules,
            result.reactive,
            relations=relations,
            action_preds=lexicon.action_preds,
            variadic=lexicon.meta_preds,
            round_bound=round_bound,
        )

    # -- setup -------------------------------------------------------------

    def enter(self) -> None:
        """Declare every relation, then load and compile the rule stream.

        Declaring up front means an unknown predicate fails HERE, loudly,
        rather than mid-run (the engine rejects a rule over an undeclared
        relation at compile time).
        """
        dlb = _load_dlb()
        if not isinstance(self.db, dlb.Db):
            raise TypeError(
                "db must be a dlb.Db opened by the caller (the driver never "
                "opens the database itself)"
            )
        for pred, arity in self.relations.items():
            self.db.declare_relation(pred, arity)
        for pred in sorted(self.variadic):
            self.db.declare_relation_variadic(pred)
        self.db.load_rules(self.rules_text)
        self.db.compile_rules()
        self._entered = True

    def seed(self, facts) -> None:
        """Add INITIAL-STATE facts directly -- no step, no observation.

        An iterable of ``(pred, cols)`` pairs (see :func:`intern_facts`).  The
        observation timeline starts at the EMPTY state, so anything already
        true in the seeded state is an edge at the first :meth:`submit`
        (call ``submit([])`` first to make the seeded state the baseline).
        """
        for pred, cols in facts:
            self.db.add_fact(pred, tuple(cols))

    # -- the step ----------------------------------------------------------

    def submit(self, events) -> StepResult:
        """Run one step: commit ``events``, then react to quiescence."""
        if not self._entered:
            raise RuntimeError("call enter() before submit()")
        events = [(pred, tuple(cols)) for pred, cols in events]
        for pred, cols in events:
            if pred not in self.action_preds:
                raise ValueError(
                    f"'{pred}' is not a predicate declared in the lexicon's "
                    "[actions] section: submit() takes external events only"
                )
        prev_snapshot = dict(self._prev)
        fired: list = []
        step_fired: set = set()
        added: list = []
        deleted: list = []
        actions: list = []
        undo: list = []
        # Effective presence per fact across the whole step: the committed state
        # cannot tell a second mutation that a buffered first one already made.
        touched: dict = {}
        try:
            rounds = self._react(
                events, undo, fired, step_fired, added, deleted, actions, touched
            )
        except BaseException:
            try:
                self._restore(undo)
            finally:
                self._prev = prev_snapshot
            raise
        return StepResult(
            tuple(fired), rounds, tuple(added), tuple(deleted), tuple(actions)
        )

    def run(self, batches, *, max_steps: int | None = None) -> list:
        """One step per event batch; raises rather than truncating silently."""
        batches = list(batches)
        if max_steps is not None and len(batches) > max_steps:
            raise ValueError(
                f"{len(batches)} event batches submitted with max_steps="
                f"{max_steps}: this would truncate the run silently"
            )
        return [self.submit(batch) for batch in batches]

    # -- internals ---------------------------------------------------------

    def _react(self, events, undo, fired, step_fired, added, deleted, actions,
               touched) -> int:
        committed = 0
        if events:
            self._commit_events(events, undo, touched)
        while True:
            # PHASE 1 of the round: observe EVERY rule's antecedent against the
            # state this round started from and collect the WHOLE firing plan,
            # before any consequence is applied.  Applying one rule's plan as
            # soon as it is found would let it consume an event fact that a
            # later rule's antecedent -- read from the same round-start state --
            # is entitled to.
            plan: list = []
            observed: dict = {}
            for rule in self.reactive_rules:
                rows = [
                    tuple(row)
                    for row in self.db.query(rule.antecedent_pred, collect=True)
                ]
                observed[rule.index] = set(rows)
                previous = self._prev.get(rule.index, frozenset())
                for row in rows:
                    if row in previous:
                        continue
                    if (rule.index, row) in step_fired:
                        raise StepError(
                            f"rule {rule.index} fired again for the same "
                            f"antecedent row {row} in round {committed + 1}: the "
                            "antecedent went false and true again inside one "
                            "step, so the reaction is oscillating; the step is "
                            "aborted"
                        )
                    plan.append((rule, row))
            if not plan:
                # The final observation becomes the baseline for the next step
                # (otherwise a row that is true again later would look like a
                # row that never went away -- a missed edge).
                self._prev.update(observed)
                return committed
            # Only rounds that FIRE are charged to the bound: a reaction that
            # settles in exactly round_bound firing rounds is not an overrun,
            # and this round would be firing round number bound + 1.
            if committed >= self.round_bound:
                raise StepError(
                    f"the reaction did not quiesce within {self.round_bound} "
                    "rounds (the round bound); the step is aborted"
                )
            # PHASE 2: apply the collected plan -- mutations and event
            # consumption together -- in the round's one transaction.
            self._fire(plan, undo, fired, step_fired, added, deleted, actions,
                       touched)
            self._prev.update(observed)
            committed += 1

    def _commit_events(self, events, undo, touched) -> None:
        txn = self.db.transaction()
        try:
            for pred, cols in events:
                self._apply(txn, "add", pred, cols, undo, None, touched)
        except BaseException:
            txn.rollback()
            raise
        txn.commit()

    def _fire(self, plan, undo, fired, step_fired, added, deleted, actions,
              touched) -> None:
        txn = self.db.transaction()
        try:
            for rule, row in plan:
                if rule.head_vars:
                    if len(row) != len(rule.head_vars):
                        raise StepError(
                            f"{rule.antecedent_pred} returned a row of arity "
                            f"{len(row)} but the schema declares "
                            f"{len(rule.head_vars)} head variables"
                        )
                    bind = dict(zip(rule.head_vars, row))
                else:
                    bind = {}
                for consequent in rule.consequents:
                    cols = self._cols(consequent.args, bind)
                    if consequent.kind == "action":
                        # A deliberate action: the world performs it, so the
                        # driver records it and leaves the state to the rule's
                        # explicit 'becomes' parts.
                        actions.append(ActionRecord(consequent.pred, cols))
                    elif consequent.kind == "add":
                        self._apply(txn, "add", consequent.pred, cols, undo,
                                    added, touched)
                    else:
                        self._apply(
                            txn, "delete", consequent.pred, cols, undo, deleted,
                            touched,
                        )
                for pred, args in rule.event_atoms:
                    # Consume the antecedent's event facts: a held antecedent
                    # must not re-fire on its own event.  When two rules in this
                    # plan read the same event, the fact is still reported (and
                    # logged) once, because the second consume changes nothing.
                    self._apply(txn, "delete", pred, self._cols(args, bind), undo,
                                deleted, touched)
                fired.append(FireRecord(rule.index, row))
                step_fired.add((rule.index, row))
        except BaseException:
            txn.rollback()
            raise
        txn.commit()

    def _apply(self, txn, op, pred, cols, undo, out, touched) -> None:
        """Buffer one mutation, logging its inverse when it changes the state.

        ``lookup`` reads the COMMITTED state even while ``txn`` is open
        (MEASURED), so it cannot see a mutation an earlier round or an earlier
        rule already buffered or committed in this step; ``touched`` carries the
        effective presence of every fact the step has moved, so a fact mutated
        twice is buffered, logged (its inverse, exactly once per change) and
        reported ONCE per change of presence.
        """
        key = (pred, cols)
        if key not in touched:
            touched[key] = self.db.lookup(pred, cols)
        if touched[key] == (op == "add"):
            return  # already in the requested state: nothing to buffer
        touched[key] = op == "add"
        undo.append(("delete" if op == "add" else "add", pred, cols))
        if out is not None:
            out.append((pred, cols))
        if op == "add":
            txn.add_fact(pred, cols)
        else:
            txn.delete_fact(pred, cols)

    def _restore(self, undo) -> None:
        if not undo:
            return
        txn = self.db.transaction()
        try:
            for op, pred, cols in reversed(undo):
                if op == "add":
                    txn.add_fact(pred, cols)
                else:
                    txn.delete_fact(pred, cols)
        except BaseException:
            txn.rollback()
            raise
        txn.commit()

    def _cols(self, args, bind) -> tuple:
        return tuple(
            bind[arg.text] if arg.kind == "var" else _col(self.db, arg.text)
            for arg in args
        )
