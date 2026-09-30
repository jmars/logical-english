"""T11 -- the reactive ENGINE round-trip (SKIPS when the engine is absent).

MEASURES that a compiled reactive rule actually runs: the driver
(``le.driver``) asks the engine whether the antecedent holds, detects the edge
at which it BECAME true, fires the consequents as state changes, records the
actions it performs, consumes the antecedent's event facts, and terminates --
at quiescence, or loudly (``StepError``) with no partial state when a reaction
oscillates or exceeds the round bound.

dlb is imported here (the same gate as tests/test_engine_roundtrip.py): the
whole module skips -- it does not fail -- when the ``dlb`` binding or its
shared library is not importable, by plain import or via ``$DLB_PATH``.
"""

import shutil
import sys
import tempfile
import unittest

from fixtures import DLB_IMPORT_ERROR, ROOT, RPS, RPS_LE, RPS_RULE, dlb

sys.path.insert(0, ROOT)
_DLB_IMPORT_ERROR = DLB_IMPORT_ERROR  # as documented in le.driver's docstring

from le import Lexicon, compile_text
from le.driver import (
    ActionRecord,
    FireRecord,
    ReactiveDriver,
    StepError,
    intern_facts,
)

# -- fixtures ---------------------------------------------------------------

OSCILLATE = Lexicon.from_text(
    """
[sorts]
day = day

[predicates]
<a day> is a candidate | candidate | day
<a day> is marked | marked | day
<a day> is flushed | flushed | day
"""
)
OSCILLATE_DOC = (
    "If a day is a candidate and the day is marked then it becomes the case "
    "that the day is flushed and it becomes the case that it is not the case "
    "that the day is marked.\n"
    "If a day is a candidate and it is not the case that the day is marked "
    "then it becomes the case that the day is marked.\n"
)

CHAIN = Lexicon.from_text(
    """
[sorts]
day = day

[predicates]
<a day> is a start | is_start | day
<a day> is reached | is_reached | day
<dayA> precedes <dayB> | precedes | day, day
"""
)
CHAIN_DOC = (
    "If a day is a start then it becomes the case that the day is reached.\n"
    "If a day is reached and the day precedes a second day and it is not the "
    "case that the second day is reached then it becomes the case that the "
    "second day is reached.\n"
)

SERVED = Lexicon.from_text(
    """
[sorts]
player = player
choice = choice
prize = prize

[names]
RpsPrize = prize

[predicates]
<a player> is served | served | player

[actions]
<a player> plays <a choice> | plays | player, choice
<a player> receives <a prize> | receives | player, prize
"""
)
SERVED_DOC = (
    "If a player P1 plays a choice C1 then it becomes the case that P1 is "
    "served.\n"
    "If a player P1 is served then P1 receives RpsPrize.\n"
)

GUARDED = Lexicon.from_text(
    """
[sorts]
player = player
choice = choice
game = game
prize = prize

[names]
RpsGame = game
RpsPrize = prize

[predicates]
<a game> is over | over | game

[actions]
<a player> plays <a choice> | plays | player, choice
<a player> receives <a prize> | receives | player, prize
"""
)
GUARDED_DOC = (
    "If a player P1 plays a choice C1 and it is not the case that RpsGame "
    "is over then P1 receives RpsPrize.\n"
)

# An aborted step must roll the CALLER'S SUBMITTED EVENTS back too, so the
# event that seeds the cascade is a submitted [actions] fact, not add_fact.
ABORT_EVENT = Lexicon.from_text(
    """
[sorts]
day = day

[predicates]
<a day> is reached | is_reached | day
<dayA> precedes <dayB> | precedes | day, day

[actions]
<a day> starts | starts | day
"""
)
ABORT_EVENT_DOC = (
    "If a day starts then it becomes the case that the day is reached.\n"
    "If a day is reached and the day precedes a second day and it is not the "
    "case that the second day is reached then it becomes the case that the "
    "second day is reached.\n"
)

# Two rules collide on 'flushed' inside one round (add then delete, net zero)
# while the last two rules oscillate -- so the collision's compensating
# restore is exercised by the abort.
COLLIDE = Lexicon.from_text(
    """
[sorts]
day = day

[predicates]
<a day> is a candidate | candidate | day
<a day> is marked | marked | day
<a day> is flushed | flushed | day
"""
)
COLLIDE_DOC = (
    "If a day is a candidate and the day is marked then it becomes the case "
    "that the day is flushed.\n"
    "If a day is a candidate and the day is marked then it becomes the case "
    "that it is not the case that the day is flushed.\n"
    "If a day is a candidate and the day is marked then it becomes the case "
    "that it is not the case that the day is marked.\n"
    "If a day is a candidate and it is not the case that the day is marked "
    "then it becomes the case that the day is marked.\n"
)


def _lexicon(text):
    return Lexicon.from_text(text)


TWO_ACTIONS = _lexicon(
    """
[sorts]
player = player
choice = choice
game = game
prize = prize

[names]
RpsGame = game
RpsPrize = prize

[predicates]
<a game> is over | over | game

[actions]
<a player> plays <a choice> | plays | player, choice
<a player> receives <a prize> | receives | player, prize
<a player> scores | scores | player
"""
)

# A reactive rule whose ANTECEDENT reads a mention: the antecedent relation is
# the variadic [meta] one, which is not an all_templates predicate (NICE 4).
META_ANTECEDENT = _lexicon(
    """
[sorts]
transaction = transaction
confirmation = confirmation
agreement = agreement
player = player
prize = prize

[names]
IsdaAgreement = agreement
AcmePlayer = player
RpsPrize = prize

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[actions]
<a player> receives <a prize> | receives | player, prize

[meta]
<a confirmation> of <a transaction> states that <atom> | states | confirmation, transaction, _
"""
)
META_ANTECEDENT_DOC = (
    "If a confirmation of a transaction states that the transaction is "
    "governed by IsdaAgreement then AcmePlayer receives RpsPrize.\n"
)


class _EngineCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="le-reactive-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def open_db(self):
        if dlb is None:
            self.skipTest(f"dlb not importable: {_DLB_IMPORT_ERROR}")
        try:
            return dlb.Db.open(self.dir)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"datalog engine unavailable: {exc}")

    def driver(self, db, text, lexicon):
        result = compile_text(text, lexicon)
        driver = ReactiveDriver.from_compile(db, result, lexicon)
        driver.enter()
        return result, driver


class TestPaperExampleOne(_EngineCase):
    """Kowalski 2020 example (1) end to end."""

    def test_one_play_awards_the_prize_once(self):
        db = self.open_db()
        try:
            with open(RPS_LE, encoding="utf-8") as handle:
                source = handle.read()
            result, driver = self.driver(db, source, RPS)
            self.assertEqual(result.rules, RPS_RULE)
            driver.seed(intern_facts(db, result.facts))  # the beats relation
            p1, p2 = db.intern("p1"), db.intern("p2")
            rock, scissors = db.intern("rock"), db.intern("scissors")
            game, prize = db.intern("rpsgame"), db.intern("rpsprize")

            step = driver.submit([("plays", (p1, rock)), ("plays", (p2, scissors))])

            self.assertEqual(
                step.fired, (FireRecord(1, (p1, rock, p2, scissors)),)
            )
            self.assertEqual(step.rounds, 1)
            self.assertEqual(step.actions, (ActionRecord("receives", (p1, prize)),))
            # 'it becomes the case that the game is over' is the state change;
            # the performed action is an output, not a state write.
            self.assertEqual(step.added, (("over", (game,)),))
            self.assertEqual(
                step.deleted, (("plays", (p1, rock)), ("plays", (p2, scissors)))
            )
            self.assertTrue(db.lookup("over", (game,)))
            self.assertFalse(db.lookup("plays", (p1, rock)))
            self.assertFalse(db.lookup("plays", (p2, scissors)))
        finally:
            db.close()

    def test_the_same_events_after_the_game_are_over_do_nothing(self):
        db = self.open_db()
        try:
            with open(RPS_LE, encoding="utf-8") as handle:
                source = handle.read()
            result, driver = self.driver(db, source, RPS)
            driver.seed(intern_facts(db, result.facts))
            p1, p2 = db.intern("p1"), db.intern("p2")
            rock, scissors = db.intern("rock"), db.intern("scissors")
            events = [("plays", (p1, rock)), ("plays", (p2, scissors))]
            driver.submit(events)
            # The events were consumed, so re-submitting them makes the
            # antecedent true again -- and only the 'not over' guard blocks it.
            step = driver.submit(events)
            self.assertEqual(step.fired, ())
            self.assertEqual(step.actions, ())
            self.assertEqual(step.added, ())
        finally:
            db.close()

    def test_two_players_who_play_the_same_choice_do_not_fire(self):
        db = self.open_db()
        try:
            with open(RPS_LE, encoding="utf-8") as handle:
                source = handle.read()
            result, driver = self.driver(db, source, RPS)
            driver.seed(intern_facts(db, result.facts))
            p1, p2 = db.intern("p1"), db.intern("p2")
            rock = db.intern("rock")
            # The '!=' guard the compiler emitted for 'another player P2':
            # without it, P2 = P1 would make this a play with itself.
            step = driver.submit([("plays", (p1, rock)), ("plays", (p1, rock))])
            self.assertEqual(step.fired, ())
        finally:
            db.close()

    def test_seeding_then_an_empty_submit_fires_the_initial_edge(self):
        # A var-less antecedent (the engine rejects a 0-arity head, so the
        # compiler pads one constant column).
        db = self.open_db()
        try:
            lexicon = GUARDED
            source = (
                "If RpsGame is over then it becomes the case that it is not "
                "the case that RpsGame is over.\n"
            )
            result, driver = self.driver(db, source, lexicon)
            self.assertEqual(result.rules, "le_antecedent_1(0) :- over(rpsgame).\n")
            game = db.intern("rpsgame")
            driver.seed([("over", (game,))])
            step = driver.submit([])
            self.assertEqual(len(step.fired), 1)
            self.assertEqual(step.fired[0].row, (0,))
            self.assertEqual(step.deleted, (("over", (game,)),))
            self.assertFalse(db.lookup("over", (game,)))
        finally:
            db.close()


class TestStepMechanics(_EngineCase):
    """Firing order, simultaneous collisions, edge vs level."""

    def test_actions_are_recorded_in_document_order(self):
        db = self.open_db()
        try:
            prize = db.intern("rpsprize")
            p1 = db.intern("p1")
            rock = db.intern("rock")
            source = (
                "If a player P1 plays a choice C1 then P1 receives RpsPrize.\n"
                "If a player P1 plays a choice C1 then P1 scores.\n"
            )
            _result, driver = self.driver(db, source, TWO_ACTIONS)
            step = driver.submit([("plays", (p1, rock))])
            self.assertEqual(
                step.actions,
                (ActionRecord("receives", (p1, prize)), ActionRecord("scores", (p1,))),
            )
            self.assertEqual([record.rule_index for record in step.fired], [1, 2])
        finally:
            db.close()

    def test_same_fact_collisions_within_one_round_are_last_wins(self):
        add_then_delete = (
            "If a player P1 plays a choice C1 then it becomes the case that "
            "RpsGame is over.\n"
            "If a player P1 plays a choice C1 then it becomes the case that "
            "it is not the case that RpsGame is over.\n"
        )
        delete_then_add = (
            "If a player P1 plays a choice C1 then it becomes the case that "
            "it is not the case that RpsGame is over.\n"
            "If a player P1 plays a choice C1 then it becomes the case that "
            "RpsGame is over.\n"
        )
        for source, expected in ((add_then_delete, False), (delete_then_add, True)):
            db = self.open_db()
            try:
                game, p1, rock = db.intern("rpsgame"), db.intern("p1"), db.intern("rock")
                _result, driver = self.driver(db, source, GUARDED)
                step = driver.submit([("plays", (p1, rock))])
                self.assertEqual(len(step.fired), 2)
                self.assertEqual(
                    db.lookup("over", (game,)), expected, source
                )
            finally:
                db.close()
                self.setUp()

    def test_a_held_antecedent_does_not_re_fire(self):
        db = self.open_db()
        try:
            p1, rock = db.intern("p1"), db.intern("rock")
            prize = db.intern("rpsprize")
            _result, driver = self.driver(db, SERVED_DOC, SERVED)
            step = driver.submit([("plays", (p1, rock))])
            # Round 1: the play is consumed and 'served' is added.  Round 2:
            # rule 2's antecedent read 'served', which is a [predicates]
            # predicate, so it persists -- and the next step must NOT re-fire.
            self.assertEqual(
                [(record.rule_index, record.row) for record in step.fired],
                [(1, (p1, rock)), (2, (p1,))],
            )
            self.assertEqual(step.rounds, 2)
            self.assertEqual(step.actions, (ActionRecord("receives", (p1, prize)),))
            self.assertTrue(db.lookup("served", (p1,)))
            quiet = driver.submit([])
            self.assertEqual(quiet.fired, ())
            self.assertEqual(quiet.rounds, 0)
            self.assertTrue(db.lookup("served", (p1,)))  # the frame property
        finally:
            db.close()

    def test_an_arithmetic_grounded_head_variable_drives_the_consequent(self):
        db = self.open_db()
        try:
            lexicon = _lexicon(
                """
[sorts]
day = day

[names]
Wednesday = day

[predicates]
<dayA> is before <dayB> | before | day, day
<a day> is a delivery day | delivery_day | day
<a day> is flagged | flagged | day
"""
            )
            source = (
                "If a day is a delivery day and a second day is 3 days before "
                "a third day and the third day is before Wednesday then it "
                "becomes the case that the second day is flagged.\n"
            )
            _result, driver = self.driver(db, source, lexicon)
            wedge = db.intern("wednesday")
            db.add_fact("delivery_day", (7,))
            db.add_fact("before", (10, wedge))
            step = driver.submit([])
            # The head's arithmetic column (7) is what the consequent writes.
            self.assertEqual(step.fired, (FireRecord(1, (7, 7, 10)),))
            self.assertEqual(step.added, (("flagged", (7,)),))
            self.assertEqual(driver.submit([]).fired, ())
        finally:
            db.close()

    def test_a_shared_event_reaches_every_rule_that_reads_it(self):
        # Two rules read the SAME event predicate and both consume it.  The
        # round collects every rule's plan against the round-start state BEFORE
        # applying anything, so neither firing can starve the other: both fire in
        # ONE round, and the one event fact is consumed (and reported) ONCE.
        db = self.open_db()
        try:
            p1, rock = db.intern("p1"), db.intern("rock")
            source = (
                "If a player P1 plays a choice C1 and it is not the case that "
                "RpsGame is over then it becomes the case that RpsGame is "
                "over.\n"
                "If a player P1 plays a choice C1 and it is not the case that "
                "RpsGame is over then P1 receives RpsPrize.\n"
            )
            _result, driver = self.driver(db, source, GUARDED)
            step = driver.submit([("plays", (p1, rock))])
            self.assertEqual([record.rule_index for record in step.fired], [1, 2])
            self.assertEqual(step.rounds, 1)
            self.assertEqual(step.deleted, (("plays", (p1, rock)),))
        finally:
            db.close()

    def test_a_fact_moved_twice_in_one_round_is_reported_once(self):
        # Both the shared event consumption AND a duplicated consequent add are
        # reported once per change of presence, not once per rule: the second
        # mutation of a fact finds the state its predecessor already buffered.
        db = self.open_db()
        try:
            p1, rock = db.intern("p1"), db.intern("rock")
            game = db.intern("rpsgame")
            source = (
                "If a player P1 plays a choice C1 then it becomes the case that "
                "RpsGame is over.\n"
                "If a player P1 plays a choice C1 then it becomes the case that "
                "RpsGame is over.\n"
            )
            _result, driver = self.driver(db, source, GUARDED)
            step = driver.submit([("plays", (p1, rock))])
            self.assertEqual([record.rule_index for record in step.fired], [1, 2])
            self.assertEqual(step.added, (("over", (game,)),))
            self.assertEqual(step.deleted, (("plays", (p1, rock)),))
            self.assertTrue(db.lookup("over", (game,)))
        finally:
            db.close()

    def test_an_event_consumed_by_one_rule_is_not_resurrected_by_a_later_guard_clear(self):
        # The documented reading of an event as a ONE-OFF occurrence: rule 1
        # consumes the play (its antecedent was true), rule 2's guard was false
        # at that moment, and clearing the guard afterwards -- here in a later
        # step, after the world changes the state -- does NOT resurrect the
        # event, because rule 2's antecedent never became true.
        db = self.open_db()
        try:
            p1, rock, game = db.intern("p1"), db.intern("rock"), db.intern("rpsgame")
            source = (
                "If a player P1 plays a choice C1 then it becomes the case that "
                "RpsGame is over.\n"
                "If a player P1 plays a choice C1 and it is not the case that "
                "RpsGame is over then P1 receives RpsPrize.\n"
            )
            _result, driver = self.driver(db, source, GUARDED)
            db.add_fact("over", (game,))
            first = driver.submit([("plays", (p1, rock))])
            self.assertEqual([record.rule_index for record in first.fired], [1])
            self.assertEqual(first.deleted, (("plays", (p1, rock)),))

            db.delete_fact("over", (game,))
            after = driver.submit([])
            self.assertEqual(after.fired, ())
            self.assertEqual(after.actions, ())
            self.assertFalse(db.lookup("plays", (p1, rock)))
        finally:
            db.close()

    def test_an_antecedents_negation_turns_the_rule_off_then_on(self):
        db = self.open_db()
        try:
            p1, game, prize = db.intern("p1"), db.intern("rpsgame"), db.intern("rpsprize")
            rock, paper = db.intern("rock"), db.intern("paper")
            _result, driver = self.driver(db, GUARDED_DOC, GUARDED)
            first = driver.submit([("plays", (p1, rock))])
            self.assertEqual(first.actions, (ActionRecord("receives", (p1, prize)),))

            # The world makes the guard true: the antecedent is stratified-
            # negated, so a new play must no longer fire the rule.
            db.add_fact("over", (game,))
            blocked = driver.submit([("plays", (p1, paper))])
            self.assertEqual(blocked.fired, ())
            self.assertEqual(blocked.actions, ())
            # ... and the play was NOT consumed, because nothing fired.
            self.assertTrue(db.lookup("plays", (p1, paper)))

            # The guard clears: the pending play now becomes the edge.
            db.delete_fact("over", (game,))
            resumed = driver.submit([])
            self.assertEqual(len(resumed.fired), 1)
            self.assertEqual(resumed.actions, (ActionRecord("receives", (p1, prize)),))
            self.assertFalse(db.lookup("plays", (p1, paper)))
        finally:
            db.close()


class TestTermination(_EngineCase):
    """Every way a reaction can fail to settle is loud."""

    def test_an_oscillating_reaction_aborts_and_restores_the_state(self):
        db = self.open_db()
        try:
            _result, driver = self.driver(db, OSCILLATE_DOC, OSCILLATE)
            driver.seed([("candidate", (1,)), ("marked", (1,))])
            before = {
                (rel, db.lookup(rel, (1,)))
                for rel in ("candidate", "marked", "flushed")
            }
            with self.assertRaises(StepError) as ctx:
                driver.submit([])
            self.assertIn("oscillating", str(ctx.exception))
            after = {
                (rel, db.lookup(rel, (1,)))
                for rel in ("candidate", "marked", "flushed")
            }
            self.assertEqual(after, before)
            self.assertTrue(db.lookup("marked", (1,)))
            self.assertFalse(db.lookup("flushed", (1,)))
        finally:
            db.close()

    def test_a_growing_cascade_hits_the_round_bound(self):
        db = self.open_db()
        try:
            _result, driver = self.driver(db, CHAIN_DOC, CHAIN)
            driver.seed([("precedes", (i, i + 1)) for i in range(1, 150)])
            db.add_fact("is_start", (1,))
            with self.assertRaises(StepError) as ctx:
                driver.submit([])
            self.assertIn("100 rounds", str(ctx.exception))
            self.assertEqual(db.count("is_reached"), 0)
        finally:
            db.close()

    def test_a_finite_cascade_settles(self):
        db = self.open_db()
        try:
            _result, driver = self.driver(db, CHAIN_DOC, CHAIN)
            driver.seed([("precedes", (i, i + 1)) for i in range(1, 6)])
            db.add_fact("is_start", (1,))
            step = driver.submit([])
            self.assertEqual(step.rounds, 6)
            self.assertEqual(
                sorted(tuple(row) for row in db.query("is_reached", collect=True)),
                [(1,), (2,), (3,), (4,), (5,), (6,)],
            )
        finally:
            db.close()

    def test_a_cascade_that_settles_on_the_round_bound_completes(self):
        # 99 links settle in exactly 100 FIRING rounds.  The quiet observation
        # round that follows is not a firing round, so it must not be charged to
        # the bound (an off-by-one here would kill a correct cascade).
        db = self.open_db()
        try:
            _result, driver = self.driver(db, CHAIN_DOC, CHAIN)
            driver.seed([("precedes", (i, i + 1)) for i in range(1, 100)])
            db.add_fact("is_start", (1,))
            step = driver.submit([])
            self.assertEqual(step.rounds, 100)
            self.assertEqual(db.count("is_reached"), 100)
        finally:
            db.close()

    def test_a_cascade_one_round_past_the_round_bound_is_loud(self):
        # One link more needs firing round 101: still the bound, still loud.
        db = self.open_db()
        try:
            _result, driver = self.driver(db, CHAIN_DOC, CHAIN)
            driver.seed([("precedes", (i, i + 1)) for i in range(1, 101)])
            db.add_fact("is_start", (1,))
            with self.assertRaises(StepError) as ctx:
                driver.submit([])
            self.assertIn("100 rounds", str(ctx.exception))
            self.assertEqual(db.count("is_reached"), 0)
        finally:
            db.close()

    def test_an_aborted_step_restores_the_caller_submitted_events(self):
        # The undo log spans phase 1 as well, so an abort that follows a
        # submitted event restores the state the step STARTED from -- the event
        # itself is rolled back with the reactions it caused.
        db = self.open_db()
        try:
            _result, driver = self.driver(db, ABORT_EVENT_DOC, ABORT_EVENT)
            driver.seed([("precedes", (i, i + 1)) for i in range(1, 151)])
            with self.assertRaises(StepError):
                driver.submit([("starts", (1,))])
            self.assertFalse(db.lookup("starts", (1,)))
            self.assertEqual(db.count("is_reached"), 0)
            self.assertEqual(db.count("precedes"), 150)  # seeded state survives
        finally:
            db.close()

    def test_an_abort_restores_a_same_round_add_then_delete_collision(self):
        # Round 1 fires rules that add and then delete 'flushed' (net zero); the
        # abort in round 3 must still leave the pre-step state exactly.
        db = self.open_db()
        try:
            _result, driver = self.driver(db, COLLIDE_DOC, COLLIDE)
            driver.seed([("candidate", (1,)), ("marked", (1,))])
            with self.assertRaises(StepError) as ctx:
                driver.submit([])
            self.assertIn("oscillating", str(ctx.exception))
            self.assertTrue(db.lookup("candidate", (1,)))
            self.assertTrue(db.lookup("marked", (1,)))
            self.assertFalse(db.lookup("flushed", (1,)))
        finally:
            db.close()

    def test_run_refuses_to_truncate_silently(self):
        db = self.open_db()
        try:
            p1, p2 = db.intern("p1"), db.intern("p2")
            rock, scissors = db.intern("rock"), db.intern("scissors")
            with open(RPS_LE, encoding="utf-8") as handle:
                result = compile_text(handle.read(), RPS)
            driver = ReactiveDriver.from_compile(db, result, RPS)
            driver.enter()
            driver.seed(intern_facts(db, result.facts))
            batches = [[("plays", (p1, rock)), ("plays", (p2, scissors))]] * 3
            with self.assertRaises(ValueError):
                driver.run(batches, max_steps=2)
            steps = driver.run(batches, max_steps=3)
            self.assertEqual(len(steps), 3)
            self.assertEqual(len(steps[0].fired), 1)
            self.assertEqual(steps[1].fired, ())  # the game is over by then
        finally:
            db.close()

    def test_a_non_action_event_is_refused(self):
        db = self.open_db()
        try:
            with open(RPS_LE, encoding="utf-8") as handle:
                result = compile_text(handle.read(), RPS)
            driver = ReactiveDriver.from_compile(db, result, RPS)
            driver.enter()
            with self.assertRaises(ValueError):
                driver.submit([("beats", (1, 2))])
            with self.assertRaises(RuntimeError):
                ReactiveDriver.from_compile(db, result, RPS).submit([])
        finally:
            db.close()


class TestReactiveMetaAntecedent(_EngineCase):
    """NICE 4: the driver declares the variadic meta relation itself.

    ``from_compile`` used to declare only the ``all_templates`` predicates, so a
    reactive rule whose antecedent reads a mention failed LOUDLY at ``enter()``
    ('unknown predicate states') unless the caller declared the relation
    variadic by hand (README gap g10).  The driver has the lexicon, so it
    declares it -- and stays loud if the relation cannot be declared.
    """

    def test_the_driver_declares_the_variadic_meta_relation(self):
        db = self.open_db()
        try:
            result, driver = self.driver(db, META_ANTECEDENT_DOC, META_ANTECEDENT)
            self.assertEqual(
                result.rules,
                "le_antecedent_1(V1, V2) :- "
                "states(V1, governs, V2, isdaagreement).\n",
            )
            conf, t1 = db.intern("conf1"), db.intern("t1")
            isa = db.intern("isdaagreement")
            player, prize = db.intern("acmeplayer"), db.intern("rpsprize")
            driver.seed([("states", (conf, db.intern("governs"), t1, isa))])

            step = driver.submit([])

            self.assertEqual(step.fired, (FireRecord(1, (conf, t1)),))
            self.assertEqual(
                step.actions, (ActionRecord("receives", (player, prize)),)
            )
        finally:
            db.close()

    def test_a_mention_of_another_predicate_fires_nothing(self):
        # Control: the antecedent reads the reified ROW, so the predicate symbol
        # column has to be 'governs'.
        db = self.open_db()
        try:
            result, driver = self.driver(db, META_ANTECEDENT_DOC, META_ANTECEDENT)
            driver.seed(
                [(
                    "states",
                    (
                        db.intern("conf1"), db.intern("commences"),
                        db.intern("t1"), db.intern("isdaagreement"),
                    ),
                )]
            )
            driver.submit([])
            self.assertEqual(driver.submit([]).actions, ())
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
