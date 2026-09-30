"""T10 -- the reactive-rule surface (compiler side; no engine needed).

MEASURES the reactive sentence form of Kowalski 2020 example (1):
``If <antecedent> then <consequent>.`` compiles into (a) an
``le_antecedent_<i>`` detection rule in the ordinary rule stream, so the engine
can be asked whether the antecedent holds, and (b) a consequent schema in
``CompileResult.reactive`` that the driver executes.  The engine round-trip of
that schema is measured in tests/test_reactive_engine.py (dlb-gated).

Every malformed shape is a LOUD CompileError naming its gap -- never a silent
acceptance and never a silent drop.
"""

import json
import os
import subprocess
import sys
import unittest

from fixtures import EXAMPLES, RPS, RPS_INI, RPS_LE, RPS_RULE, RPS_SENTENCE

from le import CompileError, Lexicon, compile_text
from le.binding import GAP_REPEAT_A
from le.lexicon import GAP_ACTION_OVERLAP
from le.lower import (
    GAP_CLOSED_LEXICON,
    GAP_CONSEQUENT_NOT_ACTION,
    GAP_CONSEQUENT_SHAPE,
    GAP_FLUENT_UPDATE,
    GAP_NEG_ONLY,
    GAP_NEG_BINDING,
    GAP_REACTIVE_FORM,
)


def rps_document():
    with open(RPS_LE, encoding="utf-8") as handle:
        return handle.read()


class TestReactiveSurface(unittest.TestCase):
    def test_paper_example_1_compiles_to_its_antecedent_rule(self):
        result = compile_text(RPS_SENTENCE, RPS)
        self.assertEqual(result.rules, RPS_RULE)
        self.assertEqual(result.facts, ())
        self.assertEqual(len(result.reactive), 1)

    def test_the_schema_carries_the_head_order_and_consequents(self):
        rule = compile_text(RPS_SENTENCE, RPS).reactive[0]
        # Head order is source order (explicit names), so the row IS the
        # antecedent's substitution for the driver.
        self.assertEqual(rule.head_vars, ("P1", "C1", "P2", "C2"))
        self.assertEqual(rule.antecedent_pred, "le_antecedent_1")
        # 'plays' is an [actions] template, so the antecedent's play facts are
        # the events this rule consumes; 'beats'/'over' are state and are not.
        self.assertEqual(rule.event_preds, ("plays",))
        self.assertEqual(
            [
                (pred, [(arg.kind, arg.text) for arg in args])
                for pred, args in rule.event_atoms
            ],
            [
                ("plays", [("var", "P1"), ("var", "C1")]),
                ("plays", [("var", "P2"), ("var", "C2")]),
            ],
        )
        self.assertEqual(
            [(c.kind, c.pred, [a.text for a in c.args]) for c in rule.consequents],
            [
                ("action", "receives", ["P1", "rpsprize"]),
                ("add", "over", ["rpsgame"]),
            ],
        )

    def test_another_binds_a_distinctness_guard(self):
        # 'another player P2' claims P2 is not P1; the guard is what makes
        # that claim hold in the engine (MEASURED: var-var '!=' filters the
        # self-pair rows).  Two players, one guard.
        result = compile_text(RPS_SENTENCE, RPS)
        self.assertIn("P2 != P1", result.rules)
        self.assertEqual(result.rules.count("!="), 1)

    def test_ordinals_introduce_variables_without_an_inequality(self):
        # 'a choice C1 ... a choice C2' are distinct VARIABLES but not
        # claimed-distinct values: both players may play the same choice, so
        # no guard is emitted for them (the paper's own example relies on it).
        rules = compile_text(RPS_SENTENCE, RPS).rules
        self.assertNotIn("C2 != C1", rules)

    def test_an_unnamed_repeat_is_still_a_loud_usage_error(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player plays a choice and a player plays a choice then "
                "P1 receives RpsPrize.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_REPEAT_A)

    def test_the_consequent_shares_the_antecedents_bindings(self):
        # 'the second day' in the consequent refers to the antecedent's own
        # second day: the same BindingState spans both halves.
        rule = compile_text(
            "If a day is a start and the day precedes a second day then it "
            "becomes the case that the second day is over_started.",
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n"
                "<a day> is a start | is_start | day\n"
                "<dayA> precedes <dayB> | precedes | day, day\n"
                "<a day> is over_started | over_started | day\n"
            ),
        ).reactive[0]
        self.assertEqual(rule.head_vars, ("V1", "V2"))
        self.assertEqual(
            [(c.kind, c.pred, [a.text for a in c.args])
             for c in rule.consequents],
            [("add", "over_started", ["V2"])],
        )

    def test_delete_consequent(self):
        rule = compile_text(
            "If RpsGame is over then it becomes the case that it is not the "
            "case that RpsGame is over.",
            RPS,
        ).reactive[0]
        # No variables to carry: the engine rejects a 0-arity rule head
        # (MEASURED), so the head is padded with one constant column.
        self.assertEqual(rule.head_vars, ())
        self.assertEqual(
            compile_text(
                "If RpsGame is over then it becomes the case that it is not "
                "the case that RpsGame is over.",
                RPS,
            ).rules,
            "le_antecedent_1(0) :- over(rpsgame).\n",
        )
        self.assertEqual(rule.consequents[0].kind, "delete")

    def test_an_arithmetic_producer_orders_and_reaches_the_head(self):
        # The declarative body machinery ('N days before' with its dependency
        # order) is reusable in an antecedent unchanged, and the produced
        # variable is a head column so the driver can bind it.
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\n"
            "[names]\nWednesday = day\n"
            "[predicates]\n"
            "<dayA> is before <dayB> | before | day, day\n"
            "<a day> is a delivery day | delivery_day | day\n"
            "<a day> is flagged | flagged | day\n"
        )
        result = compile_text(
            "If a day is a delivery day and a second day is 3 days before a "
            "third day and the third day is before Wednesday then it becomes "
            "the case that the second day is flagged.",
            lexicon,
        )
        # Producers are emitted in dependency order and the head order is the
        # canonical one (V-names ascending), which is also the row order.
        self.assertEqual(
            result.rules,
            "le_antecedent_1(V1, V2, V3) :- delivery_day(V1), "
            "before(V3, wednesday), V2 = V3 - 3.\n",
        )
        self.assertEqual(result.reactive[0].head_vars, ("V1", "V2", "V3"))
        self.assertEqual(
            [a.text for a in result.reactive[0].consequents[0].args], ["V2"]
        )

    def test_a_mixed_document_keeps_document_order(self):
        source = (
            "A player receives RpsPrize if the player plays a choice.\n"
            "Rock beats Scissors.\n"
            + RPS_SENTENCE
        )
        result = compile_text(source, RPS)
        self.assertEqual(
            result.rules,
            "receives(V1, rpsprize) :- plays(V1, V2).\n" + RPS_RULE,
        )
        self.assertEqual(result.facts, ("beats(rock, scissors).",))
        self.assertEqual(len(result.reactive), 1)

    def test_two_reactive_rules_are_numbered_in_document_order(self):
        result = compile_text(RPS_SENTENCE + "\n" + RPS_SENTENCE, RPS)
        self.assertEqual(
            [rule.index for rule in result.reactive], [1, 2]
        )
        self.assertIn("le_antecedent_1(", result.rules)
        self.assertIn("le_antecedent_2(", result.rules)

    def test_consequences_are_not_engine_text(self):
        # The consequent schema is a driver instruction, never a dl rule: the
        # rule stream holds exactly the detection rule, so the performed
        # action never reaches the engine as a rule.
        result = compile_text(RPS_SENTENCE, RPS)
        self.assertEqual(len(result.rule_lines), 1)
        self.assertEqual(result.rule_lines, (RPS_RULE.strip(),))
        self.assertNotIn("receives", result.rules)


class TestReactiveGaps(unittest.TestCase):
    def test_a_reactive_sentence_without_then_is_loud(self):
        for source in (
            "If a player P1 plays a choice C1.",
            "If then P1 receives RpsPrize.",
            "If a player P1 plays a choice C1 then.",
            "If a player P1 plays a choice C1 then then P1 receives RpsPrize.",
        ):
            with self.assertRaises(CompileError) as ctx:
                compile_text(source, RPS)
            self.assertEqual(ctx.exception.gap, GAP_REACTIVE_FORM, source)

    def test_a_nested_if_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 and if C1 beats C1 then P1 "
                "receives RpsPrize.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_REACTIVE_FORM)

    def test_a_bare_consequent_must_be_a_declared_action(self):
        # With examples/lexicon.ini 'receives' is [predicates], not [actions].
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then P1 receives a prize.",
                os.path.join(EXAMPLES, "lexicon.ini"),
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_NOT_ACTION)

    def test_an_undeclared_consequent_predicate_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then P1 wins RpsPrize.", RPS
            )
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_a_consequent_that_introduces_a_term_is_loud(self):
        for source in (
            "If a player P1 plays a choice C1 then P1 receives a prize.",
            "If RpsGame is over then it becomes the case that a player "
            "receives RpsPrize.",
            "If a player P1 plays a choice C1 then it becomes the case that "
            "it is not the case that a game is over.",
        ):
            with self.assertRaises(CompileError) as ctx:
                compile_text(source, RPS)
            self.assertEqual(
                ctx.exception.gap, GAP_CONSEQUENT_SHAPE, source
            )

    def test_a_negated_consequent_without_becomes_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then it is not the case "
                "that RpsGame is over.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_SHAPE)

    def test_becomes_with_nothing_after_it_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then it becomes the case "
                "that.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_SHAPE)

    def test_it_becomes_the_case_that_stays_a_gap_in_declarative_position(self):
        for source in (
            "It becomes the case that RpsGame is over.",
            "A player receives RpsPrize if it becomes the case that RpsGame "
            "is over.",
        ):
            with self.assertRaises(CompileError) as ctx:
                compile_text(source, RPS)
            self.assertEqual(ctx.exception.gap, GAP_FLUENT_UPDATE, source)

    def test_it_becomes_the_case_that_stays_a_gap_in_the_antecedent(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If it becomes the case that RpsGame is over then P1 receives "
                "RpsPrize.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_FLUENT_UPDATE)

    def test_a_negation_only_antecedent_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If it is not the case that RpsGame is over then it becomes "
                "the case that RpsGame is over.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_ONLY)

    def test_an_ungrounded_negation_in_an_antecedent_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 and it is not the case that "
                "RpsGame is over and it is not the case that a game is over "
                "then P1 receives RpsPrize.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_BINDING)


class TestActionLexicon(unittest.TestCase):
    def test_the_actions_section_loads_and_is_queryable(self):
        self.assertTrue(RPS.is_action("plays"))
        self.assertTrue(RPS.is_action("receives"))
        self.assertFalse(RPS.is_action("beats"))
        self.assertEqual(
            [t.pred for t in RPS.actions], ["plays", "receives"]
        )
        # An [actions] template is still a declared template: it matches.
        self.assertIn("plays", [t.pred for t in RPS.all_templates])

    def test_a_template_may_not_be_both_a_predicate_and_an_action(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is marked | marked | day\n"
                "[actions]\n<a day> is marked | marked | day\n"
            )
        self.assertEqual(ctx.exception.gap, GAP_ACTION_OVERLAP)

    def test_a_predicate_name_may_not_be_declared_in_both_sections(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is marked | marked | day\n"
                "[actions]\n<a day> gets marked | marked | day\n"
            )
        self.assertEqual(ctx.exception.gap, GAP_ACTION_OVERLAP)

    def test_a_duplicate_action_template_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is marked | marked | day\n"
                "[actions]\n<a day> is stamped | stamped | day\n"
                "<a day> is stamped | stamped | day\n"
            )
        self.assertIn("twice", str(ctx.exception))

    def test_the_unknown_section_message_names_all_four(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text("[nouns]\nday = day\n")
        for name in ("[sorts]", "[names]", "[predicates]", "[actions]"):
            self.assertIn(name, str(ctx.exception))

    def test_an_action_template_is_type_checked_like_a_predicate(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\nplayer = player\n"
                "[predicates]\n<a day> is marked | marked | day\n"
                "[actions]\n<a player> plays <a day> | plays | player, month\n"
            )
        self.assertIn("month", str(ctx.exception))


class TestReactiveDeterminism(unittest.TestCase):
    def test_the_same_document_compiles_identically(self):
        first = compile_text(rps_document(), RPS_INI)
        second = compile_text(rps_document(), RPS_INI)
        self.assertEqual(first, second)
        self.assertEqual(first.rules, second.rules)
        self.assertEqual(first.reactive, second.reactive)

    def test_the_reactive_artefact_is_independent_of_the_hash_seed(self):
        script = (
            "import sys, json; sys.path.insert(0, %r);"
            "from le import compile_text;"
            "from le.lexicon import Lexicon;"
            "r = compile_text(open(%r).read(), Lexicon.load(%r));"
            "print(json.dumps([r.rules, list(r.facts), ["
            "  [x.index, list(x.head_vars), x.antecedent_pred,"
            "   [[p, [[a.kind, a.text] for a in args]] for p, args in x.event_atoms],"
            "   [[c.kind, c.pred, [[a.kind, a.text] for a in c.args]]"
            "    for c in x.consequents]]"
            "  for x in r.reactive]]))"
        ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), RPS_LE, RPS_INI)
        outputs = []
        for seed in ("0", "7", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                env=dict(os.environ, PYTHONHASHSEED=seed),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append(proc.stdout)
        self.assertEqual(len(set(outputs)), 1, outputs)
        self.assertEqual(json.loads(outputs[0])[0], RPS_RULE)


if __name__ == "__main__":
    unittest.main()
