"""T6 -- the error battery.

MEASURES that every feature outside the v1 subset (and every usage error the
article/ordinal binding exposes) is a LOUD CompileError naming the gap, never a
silent acceptance and never a silent drop.
"""

import unittest

from fixtures import DAY, DAY_COPULA, LEGAL_INI

from le import CompileError, Lexicon, compile_text
from le.binding import (
    GAP_BARE_NOUN,
    GAP_NAME_ON_REF,
    GAP_NAME_UNBOUND,
    GAP_REPEAT_A,
    GAP_SYMBOL_COLLISION,
    GAP_THE_OTHER,
    GAP_UNBOUND_THE,
)
from le.lexicon import GAP_AMBIGUOUS_TEMPLATE
from le.lower import (
    GAP_CLOSED_LEXICON,
    GAP_CONJUNCTIVE_CONCLUSION,
    GAP_CONSEQUENT_NOT_ACTION,
    GAP_CONSEQUENT_SHAPE,
    GAP_EXISTENTIAL_HEAD,
    GAP_FACT_GROUND,
    GAP_FLUENT_UPDATE,
    GAP_PLURALS,
    GAP_REACTIVE_FORM,
    GAP_RELATIVE_CLAUSE,
    GAP_TEMPORAL_ARITH,
    GAP_TEMPORAL_WHEN,
    GAP_TERMINATOR,
    GAP_THE_FUNCTIONAL,
)


class TestBindingErrors(unittest.TestCase):
    def test_unbound_the(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if the first day is before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_UNBOUND_THE)

    def test_repeated_a_for_one_sort(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if a day is before Wednesday.", DAY
            )
        self.assertEqual(ctx.exception.gap, GAP_REPEAT_A)

    def test_repeated_ordinal_names_the_ordinal_it_repeats(self):
        # 'a second day' twice must not advise the very form that just failed.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Monday if a second day is before a third day "
                "and a second day is before Monday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_REPEAT_A)
        self.assertIn("re-uses the ordinal 'second'", str(ctx.exception))

    def test_the_other_without_another(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if the other day is before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_THE_OTHER)

    def test_bare_common_noun(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before day.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_BARE_NOUN)

    def test_explicit_name_used_before_introduction(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("P1 is before Wednesday.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_NAME_UNBOUND)

    def test_explicit_name_on_a_back_reference(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day D1 is before Wednesday if the day D1 is before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_NAME_ON_REF)


class TestDeclaredGaps(unittest.TestCase):
    def test_reactive_consequent_must_be_a_declared_action(self):
        # 'receives' is declared [predicates] in this lexicon: static
        # knowledge cannot be performed by a rule.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then P1 receives a prize.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_NOT_ACTION)

    def test_reactive_without_then_is_a_loud_form_error(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("If a player P1 plays a choice C1.", LEGAL_INI)
        self.assertEqual(ctx.exception.gap, GAP_REACTIVE_FORM)

    def test_reactive_consequent_may_not_introduce_a_variable(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If a player P1 plays a choice C1 then it becomes the case "
                "that a game is over.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_SHAPE)

    def test_it_becomes_the_case_that(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "It becomes the case that a day is before Wednesday.", DAY
            )
        self.assertEqual(ctx.exception.gap, GAP_FLUENT_UPDATE)

    def test_meta_level_states_that_is_no_longer_a_gap(self):
        # The meta-level embedding RETIRED from this battery: the paper's own
        # sentence (2) now compiles, and 'governs' appears as a TERM (its
        # mention), never as a second body atom (the USE/MENTION distinction).
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a confirmation of "
            "the transaction states that the transaction is governed by "
            "IsdaAgreement.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(V2, governs, V1, isdaagreement).\n",
        )

    def test_an_undeclared_predicate_is_still_refused_inside_a_mention(self):
        # The closed-lexicon discipline holds INSIDE a meta condition: the
        # mention is matched against the declared templates like any atom.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that the transaction is confirmed "
                "by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_functional_the_points_at_paper_example_6(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is the day before the day "
                "before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_THE_FUNCTIONAL)

    def test_relative_clause(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is before a day that is "
                "before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_RELATIVE_CLAUSE)

    def test_when_form(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before Wednesday when a day is before Monday.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_TEMPORAL_WHEN)

    def test_quantifier(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if all days are before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_PLURALS)

    def test_plural_noun(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if days are before Wednesday.", DAY
            )
        self.assertEqual(ctx.exception.gap, GAP_PLURALS)

    def test_conjunctive_conclusion(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday and a day is before Monday if a day "
                "is before Tuesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_CONJUNCTIVE_CONCLUSION)

    def test_existential_head_variable(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before another day if a first day is before a second "
                "day.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_EXISTENTIAL_HEAD)

    def test_missing_terminator(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before Wednesday", DAY)
        self.assertEqual(ctx.exception.gap, GAP_TERMINATOR)

    def test_ontology_extension(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is sparkly.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)


class TestFactAndTemplateErrors(unittest.TestCase):
    def test_fact_with_a_variable(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before Wednesday.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_FACT_GROUND)

    def test_type_clash(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before a transaction.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_ambiguous_template_match(self):
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\n"
            "[names]\nWednesday = day\n"
            "[predicates]\n"
            "<dayA> is before <dayB> | before | day, day\n"
            "<a day> is before <dayB> | before_alt | day, day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day is before Wednesday.", lexicon)
        self.assertEqual(ctx.exception.gap, GAP_AMBIGUOUS_TEMPLATE)

    def test_temporal_arithmetic_requires_a_fresh_subject(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is before a second day and "
                "the day is 3 days before the second day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TEMPORAL_ARITH)

    def test_error_message_names_the_sentence(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day precedes Monday.\n"
                "A day flummoxes Monday.",
                DAY_COPULA,
            )
        self.assertIn("sentence 2", str(ctx.exception))
        self.assertIn("flummoxes", str(ctx.exception))


class TestFactsAndNames(unittest.TestCase):
    def test_ground_facts_go_to_the_fact_stream(self):
        result = compile_text("AcmeTransaction commences on Wednesday.", LEGAL_INI)
        self.assertEqual(result.rules, "")
        self.assertEqual(result.facts, ("commences(acmetransaction, wednesday).",))

    def test_a_fact_sentence_conjunction_is_two_facts(self):
        result = compile_text(
            "AcmeTransaction commences on Wednesday and IsdaAgreement is dated "
            "as of Monday.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.facts,
            ("commences(acmetransaction, wednesday).", "dated(isdaagreement, monday)."),
        )

    def test_explicit_variable_names_are_kept(self):
        result = compile_text(
            "A player P1 is eligible if P1 plays a choice C1.", LEGAL_INI
        )
        self.assertEqual(result.rules, "eligible(P1) :- plays(P1, C1).\n")

    def test_proper_nouns_fold_to_bare_dl_symbols(self):
        result = compile_text(
            "A transaction is governed by IsdaAgreement if the transaction "
            "commences on a first day and IsdaAgreement is dated as of a "
            "second day and the first day is on or after the second day.",
            LEGAL_INI,
        )
        self.assertIn("isdaagreement", result.rules)
        self.assertNotIn("IsdaAgreement", result.rules)

    def test_two_proper_nouns_folding_onto_one_symbol_are_refused(self):
        lexicon = Lexicon.from_text(
            "[sorts]\ntransaction = transaction\nday = day\n"
            "[names]\nAcorp = transaction\nACorp = transaction\nMonday = day\n"
            "[predicates]\n"
            "<a transaction> commences on <a day> | commences | transaction, day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            compile_text("Acorp commences on Monday and ACorp commences on Monday.", lexicon)
        self.assertEqual(ctx.exception.gap, GAP_SYMBOL_COLLISION)

    def test_the_fold_collision_check_covers_the_whole_document(self):
        # The fold map is per DOCUMENT, not per clause: two sentences that
        # would merge two distinct proper nouns onto one symbol are refused.
        lexicon = Lexicon.from_text(
            "[sorts]\ntransaction = transaction\nday = day\n"
            "[names]\nAcorp = transaction\nACorp = transaction\nMonday = day\n"
            "Tuesday = day\n"
            "[predicates]\n"
            "<a transaction> commences on <a day> | commences | transaction, day\n"
            "<a transaction> is dated as of <a day> | dated | transaction, day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "Acorp commences on Monday. ACorp is dated as of Tuesday.",
                lexicon,
            )
        self.assertEqual(ctx.exception.gap, GAP_SYMBOL_COLLISION)
        self.assertIn("sentence 2", str(ctx.exception))

    def test_distinct_proper_nouns_compile_in_separate_sentences(self):
        lexicon = Lexicon.from_text(
            "[sorts]\ntransaction = transaction\nday = day\n"
            "[names]\nAcorp = transaction\nBcorp = transaction\n"
            "Monday = day\nTuesday = day\n"
            "[predicates]\n"
            "<a transaction> commences on <a day> | commences | transaction, day\n"
            "<a transaction> is dated as of <a day> | dated | transaction, day\n"
        )
        result = compile_text(
            "Acorp commences on Monday. Bcorp is dated as of Tuesday.", lexicon
        )
        self.assertEqual(
            result.facts,
            ("commences(acorp, monday).", "dated(bcorp, tuesday)."),
        )


if __name__ == "__main__":
    unittest.main()
