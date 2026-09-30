"""T1 -- lexicon load + closed predicate set.

MEASURES the drift-control property: the compiler
is driven by a declared lexicon, and an undeclared predicate is a hard error,
never a silent no-op or an invented symbol.
"""

import unittest

from fixtures import DAY, LEGAL_INI

from le import CompileError, Lexicon, compile_text
from le.lexicon import GAP_AMBIGUOUS_TEMPLATE, GAP_IDENTIFIER, GAP_UNDECLARED_NAME
from le.lower import GAP_CLOSED_LEXICON, GAP_TYPE_CLASH


class TestLexiconLoad(unittest.TestCase):
    def test_loads_declared_sorts_and_templates(self):
        lex = Lexicon.load(LEGAL_INI)
        self.assertEqual(lex.sort_of("transaction"), "transaction")
        self.assertEqual(lex.sort_of("person"), "person")
        self.assertIsNone(lex.sort_of("widget"))
        preds = [t.pred for t in lex.templates]
        self.assertIn("governs", preds)
        self.assertIn("commences", preds)

    def test_unknown_section_is_a_loud_error(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text("[nouns]\nday = day\n")
        self.assertIn("unknown section", str(ctx.exception))

    def test_slot_and_sort_count_must_match(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is before <a day> | before | day\n"
            )
        self.assertIn("slot(s)", str(ctx.exception))

    def test_undeclared_sort_in_a_template_is_an_error(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is before <a day> | before | day, fruit\n"
            )
        self.assertIn("fruit", str(ctx.exception))

    def test_declaring_a_comparison_operator_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n<a day> is on or after <a day> | >= | day, day\n"
            )
        self.assertIn("built in", str(ctx.exception))

    def test_a_reserved_connective_in_a_template_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n"
                "<a day> and <a day> | both_days | day, day\n"
            )
        self.assertIn("reserved", str(ctx.exception))

    def test_duplicate_template_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n"
                "[predicates]\n"
                "<a day> is before <a day> | before | day, day\n"
                "<a day> is before <a day> | before | day, day\n"
            )
        self.assertIn("twice", str(ctx.exception))

    def test_an_uppercase_predicate_name_is_refused(self):
        # The engine parses an uppercase head as a VARIABLE, so 'Special' is
        # rejected at load time; the lexicon must not accept it either.
        for pred in ("Special", "_special", "Spec1ial"):
            with self.assertRaises(CompileError) as ctx:
                Lexicon.from_text(
                    "[sorts]\nday = day\n"
                    f"[predicates]\n<a day> is special <a day> | {pred} | day, day\n"
                )
            self.assertEqual(ctx.exception.gap, GAP_IDENTIFIER, pred)

    def test_declared_predicate_names_are_lowercase_identifiers(self):
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\n"
            "[predicates]\n<a day> is before <a day> | day_before_2 | day, day\n"
        )
        self.assertIn("day_before_2", [t.pred for t in lexicon.templates])


class TestLexiconNames(unittest.TestCase):
    def test_names_are_loaded_with_their_sorts(self):
        lexicon = Lexicon.load(LEGAL_INI)
        self.assertEqual(lexicon.sort_of_name("IsdaAgreement"), "agreement")
        self.assertEqual(lexicon.sort_of_name("Wednesday"), "day")
        self.assertIsNone(lexicon.sort_of_name("Nonesuch"))

    def test_a_names_entry_that_is_not_a_proper_noun_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n[names]\nwednesday = day\n"
                "[predicates]\n<a day> is before <a day> | before | day, day\n"
            )
        self.assertIn("proper noun", str(ctx.exception))

    def test_a_names_entry_must_be_name_equals_sort(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n[names]\nWednesday\n"
                "[predicates]\n<a day> is before <a day> | before | day, day\n"
            )
        self.assertIn("Name = sort", str(ctx.exception))

    def test_a_name_of_an_undeclared_sort_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(
                "[sorts]\nday = day\n[names]\nWednesday = month\n"
                "[predicates]\n<a day> is before <a day> | before | day, day\n"
            )
        self.assertIn("month", str(ctx.exception))


class TestProperNounTyping(unittest.TestCase):
    def test_a_wrong_sort_proper_noun_is_a_loud_type_clash(self):
        # 'Monday' is a day; 'plays' declares a CHOICE in slot 2.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A player plays Monday if the player is eligible.", LEGAL_INI
            )
        self.assertEqual(ctx.exception.gap, GAP_TYPE_CLASH)
        self.assertIn("choice", str(ctx.exception))

    def test_a_wrong_sort_proper_noun_in_the_second_slot_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by Wednesday if the transaction "
                "commences on a day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TYPE_CLASH)

    def test_an_undeclared_proper_noun_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("Foo commences on Wednesday.", LEGAL_INI)
        self.assertEqual(ctx.exception.gap, GAP_UNDECLARED_NAME)
        self.assertIn("[names]", str(ctx.exception))

    def test_a_correctly_sorted_proper_noun_compiles(self):
        result = compile_text("AcmeTransaction commences on Wednesday.", LEGAL_INI)
        self.assertEqual(result.facts, ("commences(acmetransaction, wednesday).",))


class TestClosedPredicateSet(unittest.TestCase):
    def test_declared_predicate_compiles(self):
        result = compile_text(
            "A day is a delivery day if the day is before Wednesday.", DAY
        )
        self.assertEqual(
            result.rules, "delivery_day(V1) :- before(V1, wednesday).\n"
        )
        self.assertEqual(result.facts, ())

    def test_undeclared_predicate_is_a_loud_error(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day flummoxes Wednesday.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_declared_predicate_of_another_lexicon_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text("A transaction is governed by IsdaAgreement.", DAY)
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_slot_sort_mismatch_is_a_loud_error(self):
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\ntransaction = transaction\n"
            "[predicates]\n"
            "<a transaction> commences on <a day> | commences | transaction, day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            compile_text("A day commences on Wednesday.", lexicon)
        self.assertEqual(ctx.exception.gap, GAP_TYPE_CLASH)
        self.assertIn("transaction", str(ctx.exception))
        self.assertIn("day", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
