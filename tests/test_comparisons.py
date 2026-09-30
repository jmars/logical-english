"""T4 -- comparison copulas and the '<N> days before' arithmetic form.

MEASURES that copulas lower to dl comparison atoms (>=, <, >), that the
compiler places a comparison after the positive atoms that bind its operands
regardless of source order, that a symbol constant in an ordering comparison
is refused (the engine rejects it), and that '<N> days before' lowers to the
arithmetic producer form.
"""

import unittest

from fixtures import DAY, DAY_COPULA, LEGAL_INI

from le import CompileError, compile_text
from le.cli import validate_dl
from le.lower import GAP_SYMBOL_ORDER, GAP_TEMPORAL_ARITH, GAP_UNGROUNDED_CMP

# The comparison is written FIRST and the positive atom that grounds its
# operands SECOND: the emitted rule must invert them.
GE_SOURCE = (
    "A day is a delivery day if the day is on or after a second day and "
    "the day precedes the second day."
)


class TestComparisonCopulas(unittest.TestCase):
    def test_is_on_or_after_lowers_to_ge_and_moves_after_the_positives(self):
        result = compile_text(GE_SOURCE, DAY_COPULA)
        self.assertEqual(
            result.rules, "delivery_day(V1) :- precedes(V1, V2), V1 >= V2.\n"
        )

    def test_is_after_lowers_to_gt(self):
        result = compile_text(
            GE_SOURCE.replace("is on or after", "is after"), DAY_COPULA
        )
        self.assertEqual(
            result.rules, "delivery_day(V1) :- precedes(V1, V2), V1 > V2.\n"
        )

    def test_builtin_is_before_lowers_to_lt(self):
        result = compile_text(
            GE_SOURCE.replace("is on or after", "is before"), DAY_COPULA
        )
        self.assertEqual(
            result.rules, "delivery_day(V1) :- precedes(V1, V2), V1 < V2.\n"
        )

    def test_emit_is_structurally_valid_dl(self):
        for source in (
            GE_SOURCE,
            GE_SOURCE.replace("is on or after", "is after"),
            GE_SOURCE.replace("is on or after", "is before"),
        ):
            self.assertEqual(validate_dl(compile_text(source, DAY_COPULA).rules), ())

    def test_a_declared_before_template_wins_over_the_builtin_copula(self):
        result = compile_text(
            "A day is a delivery day if the day is before Wednesday.", DAY
        )
        self.assertEqual(
            result.rules, "delivery_day(V1) :- before(V1, wednesday).\n"
        )


class TestComparisonRefusals(unittest.TestCase):
    def test_symbol_constant_in_an_ordering_comparison_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is on or after Wednesday.",
                DAY_COPULA,
            )
        self.assertEqual(ctx.exception.gap, GAP_SYMBOL_ORDER)

    def test_comparison_of_two_different_sorts_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction commences on a day if the day is on or after "
                "the transaction.",
                LEGAL_INI,
            )
        self.assertIn("sort", str(ctx.exception))

    def test_ungrounded_comparison_operand_is_refused(self):
        # 'a second day' is introduced inside the comparison and grounded by
        # no positive atom: the engine would reject it, so must the compiler.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is on or after a second day "
                "and the day precedes a third day.",
                DAY_COPULA,
            )
        self.assertEqual(ctx.exception.gap, GAP_UNGROUNDED_CMP)


class TestDaysBeforeArithmetic(unittest.TestCase):
    def test_lowers_to_an_arithmetic_producer(self):
        result = compile_text(
            "A day is a delivery day if a first day is before a second day "
            "and the day is 3 days before the second day.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "delivery_day(V1) :- before(V2, V3), V1 = V3 - 3.\n",
        )
        self.assertEqual(validate_dl(result.rules), ())

    def test_subject_must_be_a_variable(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if a first day is before a second day "
                "and Wednesday is 3 days before the second day.",
                LEGAL_INI,
            )
        self.assertIn("variable", str(ctx.exception))

    def test_operand_must_be_numeric(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if a first day is before a second day "
                "and the day is 3 days before Wednesday.",
                LEGAL_INI,
            )
        self.assertIn("numeric", str(ctx.exception))

    def test_result_must_not_already_be_bound(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is before a second day "
                "and the day is 3 days before the second day.",
                LEGAL_INI,
            )
        self.assertIn("already bound", str(ctx.exception))

    def test_self_reference_is_refused(self):
        # 'V1 = V1 - 4' is not a producer of anything: the ordering never
        # makes it emittable, so it is a loud error rather than an emission.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is 4 days before the day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TEMPORAL_ARITH)
        self.assertIn("cycle", str(ctx.exception))

    def test_duplicate_producer_for_one_result_is_refused(self):
        # 'V1 = V2 - 3, V1 = V3 - 4' LOADS into the engine and answers with the
        # rows of one of the two producers -- silently wrong, so refuse.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if a first day is before a second day "
                "and the day is 3 days before the first day and the day is 4 "
                "days before the second day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TEMPORAL_ARITH)
        self.assertIn("produced twice", str(ctx.exception))

    def test_use_before_def_is_reordered_by_dependency(self):
        # The producers are written in the WRONG order ('the day' needs 'the
        # second day', which a later condition produces); the emitted body
        # must invert them, otherwise the engine rejects the operand.
        source = (
            "A day is a delivery day if the day is 3 days before a second day "
            "and the second day is 4 days before a third day and the third day "
            "is before Wednesday."
        )
        result = compile_text(source, LEGAL_INI)
        self.assertEqual(
            result.rules,
            "delivery_day(V1) :- before(V3, wednesday), V2 = V3 - 4, "
            "V1 = V2 - 3.\n",
        )
        self.assertEqual(validate_dl(result.rules), ())

    def test_multi_arith_chain_is_ordered_the_same_in_both_source_orders(self):
        for source in (
            # dependency order already
            "A day is a delivery day if a first day is before a second day "
            "and a third day is 4 days before the second day and the day is "
            "3 days before the third day.",
            # dependency order reversed
            "A day is a delivery day if a first day is before a second day "
            "and the day is 3 days before a third day and the third day is "
            "4 days before the second day.",
        ):
            result = compile_text(source, LEGAL_INI)
            self.assertEqual(
                result.rules,
                "delivery_day(V1) :- before(V2, V3), V4 = V3 - 4, "
                "V1 = V4 - 3.\n",
                source,
            )
            self.assertEqual(validate_dl(result.rules), ())

    def test_an_unbound_arith_operand_is_refused(self):
        # 'a third day' is introduced only by the producer's own operand and
        # no positive atom binds it: no ordering can make the producer
        # emittable.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is before a first day "
                "and a second day is 3 days before a third day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TEMPORAL_ARITH)


if __name__ == "__main__":
    unittest.main()
