"""T5 -- negation: the compiler reorders, it never trusts source order.

MEASURES that 'it is not the case that X' lowers to '!atom', that the negated
atom is emitted AFTER the positive atom that binds its variables (the engine's
stratified-negation rule), and that a body made only of negation is refused
(the engine rejects it).
"""

import unittest

from fixtures import DAY, LEGAL_INI

from le import CompileError, compile_text
from le.cli import validate_dl
from le.lower import GAP_NEG_BINDING, GAP_NEG_ONLY

# The negated atom is written FIRST; the positive atom that binds V1 is SECOND.
REORDERED = (
    "A player is eligible if it is not the case that the player is excluded "
    "and the player plays a choice."
)


class TestNegation(unittest.TestCase):
    def test_negated_atom_moves_after_the_positive_atom(self):
        result = compile_text(REORDERED, LEGAL_INI)
        self.assertEqual(
            result.rules, "eligible(V1) :- plays(V1, V2), !excluded(V1).\n"
        )
        self.assertEqual(validate_dl(result.rules), ())

    def test_negated_atom_already_last_is_unchanged(self):
        result = compile_text(
            "A player is eligible if the player plays a choice and it is not "
            "the case that the player is excluded.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules, "eligible(V1) :- plays(V1, V2), !excluded(V1).\n"
        )

    def test_negation_first_moves_after_both_positive_atoms(self):
        result = compile_text(
            "A player is eligible if it is not the case that the player is "
            "excluded and the player plays a choice and the choice beats another "
            "choice.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "eligible(V1) :- plays(V1, V2), beats(V2, V3), !excluded(V1).\n",
        )

    def test_a_body_of_only_negation_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if it is not the case that the day "
                "is before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_ONLY)

    def test_a_variable_bound_only_inside_the_negation_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if the day is before Wednesday and "
                "it is not the case that a second day is before Wednesday.",
                DAY,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_BINDING)

    def test_an_arithmetic_producer_does_not_ground_a_negation(self):
        # The engine's 'unsafe negation' check counts POSITIVE-atom bindings
        # only, so a variable produced by '<N> days before' must not be
        # accepted as grounded here.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if a first day is before a second day "
                "and the day is 4 days before the second day and it is not the "
                "case that the day is a delivery day.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_BINDING)

    def test_negation_of_a_comparison_is_refused(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is a delivery day if it is not the case that the day "
                "is on or after Wednesday and the day is a delivery day.",
                LEGAL_INI,
            )
        self.assertIn("negating", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
