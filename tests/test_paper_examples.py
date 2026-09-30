"""T2/T3 -- the paper's declarative examples.

T2 MEASURES the round-trip of Kowalski 2020 example (2), VERBATIM: the
declarative sentence and, with the meta-level ``states that`` condition, the
acceptance emission it fixes

    governs(V1, isdaagreement) :- states(V2, governs, V1, isdaagreement),
    commences(V1, V3), dated(isdaagreement, V4), V3 >= V4.

T3 MEASURES example (3)'s relational rewrite -- "Monday is the day before
another day and the other day is before Wednesday" -- i.e. ordinal/'another'
binding plus the of-chain form.
"""

import unittest

from fixtures import LEGAL_INI

from le import CompileError, compile_text
from le.binding import GAP_THE_OTHER_AMBIGUOUS

PAPER_2 = (
    "A transaction is governed by IsdaAgreement if the transaction commences "
    "on a first day and IsdaAgreement is dated as of a second day and the "
    "first day is on or after the second day."
)
PAPER_2_WITH_META = (
    "A transaction is governed by IsdaAgreement if a confirmation of the "
    "transaction states that the transaction is governed by IsdaAgreement and "
    "the transaction commences on a first day and IsdaAgreement is dated as of "
    "a second day and the first day is on or after the second day."
)
DAY_CHAIN = (
    "A day is before Wednesday if the day is before another day and the other "
    "day is before Wednesday."
)
OF_CHAIN = (
    "A confirmation of a transaction is accepted if the confirmation of the "
    "transaction is received."
)


class TestPaperExample2(unittest.TestCase):
    def test_declarative_subset_compiles_to_the_expected_clause(self):
        result = compile_text(PAPER_2, LEGAL_INI)
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- commences(V1, V2), "
            "dated(isdaagreement, V3), V2 >= V3.\n",
        )
        self.assertEqual(result.facts, ())

    def test_the_meta_level_embedding_is_no_longer_a_gap(self):
        # The paper's own sentence (2), verbatim, including the meta-level
        # 'states that' condition.  'governs' is USED as the conclusion's
        # predicate and MENTIONED as a term inside states(...).
        result = compile_text(PAPER_2_WITH_META, LEGAL_INI)
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- states(V2, governs, V1, "
            "isdaagreement), commences(V1, V3), dated(isdaagreement, V4), "
            "V3 >= V4.\n",
        )
        self.assertEqual(result.facts, ())

    def test_a_and_the_bind_the_same_variable(self):
        # 'a transaction' in the conclusion, 'the transaction' in the body:
        # one variable, per the paper's article rule.
        result = compile_text(
            "A transaction commences on a day if the transaction commences on "
            "the day.",
            LEGAL_INI,
        )
        self.assertEqual(result.rules, "commences(V1, V2) :- commences(V1, V2).\n")


class TestPaperExample3Rewrite(unittest.TestCase):
    def test_day_before_chain(self):
        result = compile_text(DAY_CHAIN, LEGAL_INI)
        self.assertEqual(
            result.rules, "before(V1, wednesday) :- before(V1, V2), before(V2, wednesday).\n"
        )

    def test_the_other_refers_to_another(self):
        # The SPEC: 'the other day' denotes the SAME entity as 'another day'
        # -- the chain must run through the variable 'another day' bound.
        result = compile_text(DAY_CHAIN, LEGAL_INI)
        self.assertEqual(
            result.rules,
            "before(V1, wednesday) :- before(V1, V2), before(V2, wednesday).\n",
        )

    def test_two_another_of_one_sort_make_the_other_ambiguous(self):
        # With two live 'another day's, 'the other day' has no unique
        # referent: a loud ambiguity, never a silent pick of the last one.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Tuesday if the day is before another day and "
                "another day is before Monday and the other day is before "
                "Tuesday.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_THE_OTHER_AMBIGUOUS)

    def test_of_chain_fills_both_slots(self):
        result = compile_text(OF_CHAIN, LEGAL_INI)
        self.assertEqual(result.rules, "accepted(V1, V2) :- received(V1, V2).\n")

    def test_the_other_without_another_is_a_loud_error(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A day is before Wednesday if the other day is before Wednesday.",
                LEGAL_INI,
            )
        self.assertIn("another", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
