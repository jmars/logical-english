"""T12 -- the META-LEVEL 'states that' embedding (Kowalski 2020 example 2).

MEASURES the USE/MENTION distinction the paper's paragraph 18 spells out: the
same predicate is USED at the object level (the conclusion's ``governs``) and
MENTIONED at the meta level (a TERM inside ``states(...)``).  A compiler that
compiled the mention as a body atom of the outer clause would silently invert
the semantics, so the tests pin the use/mention boundary three ways: textually
(``governs(`` occurs once, as the conclusion), structurally (the reified row),
and semantically (the engine round-trip in
``tests/test_engine_roundtrip.py::TestMetaEngineRoundtrip``, which derives
``governs`` from a POSTED mention and derives nothing from the three controls).

The encoding is M2, flat reification, through a VARIADIC relation:

    'a confirmation of the transaction states that the transaction is
     governed by IsdaAgreement'
        ->  states(V2, governs, V1, isdaagreement)

i.e. ``states(<subject terms>, <mentioned pred>, <its arguments>)`` -- the
mention reduced to a predicate symbol plus columns, so a function-free engine
can hold it.  A subject term the mention already carries is not repeated (the
paper's ``states(Conf, governs(T, A))``), and the arity cap is the engine's 8.
"""

import os
import re
import subprocess
import sys
import unittest

from fixtures import LEGAL_INI, ROOT

from le import CompileError, Lexicon, compile_text
from le.cli import _split_top_commas
from le.lexicon import GAP_ACTION_OVERLAP, GAP_RESERVED_WORD

# Every meta gap is reachable from le.lower, where the other gap tests look
# for the compiler's refusal vocabulary.
from le.lower import (
    GAP_CLOSED_LEXICON,
    GAP_CONSEQUENT_NOT_ACTION,
    GAP_FACT_GROUND,
    GAP_META_ARITY,
    GAP_META_SHAPE,
    GAP_META_VARIABLE_PRED,
    GAP_NEG_BINDING,
    GAP_TYPE_CLASH,
)

PAPER_2_META = (
    "A transaction is governed by IsdaAgreement if a confirmation of the "
    "transaction states that the transaction is governed by IsdaAgreement and "
    "the transaction commences on a first day and IsdaAgreement is dated as of "
    "a second day and the first day is on or after the second day."
)
PAPER_2_META_RULE = (
    "governs(V1, isdaagreement) :- states(V2, governs, V1, isdaagreement), "
    "commences(V1, V3), dated(isdaagreement, V4), V3 >= V4.\n"
)

# Two [meta] verb phrases, both reifying into the SAME 'states' relation (the
# variadic one), plus the 5- and 6-slot predicates the arity cap is measured
# with.
WIDE_INI = """
[sorts]
transaction = transaction
confirmation = confirmation
agreement = agreement
day = day
week = week
month = month
year = year
era = era

[names]
IsdaAgreement = agreement

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement
<a day> is filed in <a week> in <a month> in <a year> in <an era> | filed | day, week, month, year, era
<a day> is logged in <a week> in <a month> in <a year> in <an era> in <an era> | logged | day, week, month, year, era, era
<a transaction> is traced in <a week> in <a month> in <a year> in <an era> in <an era> | traced | transaction, week, month, year, era, era

[meta]
<a confirmation> of <a transaction> states that <atom> | states | confirmation, transaction, _
"""

# A second, independent meta verb phrase: the mechanism is a declared VERB
# CLASS, not the hardcoded string 'states'.
REQUIRES_INI = """
[sorts]
party = party
transaction = transaction
agreement = agreement

[names]
IsdaAgreement = agreement
AcmeParty = party

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[meta]
<a party> requires that <atom> | requires | party, _
"""

# A meta verb phrase with a reactive vocabulary beside it (test (h)).
META_ACTIONS_INI = """
[sorts]
transaction = transaction
confirmation = confirmation
agreement = agreement
player = player
prize = prize
game = game

[names]
IsdaAgreement = agreement
AcmePlayer = player
RpsPrize = prize
RpsGame = game

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement
<a game> is over | over | game

[actions]
<a player> receives <a prize> | receives | player, prize

[meta]
<a confirmation> of <a transaction> states that <atom> | states | confirmation, transaction, _
<a player> states that <atom> | states | player, _
"""

# A meta verb phrase with TWO subject slots, neither of which the mention
# carries: both stay columns of the reified row.
PAIR_INI = """
[sorts]
party = party
confirmation = confirmation
transaction = transaction
agreement = agreement

[names]
IsdaAgreement = agreement

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[meta]
<a party> alongside <a confirmation> states that <atom> | states | party, confirmation, _
"""

# A SINGLE-subject-slot meta verb phrase: the subject column can be observed in
# isolation (SHOULD-FIX 1, a constant subject is always a column), and the
# README's 'no subject column' extreme is reachable through it (a subject
# VARIABLE the mention carries back).  Its 'player' twin lives in its own
# lexicon below: two same-shape subject forms would match each other's subject
# (a slot's sort is checked only AFTER the template matches), which is the
# loud GAP_AMBIGUOUS_TEMPLATE, not a collision.
ONE_SLOT_INI = """
[sorts]
transaction = transaction
agreement = agreement

[names]
AcmeTransaction = transaction
IsdaAgreement = agreement

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[meta]
<a transaction> states that <atom> | states | transaction, _
"""

# The same verb phrase with a DIFFERENT subject sort: the collision the review
# measured is between these two readings, not between two readings of one.
PLAYER_VB_INI = """
[sorts]
transaction = transaction
agreement = agreement
player = player

[names]
AcmePlayer = player
AcmeTransaction = transaction
IsdaAgreement = agreement

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[meta]
<a player> states that <atom> | states | player, _
"""

# Two verb phrases in ONE lexicon whose subject FORMS differ (so neither is
# ambiguous) but whose rows collided: the 2-slot verb's transaction subject is
# carried back by the mention, and folding it dropped the verb's own 'alongside'
# prefix, leaving exactly the 1-slot verb's row.
PAIR_VB_INI = """
[sorts]
transaction = transaction
agreement = agreement
player = player

[names]
AcmePlayer = player
AcmeTransaction = transaction
IsdaAgreement = agreement

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement

[meta]
<a player> states that <atom> | states | player, _
<a transaction> alongside <a player> states that <atom> | states | transaction, player, _
"""

# The base every lexicon-declaration diagnosis is built on.
BASE_INI = (
    "[sorts]\nday = day\ntransaction = transaction\n"
    "confirmation = confirmation\n"
    "[predicates]\n<dayA> is before <dayB> | before | day, day\n"
)


def _lexicon(text: str) -> Lexicon:
    return Lexicon.from_text(text)


class TestPaperExampleTwo(unittest.TestCase):
    """(a) the verbatim paper example; (a2) the shared variables."""

    def test_paper_example2_compiles_verbatim_to_the_acceptance_emission(self):
        # (a) EXACT emission: the paper's sentence (2) with the meta condition,
        # and no facts.
        result = compile_text(PAPER_2_META, LEGAL_INI)
        self.assertEqual(result.rules, PAPER_2_META_RULE)
        self.assertEqual(result.facts, ())

    def test_the_mention_shares_the_outer_clause_variables(self):
        # (a2) 'the transaction' inside the mention IS the head's V1 (one
        # BindingState per clause), so V1 occurs in the head, in the reified
        # 'states' row AND in 'commences'.
        result = compile_text(PAPER_2_META, LEGAL_INI)
        occurrences = re.findall(r"\bV1\b", result.rules)
        self.assertEqual(len(occurrences), 3, result.rules)
        self.assertIn("states(V2, governs, V1, isdaagreement)", result.rules)
        self.assertIn("commences(V1, V3)", result.rules)

    def test_a_subject_slot_the_mention_does_not_carry_is_a_column(self):
        # The fold is exact, not a blanket drop of the subject slots: neither
        # 'a party' nor 'a confirmation' is in the mention, so both are columns
        # -- while the transaction the mention carries appears once.
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a party alongside a "
            "confirmation states that the transaction is governed by "
            "IsdaAgreement.",
            _lexicon(PAIR_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(V2, V3, governs, V1, isdaagreement).\n",
        )

    def test_a_sort_clash_inside_the_mention_is_loud(self):
        # (a3) the mention's own template types its own arguments.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that Wednesday is governed by "
                "IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_TYPE_CLASH)


class TestUseMention(unittest.TestCase):
    """(b)/(c) the boundary is structural and the lexicon stays closed."""

    def test_the_mention_never_becomes_a_body_atom(self):
        # (b) The body of the compiled clause is EXACTLY the three positive
        # atoms; 'governs' occurs once, as the HEAD.  A mention compiled as a
        # conjunction of the outer clause would show a second body atom here.
        rules = compile_text(PAPER_2_META, LEGAL_INI).rules
        head, body = rules.rstrip(".\n").split(" :- ")
        self.assertEqual(head, "governs(V1, isdaagreement)")
        self.assertEqual(
            [atom.strip() for atom in _split_top_commas(body)],
            [
                "states(V2, governs, V1, isdaagreement)",
                "commences(V1, V3)",
                "dated(isdaagreement, V4)",
                "V3 >= V4",
            ],
        )
        self.assertEqual(rules.count("governs("), 1, rules)
        # ... and the mentioned predicate appears as a BARE symbol (a term),
        # never as a call.
        self.assertIn(", governs, ", rules)

    def test_an_undeclared_mentioned_predicate_is_a_loud_closed_lexicon_error(self):
        # (c) the closed lexicon holds inside the mention.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that the transaction is confirmed "
                "by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_a_meta_verb_phrase_without_a_matching_subject_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "states that the transaction is governed by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_META_SHAPE)


class TestArityCap(unittest.TestCase):
    """(d) subject slots + 1 + the mentioned arity, against the engine's 8."""

    def test_a_five_argument_mention_is_eight_columns_and_compiles(self):
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a confirmation of "
            "the transaction states that a day is filed in a week in a month "
            "in a year in an era.",
            _lexicon(WIDE_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(V2, V1, filed, V3, V4, V5, V6, V7).\n",
        )
        row = result.rules.split("states(")[1].split(")")[0]
        self.assertEqual(len(row.split(", ")), 8)

    def test_a_six_argument_mention_is_the_loud_arity_gap(self):
        # 2 subject slots + 1 predicate column + 6 arguments = 9, and the
        # mention carries neither subject term back, so no fold can save it.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that a day is logged in a week in a "
                "month in a year in an era in a second era.",
                _lexicon(WIDE_INI),
            )
        self.assertEqual(ctx.exception.gap, GAP_META_ARITY)
        message = str(ctx.exception)
        self.assertIn("9", message)
        self.assertIn("8", message)
        self.assertIn("logged", message)
        self.assertNotIn("folded", message)

    def test_a_mention_that_carries_a_subject_argument_back_still_fits(self):
        # The cap is checked on the FOLDED row, not on the declared shape: this
        # mention's first argument IS the meta condition's own transaction, so
        # the transaction subject is not repeated and the row is
        # states(V2, traced, V1, V3..V7) = 8 columns.  Counting the two declared
        # subject slots refused it with '9 columns' (the reviewer's measured
        # false refusal).
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a confirmation of "
            "the transaction states that the transaction is traced in a week "
            "in a month in a year in an era in a second era.",
            _lexicon(WIDE_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(V2, traced, V1, V3, V4, V5, V6, V7).\n",
        )
        row = result.rules.split("states(")[1].split(")")[0]
        self.assertEqual(len(row.split(", ")), 8)


class TestMentionShape(unittest.TestCase):
    """(e) everything that is not a positive DECLARED atom is a loud gap."""

    def _gap(self, condition: str):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if " + condition,
                LEGAL_INI,
            )
        return ctx.exception

    def test_a_comparison_mention_is_not_a_predicate_mention(self):
        exc = self._gap(
            "a confirmation of the transaction states that the first day is on "
            "or after a second day."
        )
        self.assertEqual(exc.gap, GAP_META_SHAPE)
        self.assertIn("comparison", str(exc))

    def test_an_arithmetic_mention_is_not_a_predicate_mention(self):
        exc = self._gap(
            "a confirmation of the transaction states that a first day is 3 "
            "days before a second day."
        )
        self.assertEqual(exc.gap, GAP_META_SHAPE)

    def test_a_negated_mention_is_loud(self):
        exc = self._gap(
            "a confirmation of the transaction states that it is not the case "
            "that the transaction is governed by IsdaAgreement."
        )
        self.assertEqual(exc.gap, GAP_META_SHAPE)
        self.assertIn("negated", str(exc))

    def test_an_empty_mention_is_loud(self):
        exc = self._gap("a confirmation of the transaction states that.")
        self.assertEqual(exc.gap, GAP_META_SHAPE)

    def test_a_coordination_fragment_that_is_not_an_atom_is_loud(self):
        # 'and' splits the condition list BEFORE the meta match runs, and the
        # meta condition consumes exactly ONE atom: here the fragment after
        # 'and' is not a standalone atom, so the closed lexicon refuses it by
        # name -- never a silent second conjunct.  The message names the
        # fragment rather than the coordination (README, residual).
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that the transaction commences on a "
                "first day and is governed by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_a_coordination_fragment_that_is_an_atom_is_an_ordinary_condition(self):
        # The other half of the same rule, pinned rather than left implicit: a
        # fragment that IS a declared atom compiles as an ordinary CONDITION of
        # the clause, exactly as the paper's example (2) requires (its meta
        # condition is followed by three further conjuncts).  The mention
        # therefore reifies the FIRST conjunct only -- a mention of a
        # coordination has no encoding in v1 (README).
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a confirmation of "
            "the transaction states that the transaction commences on a first "
            "day and the transaction commences on a second day.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(V2, commences, V1, V3), commences(V1, V4).\n",
        )
        # Use/mention integrity still holds: the mentioned predicate of the
        # first conjunct is a term inside 'states', never a body atom.
        self.assertEqual(result.rules.count("commences("), 1, result.rules)


class TestNegatedMetaCondition(unittest.TestCase):
    """(f) a meta condition negated as a whole is an ordinary '!atom'."""

    def test_a_grounded_negated_meta_condition_emits_a_negated_states_atom(self):
        result = compile_text(
            "A confirmation of a transaction is accepted if the confirmation "
            "of the transaction is received and it is not the case that the "
            "confirmation of the transaction states that the transaction is "
            "governed by IsdaAgreement.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "accepted(V1, V2) :- received(V1, V2), "
            "!states(V1, governs, V2, isdaagreement).\n",
        )

    def test_an_ungrounded_negated_meta_condition_is_loud(self):
        # 'a second confirmation' occurs only inside the negation, so nothing
        # binds V3 positively (the engine's stratified-negation rule).
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A confirmation of a transaction is accepted if the "
                "confirmation of the transaction is received and it is not the "
                "case that a second confirmation of the transaction states "
                "that the transaction is governed by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_NEG_BINDING)


class TestMetaFacts(unittest.TestCase):
    """(g) a mention can be POSTED as a ground fact."""

    def test_a_meta_fact_with_proper_noun_subjects_is_ground(self):
        result = compile_text(
            "AcmeConfirmation of AcmeTransaction states that AcmeTransaction "
            "is governed by IsdaAgreement.",
            LEGAL_INI,
        )
        self.assertEqual(result.rules, "")
        # The fold keys on VARIABLE identity, so neither proper-noun subject is
        # dropped: 'acmetransaction' is a column of the subject prefix AND an
        # argument of the mention.  Folding it away would let the 2-slot
        # '<a confirmation> of <a transaction>' verb reify onto the same row as
        # a 1-slot verb (see TestSubjectFold for that collision).
        self.assertEqual(
            result.facts,
            (
                "states(acmeconfirmation, acmetransaction, governs, "
                "acmetransaction, isdaagreement).",
            ),
        )

    def test_a_meta_fact_with_a_variable_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A confirmation of AcmeTransaction states that AcmeTransaction "
                "is governed by IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_FACT_GROUND)


class TestSubjectFold(unittest.TestCase):
    """SHOULD-FIX 1 -- the fold keys on VARIABLE identity.

    A subject term the mention carries back is dropped from the subject prefix
    only when it is the SAME VARIABLE (one entity, one column).  A CONSTANT
    subject is always kept: folding it would vanish the subject column and let
    two different ``[meta]`` verb phrases reify onto the same row, so a fact
    posted through one reading would be silently read by a rule compiled
    through the other (the review's measured collision).
    """

    def test_a_constant_subject_is_always_a_column(self):
        # Reviewer's measured input: before the fix this emitted
        # 'states(governs, acmetransaction, isdaagreement)' -- no subject.
        result = compile_text(
            "AcmeTransaction states that AcmeTransaction is governed by "
            "IsdaAgreement.",
            _lexicon(ONE_SLOT_INI),
        )
        self.assertEqual(
            result.facts,
            (
                "states(acmetransaction, governs, acmetransaction, "
                "isdaagreement).",
            ),
        )

    def test_two_verb_phrases_with_different_constant_subjects_do_not_collide(self):
        # The cross-template collision: with a constant subject folded, both
        # readings emitted the SAME 3-column row, so nothing in the row said
        # which verb phrase had posted it -- a fact posted through one reading
        # would be silently read by a rule compiled through the other.
        via_transaction = compile_text(
            "AcmeTransaction states that AcmeTransaction is governed by "
            "IsdaAgreement.",
            _lexicon(ONE_SLOT_INI),
        )
        via_player = compile_text(
            "AcmePlayer states that AcmeTransaction is governed by "
            "IsdaAgreement.",
            _lexicon(PLAYER_VB_INI),
        )
        self.assertNotEqual(via_transaction.facts, via_player.facts)
        self.assertEqual(
            via_player.facts,
            (
                "states(acmeplayer, governs, acmetransaction, isdaagreement).",
            ),
        )

    def test_a_two_slot_verb_does_not_collide_with_a_one_slot_verb(self):
        # The collision INSIDE one lexicon, and it needs no ambiguity: the
        # subject forms differ in length, so each input matches exactly one verb.
        # Pre-fix both of these emitted 'states(acmeplayer, governs,
        # acmetransaction, isdaagreement)' -- the 2-slot verb's own transaction
        # column was folded away, leaving the 1-slot verb's row.
        lexicon = _lexicon(PAIR_VB_INI)
        one_slot = compile_text(
            "AcmePlayer states that AcmeTransaction is governed by "
            "IsdaAgreement.",
            lexicon,
        )
        two_slot = compile_text(
            "AcmeTransaction alongside AcmePlayer states that AcmeTransaction "
            "is governed by IsdaAgreement.",
            lexicon,
        )
        self.assertNotEqual(one_slot.facts, two_slot.facts)
        self.assertEqual(
            two_slot.facts,
            (
                "states(acmetransaction, acmeplayer, governs, "
                "acmetransaction, isdaagreement).",
            ),
        )

    def test_a_subject_variable_the_mention_carries_yields_no_subject_column(self):
        # The README's documented extreme, reached with a VARIABLE subject: the
        # mention already carries every subject term, so the row is
        # states(pred, args...).
        result = compile_text(
            "A transaction is governed by IsdaAgreement if the transaction "
            "states that the transaction is governed by IsdaAgreement.",
            _lexicon(ONE_SLOT_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "states(governs, V1, isdaagreement).\n",
        )


class TestMetaAsHead(unittest.TestCase):
    """A rule may CONCLUDE a mention, not only read one (the embedding's
    step-2 design).

    Nothing in the encoding distinguishes a head position from a body or a fact
    position: the meta condition lowers to an ordinary positive atom, so
    'states(...)' can be derived by a rule exactly as it can be posted by a
    fact.
    """

    def test_a_rule_can_conclude_a_mention(self):
        result = compile_text(
            "A confirmation of a transaction states that the transaction is "
            "governed by IsdaAgreement if the confirmation of the transaction "
            "is received.",
            LEGAL_INI,
        )
        self.assertEqual(
            result.rules,
            "states(V1, governs, V2, isdaagreement) :- received(V1, V2).\n",
        )
        self.assertEqual(result.facts, ())

    def test_an_ungrounded_mention_head_is_the_ordinary_head_error(self):
        # The head is a positive atom, so it obeys the ordinary head rule: a
        # mention whose only variable is the conclusion's own subject is not
        # grounded.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A confirmation of a transaction states that a second "
                "transaction is governed by IsdaAgreement if the confirmation "
                "of the transaction is received.",
                LEGAL_INI,
            )
        self.assertTrue(ctx.exception.gap)


class TestReactiveMeta(unittest.TestCase):
    """(h) the meta machinery is reachable from the reactive form."""

    def test_a_meta_condition_in_a_reactive_antecedent_lowers_through_it(self):
        result = compile_text(
            "If a confirmation of a transaction states that the transaction "
            "is governed by IsdaAgreement then AcmePlayer receives RpsPrize.",
            _lexicon(META_ACTIONS_INI),
        )
        self.assertEqual(
            result.rules,
            "le_antecedent_1(V1, V2) :- "
            "states(V1, governs, V2, isdaagreement).\n",
        )
        self.assertEqual(result.reactive[0].head_vars, ("V1", "V2"))

    def test_a_meta_atom_as_a_reactive_consequent_is_not_an_action(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "If AcmePlayer receives RpsPrize then a confirmation of a "
                "transaction states that the transaction is governed by "
                "IsdaAgreement.",
                _lexicon(META_ACTIONS_INI),
            )
        self.assertEqual(ctx.exception.gap, GAP_CONSEQUENT_NOT_ACTION)


class TestLexiconMetaDeclarations(unittest.TestCase):
    """(i) every malformed [meta] declaration is loud at load time."""

    def _load(self, meta: str):
        with self.assertRaises(CompileError) as ctx:
            _lexicon(BASE_INI + "[meta]\n" + meta + "\n")
        return ctx.exception

    def test_the_atom_slot_must_be_last(self):
        exc = self._load("<a day> states that the day | states | day")
        self.assertEqual(exc.gap, GAP_META_SHAPE)
        self.assertIn("<atom>", str(exc))

    def test_the_atom_slot_sort_must_be_underscore(self):
        exc = self._load("<a day> states that <atom> | states | day, day")
        self.assertEqual(exc.gap, GAP_META_SHAPE)
        self.assertIn("_", str(exc))

    def test_the_verb_phrase_must_end_in_that(self):
        exc = self._load("<a day> states <atom> | states | day, _")
        self.assertEqual(exc.gap, GAP_META_SHAPE)
        self.assertIn("that", str(exc))

    def test_a_slot_where_the_verb_phrase_belongs_is_a_variable_predicate(self):
        exc = self._load("<a day> states <a phrase> <atom> | states | day, day, _")
        self.assertEqual(exc.gap, GAP_META_VARIABLE_PRED)

    def test_the_reifying_predicate_may_not_collide_with_another_section(self):
        exc = self._load("<a day> states that <atom> | before | day, _")
        self.assertEqual(exc.gap, GAP_ACTION_OVERLAP)

    def test_that_may_not_be_a_template_literal_elsewhere(self):
        with self.assertRaises(CompileError) as ctx:
            _lexicon(
                BASE_INI + "<a day> is that <a day> | weird | day, day\n"
            )
        self.assertEqual(ctx.exception.gap, GAP_RESERVED_WORD)

    def test_the_declared_meta_predicates_are_exposed(self):
        lexicon = _lexicon(
            BASE_INI
            + "[meta]\n<a day> states that <atom> | states | day, _\n"
        )
        self.assertEqual(lexicon.meta_preds, frozenset({"states"}))
        self.assertEqual([tpl.pred for tpl in lexicon.meta], ["states"])
        # ... and a meta template is NOT matchable as an ordinary atom, so
        # match_atom's behaviour (and every existing output) is untouched.
        self.assertNotIn("states", [t.pred for t in lexicon.all_templates])


class TestSecondMetaVerb(unittest.TestCase):
    """(j) a declared verb CLASS -- the mechanism is not the string 'states'."""

    def test_a_second_meta_verb_compiles_the_same_way(self):
        result = compile_text(
            "A transaction is governed by IsdaAgreement if a party requires "
            "that the transaction is governed by IsdaAgreement.",
            _lexicon(REQUIRES_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "requires(V2, governs, V1, isdaagreement).\n",
        )

    def test_a_second_meta_verb_keeps_its_own_subject_slot(self):
        # The subject slot is a real column (a party requires that P says WHO
        # requires it), unlike the shared 'of' slot of the paper's verb.
        result = compile_text(
            "A transaction is governed by IsdaAgreement if AcmeParty requires "
            "that the transaction is governed by IsdaAgreement.",
            _lexicon(REQUIRES_INI),
        )
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- "
            "requires(acmeparty, governs, V1, isdaagreement).\n",
        )

    def test_a_nested_mention_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a party "
                "requires that a party requires that the transaction is "
                "governed by IsdaAgreement.",
                _lexicon(REQUIRES_INI),
            )
        self.assertEqual(ctx.exception.gap, GAP_META_SHAPE)


class TestMetaDeterminism(unittest.TestCase):
    """(n) the meta document is byte-identical under any PYTHONHASHSEED."""

    SOURCE = os.path.join(ROOT, "examples", "paper_example2_meta.le")

    def test_same_input_twice_is_byte_identical(self):
        with open(self.SOURCE, encoding="utf-8") as handle:
            text = handle.read()
        self.assertEqual(
            compile_text(text, LEGAL_INI), compile_text(text, LEGAL_INI)
        )

    def test_output_is_independent_of_the_hash_seed(self):
        script = (
            "import sys; sys.path.insert(0, %r);"
            "from le import compile_text;"
            "import json;"
            "print(json.dumps([compile_text(open(%r).read(), %r).rules]))"
        ) % (ROOT, self.SOURCE, LEGAL_INI)
        outputs = []
        for seed in ("0", "7", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append(proc.stdout)
        self.assertEqual(len(set(outputs)), 1, outputs)
        self.assertIn("states(V2, governs, V1, isdaagreement)", outputs[0])


class TestMetaLexiconSurface(unittest.TestCase):
    """The declared [meta] surface is reachable through the loaded lexicon."""

    def test_the_example_lexicon_declares_one_meta_verb_phrase(self):
        lexicon = Lexicon.load(LEGAL_INI)
        self.assertEqual([tpl.text for tpl in lexicon.meta], [
            "<a confirmation> of <a transaction> states that <atom>",
        ])
        self.assertEqual(lexicon.meta_preds, frozenset({"states"}))
        # The meta verb phrase's words are reserved as template literals too.
        self.assertIn("states", lexicon.literals)
        self.assertIn("that", lexicon.literals)

    def test_a_nested_meta_condition_is_loud(self):
        # A mention is an OBJECT-level atom; a mention of a mention has no
        # reification in v1, so it is refused by name.
        with self.assertRaises(CompileError) as ctx:
            compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that the confirmation of the "
                "transaction states that the transaction is governed by "
                "IsdaAgreement.",
                LEGAL_INI,
            )
        self.assertEqual(ctx.exception.gap, GAP_META_SHAPE)


if __name__ == "__main__":
    unittest.main()
