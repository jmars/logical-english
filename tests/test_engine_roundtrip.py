"""T7 -- engine round-trip through dlb (SKIPS when the engine is absent).

MEASURES that the compiled output is not merely well-formed text: the rules
load into datalog-dafsa via ``load_rules`` / ``query_rules_ro``, the facts
stream round-trips through the ``add_fact`` path (interned symbols), and a
compiled rule derives the row the paper's example implies.

dlb is imported ONLY here (by design, the one gate for the engine-backed
battery); the whole module skips -- it does
not fail -- when the ``dlb`` binding (or its ``libdatalog.so``) is not
importable, by plain import or via ``$DLB_PATH``.
"""

import re
import shutil
import sys
import tempfile
import unittest

from fixtures import DLB_IMPORT_ERROR, LEGAL_INI, ROOT, dlb

sys.path.insert(0, ROOT)
_DLB_IMPORT_ERROR = DLB_IMPORT_ERROR  # as documented in le.driver's docstring

from le import Lexicon, compile_text

PAPER_2 = (
    "A transaction is governed by IsdaAgreement if the transaction commences "
    "on a first day and IsdaAgreement is dated as of a second day and the "
    "first day is on or after the second day."
)
_FACT = re.compile(r"([a-z][A-Za-z0-9_]*)\((.*)\)\.\Z")


def _parse_fact(text):
    match = _FACT.match(text)
    if not match:
        raise AssertionError(f"not a dl fact: {text!r}")
    pred = match.group(1)
    args = [a.strip() for a in match.group(2).split(",") if a.strip()]
    return pred, args


def _col(db, token):
    """A dl fact-argument token -> the u32 column value the engine stores."""
    if token.isdigit():
        return int(token)
    if token.startswith('"') and token.endswith('"'):
        return db.intern(token[1:-1])
    return db.intern(token)


class _EngineCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="le-roundtrip-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def open_db(self):
        if dlb is None:
            self.skipTest(f"dlb not importable: {_DLB_IMPORT_ERROR}")
        try:
            return dlb.Db.open(self.dir)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"datalog engine unavailable: {exc}")


class TestEngineRoundtrip(_EngineCase):
    def test_paper_example2_rule_derives_and_filters(self):
        db = self.open_db()
        try:
            db.declare_relation("commences", 2)
            db.declare_relation("dated", 2)
            tx = db.intern("acmetx")
            isa = db.intern("isdaagreement")
            db.add_fact("commences", (tx, 100))
            db.add_fact("dated", (isa, 50))
            rules = compile_text(PAPER_2, LEGAL_INI).rules
            rows = sorted(tuple(r) for r in db.query_rules_ro(rules, "governs"))
            self.assertEqual(rows, [(tx, isa)])
        finally:
            db.close()

    def test_paper_example2_rule_filters_a_late_commencement(self):
        db = self.open_db()
        try:
            db.declare_relation("commences", 2)
            db.declare_relation("dated", 2)
            tx = db.intern("acmetx")
            isa = db.intern("isdaagreement")
            db.add_fact("commences", (tx, 100))
            db.add_fact("dated", (isa, 150))  # dated AFTER the commencement
            rules = compile_text(PAPER_2, LEGAL_INI).rules
            rows = list(db.query_rules_ro(rules, "governs"))
            self.assertEqual(rows, [])
        finally:
            db.close()

    def test_facts_stream_round_trips_through_add_fact(self):
        db = self.open_db()
        try:
            result = compile_text(
                "AcmeTransaction commences on Wednesday.", LEGAL_INI
            )
            pred, args = _parse_fact(result.facts[0])
            db.declare_relation(pred, len(args))
            cols = tuple(_col(db, a) for a in args)
            db.add_fact(pred, cols)
            self.assertTrue(db.lookup(pred, cols))
        finally:
            db.close()

    def test_compiled_bare_symbol_matches_an_interned_symbol(self):
        db = self.open_db()
        try:
            rules = compile_text(
                "A transaction is governed by IsdaAgreement if the transaction "
                "is governed by IsdaAgreement.",
                LEGAL_INI,
            ).rules
            facts = compile_text(
                "AcmeTransaction is governed by IsdaAgreement.", LEGAL_INI
            ).facts
            db.declare_relation("governs", 2)
            pred, args = _parse_fact(facts[0])
            cols = tuple(_col(db, a) for a in args)
            db.add_fact(pred, cols)
            rows = sorted(tuple(r) for r in db.query_rules_ro(rules, "governs"))
            self.assertEqual(rows, [cols])
        finally:
            db.close()

    def test_arithmetic_grounded_head_variable_is_accepted(self):
        db = self.open_db()
        try:
            db.declare_relation("before", 2)
            db.add_fact("before", (1, 4))
            db.add_fact("before", (5, 9))
            rules = compile_text(
                "A day is a delivery day if a first day is before a second day "
                "and the day is 3 days before the second day.",
                LEGAL_INI,
            ).rules
            rows = sorted(tuple(r) for r in db.query_rules_ro(rules, "delivery_day"))
            self.assertEqual(rows, [(1,), (6,)])
        finally:
            db.close()

    def test_reordered_multi_arith_chain_loads_and_derives(self):
        db = self.open_db()
        try:
            db.declare_relation("before", 2)
            wed = db.intern("wednesday")
            db.add_fact("before", (10, wed))
            # The producers are written in the WRONG order in the source; the
            # compiler emits 'V2 = V3 - 4' before 'V1 = V2 - 3' so that each
            # operand is bound by a preceding atom.  The engine then derives
            # (10 - 4) - 3 = 3.
            rules = compile_text(
                "A day is a delivery day if the day is 3 days before a second "
                "day and the second day is 4 days before a third day and the "
                "third day is before Wednesday.",
                LEGAL_INI,
            ).rules
            self.assertEqual(
                rules,
                "delivery_day(V1) :- before(V3, wednesday), V2 = V3 - 4, "
                "V1 = V2 - 3.\n",
            )
            rows = sorted(
                tuple(r) for r in db.query_rules_ro(rules, "delivery_day")
            )
            self.assertEqual(rows, [(3,)])
        finally:
            db.close()

    def test_negation_rule_loads_and_runs(self):
        db = self.open_db()
        try:
            db.declare_relation("plays", 2)
            db.declare_relation("excluded", 1)
            rules = compile_text(
                "A player is eligible if it is not the case that the player is "
                "excluded and the player plays a choice.",
                LEGAL_INI,
            ).rules
            p1, p2, c1 = db.intern("p1"), db.intern("p2"), db.intern("c1")
            db.add_fact("plays", (p1, c1))
            db.add_fact("plays", (p2, c1))
            db.add_fact("excluded", (p2,))
            rows = sorted(tuple(r) for r in db.query_rules_ro(rules, "eligible"))
            self.assertEqual(rows, [(p1,)])
        finally:
            db.close()


META_TWO_ARITY_INI = """
[sorts]
transaction = transaction
confirmation = confirmation
agreement = agreement
player = player
game = game

[names]
IsdaAgreement = agreement
RpsGame = game

[predicates]
<a transaction> is governed by <an agreement> | governs | transaction, agreement
<a game> is over | over | game
<a player> is eligible | eligible | player

[meta]
<a confirmation> of <a transaction> states that <atom> | states | confirmation, transaction, _
<a player> states that <atom> | states | player, _
"""

META_PAPER_2 = (
    "A transaction is governed by IsdaAgreement if a confirmation of the "
    "transaction states that the transaction is governed by IsdaAgreement and "
    "the transaction commences on a first day and IsdaAgreement is dated as of "
    "a second day and the first day is on or after the second day."
)


class TestMetaEngineRoundtrip(_EngineCase):
    """The reified mention is not merely well-formed text (T12's other half).

    MEASURES (k) that the compiled clause DERIVES from a mention POSTED through
    the add_fact path -- a symbol constant in the second column joining the
    head's USE of the same predicate -- with the three controls that must derive
    nothing, and (l) that ONE variadic ``states`` relation holds mentions of two
    different arities, which is why the relation is declared variadic at all.
    """

    def _open(self):
        db = self.open_db()
        db.declare_relation_variadic("states")
        for name in ("commences", "dated"):
            db.declare_relation(name, 2)
        return db

    def test_paper_example2_derives_from_a_posted_mention(self):
        db = self._open()
        try:
            rules = compile_text(META_PAPER_2, LEGAL_INI).rules
            conf, t1 = db.intern("conf1"), db.intern("t1")
            isa = db.intern("isdaagreement")
            db.add_fact("states", (conf, db.intern("governs"), t1, isa))
            db.add_fact("commences", (t1, 5))
            db.add_fact("dated", (isa, 3))
            rows = sorted(tuple(r) for r in db.query_rules_ro(rules, "governs"))
            self.assertEqual(rows, [(t1, isa)])
        finally:
            db.close()

    def test_a_wrong_predicate_symbol_in_the_mention_derives_nothing(self):
        # Control 1: the ENGINE (not the compiler) is what makes the symbol
        # column meaningful -- a mention of another predicate is a different
        # row, so the object-level USE of 'governs' stays unsatisfied.
        db = self._open()
        try:
            rules = compile_text(META_PAPER_2, LEGAL_INI).rules
            conf, t1 = db.intern("conf1"), db.intern("t1")
            isa = db.intern("isdaagreement")
            db.add_fact("states", (conf, db.intern("commences"), t1, isa))
            db.add_fact("commences", (t1, 5))
            db.add_fact("dated", (isa, 3))
            self.assertEqual(list(db.query_rules_ro(rules, "governs")), [])
        finally:
            db.close()

    def test_no_mention_at_all_derives_nothing(self):
        # Control 2: the head needs the mention, not just the dates.
        db = self._open()
        try:
            rules = compile_text(META_PAPER_2, LEGAL_INI).rules
            t1, isa = db.intern("t1"), db.intern("isdaagreement")
            db.add_fact("commences", (t1, 5))
            db.add_fact("dated", (isa, 3))
            self.assertEqual(list(db.query_rules_ro(rules, "governs")), [])
        finally:
            db.close()

    def test_a_late_commencement_derives_nothing(self):
        # Control 3: the comparison still filters (3 >= 5 is false).
        db = self._open()
        try:
            rules = compile_text(META_PAPER_2, LEGAL_INI).rules
            conf, t1 = db.intern("conf1"), db.intern("t1")
            isa = db.intern("isdaagreement")
            db.add_fact("states", (conf, db.intern("governs"), t1, isa))
            db.add_fact("commences", (t1, 3))
            db.add_fact("dated", (isa, 5))
            self.assertEqual(list(db.query_rules_ro(rules, "governs")), [])
        finally:
            db.close()

    def test_one_variadic_relation_holds_two_mention_arities(self):
        # (l) 'states' carries a 4-column mention (governs/2) and a 3-column one
        # (over/1) at the same time; both rules derive.  A fixed arity could not
        # hold both, and a per-arity pair of relations would fragment them.
        db = self._open()
        try:
            rules = compile_text(
                "A transaction is governed by IsdaAgreement if a confirmation "
                "of the transaction states that the transaction is governed by "
                "IsdaAgreement.\n"
                "A player is eligible if the player states that RpsGame is "
                "over.",
                Lexicon.from_text(META_TWO_ARITY_INI),
            ).rules
            self.assertEqual(
                rules,
                "governs(V1, isdaagreement) :- "
                "states(V2, governs, V1, isdaagreement).\n"
                "eligible(V3) :- states(V3, over, rpsgame).\n",
            )
            conf, t1, p1 = db.intern("conf1"), db.intern("t1"), db.intern("p1")
            isa = db.intern("isdaagreement")
            db.add_fact("states", (conf, db.intern("governs"), t1, isa))
            db.add_fact("states", (p1, db.intern("over"), db.intern("rpsgame")))
            self.assertEqual(
                sorted(tuple(r) for r in db.query_rules_ro(rules, "governs")),
                [(t1, isa)],
            )
            self.assertEqual(
                sorted(tuple(r) for r in db.query_rules_ro(rules, "eligible")),
                [(p1,)],
            )
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
