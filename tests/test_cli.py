"""T9 -- CLI smoke test.

MEASURES that ``python -m le FILE`` prints the rule stream, that ``--facts``
adds the fact stream, that ``--json`` keeps the two streams distinct and its
two keys stable, that ``--check`` accepts the emitted rules, that
``--reactive`` prints the driver manifest, and that a compile error exits
non-zero with the gap in the message.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from fixtures import EXAMPLES, ROOT

from le.cli import validate_dl

PAPER2 = os.path.join(EXAMPLES, "paper_example2.le")
PAPER2_META = os.path.join(EXAMPLES, "paper_example2_meta.le")
FACTS = os.path.join(EXAMPLES, "facts.le")
RPS = os.path.join(EXAMPLES, "rps.le")

# A document whose two mentions reify into one VARIADIC 'states' relation with
# two different column counts (4 and 3), beside the lexicon that declares them.
TWO_ARITY_LE = (
    "A transaction is governed by IsdaAgreement if a confirmation of the "
    "transaction states that the transaction is governed by IsdaAgreement.\n"
    "A player is eligible if the player states that RpsGame is over.\n"
)
TWO_ARITY_INI = """[sorts]
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


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "le", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


class TestCli(unittest.TestCase):
    def test_default_prints_the_rule_stream(self):
        proc = run(PAPER2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "governs(V1, isdaagreement) :- commences(V1, V2), "
            "dated(isdaagreement, V3), V2 >= V3.\n",
        )

    def test_facts_flag_adds_the_fact_stream(self):
        proc = run(FACTS, "--facts")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "commences(acmetransaction, wednesday).\n"
            "dated(isdaagreement, monday).\n",
        )

    def test_json_flag_keeps_the_streams_apart(self):
        proc = run(FACTS, "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["rules"], "")
        self.assertEqual(
            payload["facts"],
            ["commences(acmetransaction, wednesday).",
             "dated(isdaagreement, monday)."],
        )

    def test_check_accepts_the_emitted_rules(self):
        proc = run(PAPER2, "--check")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("check: OK", proc.stderr)

    def test_lexicon_defaults_to_a_sibling_ini(self):
        proc = run(PAPER2)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_compile_error_exits_non_zero_with_the_gap(self):
        bad = os.path.join(EXAMPLES, "tmp_err.le")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write(
                "If a player P1 plays a choice C1 then P1 receives a prize.\n"
            )
        try:
            proc = run(bad)
        finally:
            os.unlink(bad)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("gap:", proc.stderr)
        self.assertIn("[actions]", proc.stderr)

    def test_missing_lexicon_exits_non_zero(self):
        proc = run(PAPER2, "--lexicon", os.path.join(EXAMPLES, "nope.ini"))
        self.assertEqual(proc.returncode, 1)


class TestReactiveCli(unittest.TestCase):
    def test_reactive_prints_the_driver_manifest(self):
        proc = run(RPS, "--reactive")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(
            set(payload), {"rules", "facts", "reactive", "actions"}
        )
        self.assertEqual(
            payload["rules"],
            "le_antecedent_1(P1, C1, P2, C2) :- plays(P1, C1), "
            "plays(P2, C2), beats(C1, C2), !over(rpsgame), P2 != P1.\n",
        )
        self.assertEqual(payload["actions"], ["plays", "receives"])
        rule = payload["reactive"][0]
        self.assertEqual(rule["index"], 1)
        self.assertEqual(rule["antecedent_pred"], "le_antecedent_1")
        self.assertEqual(rule["head_vars"], ["P1", "C1", "P2", "C2"])
        self.assertEqual(rule["event_preds"], ["plays"])
        self.assertEqual(
            rule["consequents"],
            [
                {"kind": "action", "pred": "receives",
                 "args": ["P1", "rpsprize"]},
                {"kind": "add", "pred": "over", "args": ["rpsgame"]},
            ],
        )

    def test_json_without_reactive_keeps_its_two_keys(self):
        proc = run(RPS, "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(set(payload), {"rules", "facts"})

    def test_check_accepts_a_reactive_document(self):
        proc = run(RPS, "--check")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("check: OK", proc.stderr)


class TestMetaCli(unittest.TestCase):
    """(m) the meta document through the CLI, including the --check exemption."""

    def test_default_prints_the_reified_rule(self):
        proc = run(PAPER2_META)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "governs(V1, isdaagreement) :- states(V2, governs, V1, "
            "isdaagreement), commences(V1, V3), dated(isdaagreement, V4), "
            "V3 >= V4.\n",
        )

    def test_check_accepts_the_meta_document(self):
        proc = run(PAPER2_META, "--check")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("check: OK", proc.stderr)

    def test_json_keeps_its_two_keys_for_the_meta_document(self):
        proc = run(PAPER2_META, "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(set(payload), {"rules", "facts"})
        self.assertEqual(payload["facts"], [])

    def test_check_exempts_the_variadic_meta_relation(self):
        # Two mentions of different arity in one 'states' relation.  Without the
        # exemption the one-arity-per-predicate rule flags it (the engine's
        # variadic relations are the exception); the CLI passes the lexicon's
        # meta predicates, and every other predicate keeps the strict rule.
        # The two documents live in a temporary directory, not in examples/.
        with tempfile.TemporaryDirectory(prefix="le-cli-") as tmp:
            doc = os.path.join(tmp, "two_arity.le")
            ini = os.path.join(tmp, "two_arity.ini")
            with open(doc, "w", encoding="utf-8") as handle:
                handle.write(TWO_ARITY_LE)
            with open(ini, "w", encoding="utf-8") as handle:
                handle.write(TWO_ARITY_INI)
            proc = run(doc, "--lexicon", ini, "--check")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("check: OK (2 rule(s), 0 fact(s))", proc.stderr)
            rules = subprocess.run(
                [sys.executable, "-m", "le", doc, "--lexicon", ini],
                cwd=ROOT, capture_output=True, text=True,
            ).stdout
        strict = validate_dl(rules)
        self.assertTrue(any("arity" in p for p in strict), strict)
        self.assertEqual(validate_dl(rules, frozenset({"states"})), ())


class TestRenderCli(unittest.TestCase):
    """(slice 4) --render: the compiled streams read back as LE sentences."""

    def test_render_prints_both_streams_as_sentences(self):
        # facts.le has no rules, so the output is its two fact sentences.
        proc = run(FACTS, "--render")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "AcmeTransaction commences on Wednesday.\n"
            "IsdaAgreement is dated as of Monday.\n",
        )

    def test_render_prints_the_rule_then_the_facts(self):
        with tempfile.TemporaryDirectory(prefix="le-cli-") as tmp:
            doc = os.path.join(tmp, "both.le")
            ini = os.path.join(tmp, "both.ini")
            with open(doc, "w", encoding="utf-8") as handle:
                handle.write(
                    "AcmeTransaction commences on Wednesday.\n"
                    "A transaction is governed by IsdaAgreement if the "
                    "transaction is governed by IsdaAgreement.\n"
                )
            shutil.copy(EXAMPLES + "/lexicon.ini", ini)
            proc = run(doc, "--lexicon", ini)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(
                proc.stdout,
                "governs(V1, isdaagreement) :- governs(V1, isdaagreement).\n",
            )
            proc = run(doc, "--lexicon", ini, "--render")
            self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "A transaction is governed by IsdaAgreement if the transaction "
            "is governed by IsdaAgreement.\n"
            "AcmeTransaction commences on Wednesday.\n",
        )

    def test_render_of_a_reactive_document_is_its_facts_only(self):
        # The le_antecedent_* artifact has no LE sentence behind it (its
        # source sentence is carried by the --reactive schema), so a
        # reactive document renders its facts only -- not a crash, not a
        # rendered artifact rule.
        proc = run(RPS, "--render")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "Rock beats Scissors.\n"
            "Scissors beats Paper.\n"
            "Paper beats Rock.\n",
        )

    def test_render_does_not_change_the_dl_streams(self):
        # --render REPLACES the dl output with the LE reading; it does not
        # append.  (The reverse direction has no dl of its own.)
        default = run(FACTS, "--facts").stdout
        proc = run(FACTS, "--render")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotEqual(proc.stdout, default)
        self.assertEqual(
            proc.stdout,
            "AcmeTransaction commences on Wednesday.\n"
            "IsdaAgreement is dated as of Monday.\n",
        )

    def test_render_refusal_exits_non_zero_with_the_gap(self):
        # day_chain's 'before' has two declared templates: no escape on the
        # CLI, so the ambiguity is loud (the ini [render] section is the
        # document's own escape).
        proc = run(os.path.join(EXAMPLES, "day_chain.le"), "--render")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("gap:", proc.stderr)
        self.assertIn("reverse render", proc.stderr)


class TestValidateDl(unittest.TestCase):
    def test_accepts_a_well_formed_rule(self):
        self.assertEqual(
            validate_dl("p(X, Y) :- q(X, Y), X >= Y.\n"), ()
        )

    def test_rejects_a_rule_without_a_positive_atom(self):
        self.assertTrue(validate_dl("p(X) :- !q(X).\n"))

    def test_rejects_an_arity_mismatch(self):
        problems = validate_dl("p(X) :- q(X), q(X, Y).\n")
        self.assertTrue(any("arity" in p for p in problems), problems)

    def test_rejects_unbalanced_parentheses(self):
        self.assertTrue(validate_dl("p(X :- q(X).\n"))

    def test_rejects_a_missing_full_stop(self):
        self.assertTrue(validate_dl("p(X) :- q(X)\n"))


if __name__ == "__main__":
    unittest.main()
