"""The REVERSE direction: dl -> Logical English.

Three pins, and what each CANNOT catch:

* dl -> LE -> dl IDENTITY -- the rendered sentence is fed through the REAL
  ``compile_text`` and the result must be BYTE-EQUAL to the input dl.  This
  is the ORACLE for the inverse-binding algorithm: an 'a'-where-'the'
  article error makes the compiler fire GAP_UNBOUND_THE / GAP_REPEAT_A, so a
  wrong binding order is LOUD rather than silently wrong
  (test_identity_surfaces_article_errors_loudly proves the asymmetry).
  CANNOT catch: a renderer+compiler pair consistently wrong together.
* GOLDEN STRING PINS -- the exact rendered text for the example corpus, so
  drift is caught by the string even where the semantic pin would pass.
* LE -> dl -> LE STABILITY -- the render of (compile of render) equals the
  render; the renderer does not drift on its own output.

Known non-identity (MEASURED, plan open question (1)): 'another' /
'the other' are information-lost in dl -- both lower to a fresh variable --
so the CANONICAL inverse is the ordinal form ('a first day'); the identity
test asserts equality to that canonical form, and
test_another_renders_as_the_canonical_ordinal_form pins it explicitly.
"""

import shutil
import tempfile
import unittest

from fixtures import DAY, EXAMPLES, LEGAL, RPS

from le import (
    CompileError,
    GAP_CLOSED_LEXICON,
    Lexicon,
    compile_text,
)
from le.render import (
    GAP_RENDER_AMBIGUOUS_TEMPLATE,
    GAP_RENDER_ARITH,
    GAP_RENDER_META_SHAPE,
    GAP_RENDER_NAME_COLLISION,
    GAP_RENDER_NO_TEMPLATE,
    GAP_RENDER_NOT_GROUND,
    GAP_RENDER_ORDER,
    GAP_RENDER_OVERRIDE,
    GAP_RENDER_PHRASE_COLLISION,
    GAP_RENDER_SHAPE,
    GAP_RENDER_TYPE_CLASH,
    GAP_RENDER_UNDECLARED_SYMBOL,
    GAP_RENDER_UNSORTED_SLOT,
    render_facts,
    render_result,
    render_rule,
)

# The four example documents' compiled outputs (MEASURED; also pinned by
# tests/test_paper_examples.py and tests/golden/render_baseline.txt).
PAPER2_DL = (
    "governs(V1, isdaagreement) :- commences(V1, V2), "
    "dated(isdaagreement, V3), V2 >= V3."
)
PAPER2_LE = (
    "A transaction is governed by IsdaAgreement if the transaction commences "
    "on a day and IsdaAgreement is dated as of a first day and the day is on "
    "or after the first day."
)
PAPER2_META_DL = (
    "governs(V1, isdaagreement) :- states(V2, governs, V1, isdaagreement), "
    "commences(V1, V3), dated(isdaagreement, V4), V3 >= V4."
)
PAPER2_META_LE = (
    "A transaction is governed by IsdaAgreement if a confirmation of the "
    "transaction states that the transaction is governed by IsdaAgreement "
    "and the transaction commences on a day and IsdaAgreement is dated as of "
    "a first day and the day is on or after the first day."
)
DAY_CHAIN_DL = "before(V1, wednesday) :- before(V1, V2), before(V2, wednesday)."
# The canonical ordinal inverse: the source says 'another day', which dl
# cannot carry (it lowers to a fresh variable), so the render is the ordinal.
DAY_CHAIN_LE = (
    "A day is before Wednesday if the day is before a first day and the "
    "first day is before Wednesday."
)
FACTS_DL = (
    "commences(acmetransaction, wednesday).",
    "dated(isdaagreement, monday).",
)
FACTS_LE = (
    "AcmeTransaction commences on Wednesday.",
    "IsdaAgreement is dated as of Monday.",
)
RPS_FACTS_DL = (
    "beats(rock, scissors).",
    "beats(scissors, paper).",
    "beats(paper, rock).",
)
RPS_FACTS_LE = (
    "Rock beats Scissors.",
    "Scissors beats Paper.",
    "Paper beats Rock.",
)

# 'before' is the MEASURED ambiguity: two templates share the predicate.
BEFORE_PLAIN = next(
    t for t in LEGAL.all_templates
    if t.pred == "before" and t.text == "<dayA> is before <dayB>"
)
BEFORE_DAY = next(
    t for t in LEGAL.all_templates
    if t.pred == "before" and t.text == "<dayA> is the day before <dayB>"
)


def _identity(dl_line: str, lexicon, *, template=None) -> None:
    """Assert dl -> LE -> dl is byte-identical (the strong pin)."""
    sentence = render_rule(dl_line, lexicon, template=template)
    assert compile_text(sentence, lexicon).rules == dl_line + "\n", sentence


class TestRuleIdentity(unittest.TestCase):
    def test_paper_example2_identity(self):
        _identity(PAPER2_DL, LEGAL)

    def test_paper_example2_meta_identity(self):
        # Includes the meta states(...) body row and the >= comparison.
        _identity(PAPER2_META_DL, LEGAL)

    def test_day_chain_identity_via_the_ordinal_canonical_form(self):
        # The source's 'another day' is information-lost in dl, so the
        # identity target is the ORDINAL render, not the source wording.
        _identity(DAY_CHAIN_DL, LEGAL, template=BEFORE_PLAIN)
        _identity(DAY_CHAIN_DL, LEGAL, template=BEFORE_DAY)

    def test_every_comparison_operator(self):
        # DAY_COPULA leaves the copulas builtin (nothing declares 'is
        # before'), so every operator's inverse phrase re-reads as the same
        # comparison.  DAY (which DECLARES '<dayA> is before <dayB>') is the
        # collision case: 'is before' re-reads as the template, so the
        # render is LOUD rather than silently re-meaning the template.
        from fixtures import DAY_COPULA

        tpl = next(t for t in DAY_COPULA.all_templates if t.pred == "precedes")
        for op in ("<=", ">=", "<", ">"):
            with self.subTest(op=op):
                _identity(
                    f"delivery_day(V1) :- precedes(V1, V2), V2 {op} V1.",
                    DAY_COPULA,
                    template=tpl,
                )
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "delivery_day(V1) :- before(V1, V2), V2 < V1.", DAY,
                template=next(t for t in DAY.all_templates if t.pred == "before"),
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_PHRASE_COLLISION)

    def test_negation_wraps_the_atom(self):
        _identity(
            "delivery_day(V1) :- before(V1, V2), !delivery_day(V2).", DAY,
            template=next(t for t in DAY.all_templates if t.pred == "before"),
        )

    def test_negated_meta_body_atom_round_trips(self):
        dl = (
            "governs(V1, isdaagreement) :- received(V2, V1), "
            "!states(V2, governs, V1, isdaagreement)."
        )
        sentence = render_rule(dl, LEGAL)
        self.assertIn(
            "it is not the case that the confirmation of the transaction "
            "states that",
            sentence,
        )
        self.assertEqual(compile_text(sentence, LEGAL).rules, dl + "\n")

    def test_explicit_names_round_trip(self):
        dl = "eligible(P1) :- receives(P1, V1)."
        sentence = render_rule(dl, LEGAL)
        self.assertEqual(
            sentence, "A player P1 is eligible if P1 receives a prize."
        )
        self.assertEqual(compile_text(sentence, LEGAL).rules, dl + "\n")

    def test_three_same_sort_variables_use_second_and_third(self):
        dl = "delivery_day(V1) :- before(V1, V2), before(V2, V3), before(V3, V4)."
        sentence = render_rule(
            dl, DAY, template=next(t for t in DAY.all_templates if t.pred == "before")
        )
        self.assertEqual(
            sentence,
            "A day is a delivery day if the day is before a first day and "
            "the first day is before a second day and the second day is "
            "before a third day.",
        )
        self.assertEqual(compile_text(sentence, DAY).rules, dl + "\n")

    def test_folded_meta_subject_round_trips(self):
        # The compiler DROPS a subject column whose variable also appears in
        # the mention; the renderer recovers it by sort-matching.
        dl = "commences(V1, V2) :- states(V3, commences, V1, V2)."
        sentence = render_rule(dl, LEGAL)
        self.assertEqual(
            sentence,
            "A transaction commences on a day if a confirmation of the "
            "transaction states that the transaction commences on the day.",
        )
        self.assertEqual(compile_text(sentence, LEGAL).rules, dl + "\n")

    def test_meta_head_round_trips(self):
        dl = "states(V1, governs, V2, isdaagreement) :- received(V1, V2)."
        sentence = render_rule(dl, LEGAL)
        self.assertEqual(
            sentence,
            "A confirmation of a transaction states that the transaction is "
            "governed by IsdaAgreement if the confirmation of the "
            "transaction is received.",
        )
        self.assertEqual(compile_text(sentence, LEGAL).rules, dl + "\n")

    def test_doc_level_numbering_does_not_restart(self):
        # The compiler's counter is DOCUMENT-wide, so a second rule's
        # variables need not start at V1; per-rule rendering still works and
        # the whole rendered document recompiles byte-identically.
        doc = (
            "A transaction is governed by IsdaAgreement if the transaction "
            "commences on a day.\n"
            "A confirmation of a transaction is accepted if the "
            "confirmation of the transaction is received.\n"
        )
        result = compile_text(doc, LEGAL)
        self.assertEqual(
            result.rules,
            "governs(V1, isdaagreement) :- commences(V1, V2).\n"
            "accepted(V3, V4) :- received(V3, V4).\n",
        )
        rendered = "\n".join(render_rule(l, LEGAL) for l in result.rule_lines)
        self.assertEqual(compile_text(rendered + "\n", LEGAL).rules, result.rules)

    def test_identity_surfaces_article_errors_loudly(self):
        # The ORACLE's asymmetry proof: an 'a'-where-'the' error is not a
        # wrong-but-accepted render -- the compiler itself refuses it.  A
        # swapped article in either direction is loud.
        with self.assertRaises(CompileError) as unbound:
            compile_text(
                "A transaction is governed by IsdaAgreement if the "
                "transaction commences on the day.",
                LEGAL,
            )
        self.assertIn("no antecedent", str(unbound.exception))
        with self.assertRaises(CompileError) as repeat:
            compile_text(
                "A transaction is governed by IsdaAgreement if the "
                "transaction commences on a day and a transaction is "
                "governed by IsdaAgreement.",
                LEGAL,
            )
        self.assertIn("second unanchored binding", str(repeat.exception))


class TestGoldenStrings(unittest.TestCase):
    """Pin the exact rendered text (a pair-wise-wrong renderer+compiler
    passes the identity pin; the string pin catches the drift)."""

    def test_paper_example2_golden(self):
        self.assertEqual(render_rule(PAPER2_DL, LEGAL), PAPER2_LE)

    def test_paper_example2_meta_golden(self):
        self.assertEqual(render_rule(PAPER2_META_DL, LEGAL), PAPER2_META_LE)

    def test_day_chain_golden_is_the_ordinal_form(self):
        self.assertEqual(
            render_rule(DAY_CHAIN_DL, LEGAL, template=BEFORE_PLAIN),
            DAY_CHAIN_LE,
        )

    def test_facts_golden(self):
        self.assertEqual(render_facts(FACTS_DL, LEGAL), FACTS_LE)

    def test_rps_facts_golden(self):
        self.assertEqual(render_facts(RPS_FACTS_DL, RPS), RPS_FACTS_LE)

    def test_the_document_sources_round_trip_byte_equal(self):
        # The four example documents: compiled -> rendered -> recompiled is
        # byte-identical to the compiled output.
        docs = (
            ("paper_example2", LEGAL, None),
            ("paper_example2_meta", LEGAL, None),
            ("day_chain", LEGAL, BEFORE_PLAIN),
        )
        for name, lexicon, template in docs:
            with self.subTest(doc=name):
                source = open(
                    f"{EXAMPLES}/{name}.le", encoding="utf-8"
                ).read()
                compiled = compile_text(source, lexicon)
                rendered = "\n".join(
                    render_rule(l, lexicon, template=template)
                    for l in compiled.rule_lines
                )
                self.assertEqual(
                    compile_text(rendered + "\n", lexicon).rules, compiled.rules
                )
        facts = compile_text(
            open(f"{EXAMPLES}/facts.le", encoding="utf-8").read(), LEGAL
        )
        self.assertEqual(render_facts(facts.facts, LEGAL), FACTS_LE)
        rps_facts = compile_text(
            open(f"{EXAMPLES}/rps.le", encoding="utf-8").read(), RPS
        )
        self.assertEqual(render_facts(rps_facts.facts, RPS), RPS_FACTS_LE)

    def test_another_renders_as_the_canonical_ordinal_form(self):
        # MEASURED non-identity: 'another day' lowers to a fresh variable,
        # indistinguishable from 'a first day', so the canonical inverse is
        # the ordinal form.  (The user who wants the source wording supplies
        # it through the template= choice.)
        self.assertIn("a first day", render_rule(
            DAY_CHAIN_DL, LEGAL, template=BEFORE_PLAIN
        ))


class TestStability(unittest.TestCase):
    """render(compiled(x)) == render(compiled(render(compiled(x)))): the
    renderer is idempotent under one more compile-render cycle.  CANNOT
    catch a pair-wise-wrong-but-stable renderer/compiler (same class as the
    identity pin); the golden pins carry that risk."""

    def test_rules_are_stable(self):
        cases = (
            (PAPER2_DL, LEGAL, None),
            (PAPER2_META_DL, LEGAL, None),
            (DAY_CHAIN_DL, LEGAL, BEFORE_PLAIN),
        )
        for dl, lexicon, template in cases:
            with self.subTest(dl=dl[:24]):
                once = render_rule(dl, lexicon, template=template)
                twice = render_rule(
                    compile_text(once, lexicon).rule_lines[0], lexicon,
                    template=template,
                )
                self.assertEqual(once, twice)

    def test_facts_are_stable(self):
        for facts, lexicon in ((FACTS_DL, LEGAL), (RPS_FACTS_DL, RPS)):
            with self.subTest(lexicon=lexicon.name):
                once = render_facts(facts, lexicon)
                recompiled = compile_text(" ".join(once), lexicon).facts
                self.assertEqual(render_facts(recompiled, lexicon), once)


class TestRefusals(unittest.TestCase):
    """Every gap fires on an input CONSTRUCTED to trigger it, with the exact
    gap text asserted (a test that cannot fail is not a test)."""

    def test_ambiguous_predicate_is_loud(self):
        # MEASURED: examples/lexicon.ini declares TWO 'before' templates.
        with self.assertRaises(CompileError) as ctx:
            render_rule(DAY_CHAIN_DL, LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_AMBIGUOUS_TEMPLATE)

    def test_ambiguous_predicate_escape_via_template(self):
        # The caller picks the surface wording; both choices round-trip.
        for template in (BEFORE_PLAIN, BEFORE_DAY):
            with self.subTest(template=template.text):
                sentence = render_rule(
                    DAY_CHAIN_DL, LEGAL, template=template
                )
                self.assertEqual(
                    compile_text(sentence, LEGAL).rules,
                    DAY_CHAIN_DL + "\n",
                )

    def test_ambiguous_fact_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("before(monday, wednesday).",), LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_AMBIGUOUS_TEMPLATE)

    def test_no_template_le_antecedent(self):
        # The reactive artifact predicate: no template, never a skip.
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "le_antecedent_1(P1, C1, P2, C2) :- plays(P1, C1), "
                "plays(P2, C2), beats(C1, C2), !over(rpsgame), P2 != P1.",
                RPS,
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_NO_TEMPLATE)

    def test_no_template_witness_predicate(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("w_orphaned(rock, paper).",), RPS)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_NO_TEMPLATE)

    def test_no_template_states_row_is_a_meta_row_not_a_skip(self):
        # A states(...) row with no consistent reading fits no [meta]
        # template -- loud, never silently skipped.
        with self.assertRaises(CompileError) as ctx:
            render_facts(("states(a, b).",), LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_META_SHAPE)

    def test_undeclared_symbol_is_loud_never_an_invented_noun(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("commences(t57, wednesday).",), LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_UNDECLARED_SYMBOL)

    def test_undeclared_symbol_display_map_escape(self):
        # The CALLER may extend [names] explicitly; the renderer invents
        # nothing.  The surface must itself be compiler-readable (a proper
        # noun), or the escape is refused.
        got = render_facts(
            ("commences(t57, wednesday).",), LEGAL, names={"t57": "Task"}
        )
        self.assertEqual(got, ("Task commences on Wednesday.",))
        with self.assertRaises(CompileError):
            render_facts(
                ("commences(t57, wednesday).",), LEGAL,
                names={"t57": "Task57"},
            )

    def test_unsorted_slot_is_loud(self):
        # A variable whose only occurrence is a comparison operand has no
        # declared slot, so its noun phrase cannot be chosen (the forward
        # compiler refuses the same shape: an ungrounded comparison).
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\n[names]\nWednesday = day\n"
            "[predicates]\n<dayA> is before <dayB> | before | day, day\n"
            "<a day> is a delivery day | delivery_day | day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "delivery_day(V1) :- before(V1, V2), V2 >= V3.", lexicon
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_UNSORTED_SLOT)

    def test_integers_render_as_digits_render_only(self):
        # Render-only (MEASURED: the forward compiler refuses digits in a
        # slot), so the identity round-trip is explicitly NOT claimed here.
        got = render_facts(("commences(acmetransaction, 3).",), LEGAL)
        self.assertEqual(got, ("AcmeTransaction commences on 3.",))

    def test_variable_in_a_fact_line_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("commences(V1, wednesday).",), LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_NOT_GROUND)

    def test_arithmetic_producer_is_a_loud_gap(self):
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "delivery_day(V1) :- before(V2, V3), V1 = V3 - 3.",
                LEGAL, template=BEFORE_PLAIN,
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_ARITH)

    def test_body_out_of_compiler_order_is_loud(self):
        # Comparisons are emitted LAST by the compiler; a comparison before
        # a positive atom is not the compiler's emission and is refused.
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "governs(V1, isdaagreement) :- V2 >= V3, "
                "commences(V1, V2), dated(isdaagreement, V3).",
                LEGAL,
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_ORDER)

    def test_non_consecutive_numbering_renders_alpha_equivalently(self):
        # A renumbered rule (V3 mentioned before V2) is not an emission the
        # compiler could produce -- but rendering it is MEANING-SAFE: each
        # variable's rank by number gives it one (sort, ordinal) key, so the
        # sentence recompiles to the SAME rule under a consistent renaming
        # (the previous refusal here guarded a rename that was never a
        # semantic difference; the docstring's fear was unfounded).
        renumbered = (
            "governs(V1, isdaagreement) :- commences(V1, V3), "
            "dated(isdaagreement, V2), V3 >= V2."
        )
        sentence = render_rule(renumbered, LEGAL)
        canonical = (
            "governs(V1, isdaagreement) :- commences(V1, V2), "
            "dated(isdaagreement, V3), V2 >= V3.\n"
        )
        self.assertEqual(compile_text(sentence, LEGAL).rules, canonical)

    def test_name_collision_is_loud(self):
        # Two surfaces for one dl symbol: the renderer refuses to pick
        # (triggered through the names= display map's quoted/unquoted
        # double keying; the loader itself admits Ab/AB today).
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\n[names]\nWednesday = day\n"
            "[predicates]\n<a day> is here | here | day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            render_facts(
                ("here(acme).",), lexicon,
                names={"acme": "Foo", '"acme"': "Bar"},
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_NAME_COLLISION)

    def test_type_clash_is_loud(self):
        # A constant whose declared [names] sort disagrees with the slot.
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\nthing = thing\n"
            "[names]\nAcme = thing\nWednesday = day\n"
            "[predicates]\n<a day> relates to <a thing> | relates | day, thing\n"
        )
        with self.assertRaises(CompileError) as ctx:
            render_facts(("relates(acme, wednesday).",), lexicon)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_TYPE_CLASH)

    def test_unused_template_override_is_loud(self):
        # template= naming a predicate the input never uses.
        with self.assertRaises(CompileError) as ctx:
            render_rule(
                "accepted(V1, V2) :- received(V1, V2).", LEGAL,
                template=BEFORE_PLAIN,
            )
        self.assertEqual(ctx.exception.gap, GAP_RENDER_OVERRIDE)

    def test_head_only_variable_is_refused_like_the_compiler(self):
        # A variable only the conclusion mentions is NOT a compiler
        # emission (GAP_EXISTENTIAL_HEAD, le.lower._check_clause); the
        # renderer refuses it too, so a rendered rule never fails to
        # recompile.
        with self.assertRaises(CompileError) as ctx:
            render_rule("received(V1, V2) :- governs(V2, isdaagreement).", LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_ORDER)
        self.assertIn("GAP_EXISTENTIAL_HEAD", str(ctx.exception))

    def test_an_int_initial_sentence_is_refused_not_capped(self):
        # '3 Happens on Wednesday.' would be a capitalisation artefact, not
        # a rendering: an int subject slot is refused loudly (the forward
        # compiler refuses digits in a slot, so the round-trip cannot hold
        # anyway).
        lexicon = Lexicon.from_text(
            "[sorts]\nday = day\nthing = thing\n"
            "[names]\nWednesday = day\n"
            "[predicates]\n<thing> happens on <a day> | happens | thing, day\n"
        )
        with self.assertRaises(CompileError) as ctx:
            render_facts(("happens(3, wednesday).",), lexicon)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_SHAPE)

    def test_integer_render_only_contract_pins_recompile_too(self):
        # The render-only contract is about the ROUND-TRIP, not the string:
        # recompiling the rendered sentence must fire the compiler's own
        # closed-lexicon refusal.
        sentence = render_facts(("commences(acmetransaction, 3).",), LEGAL)[0]
        self.assertEqual(sentence, "AcmeTransaction commences on 3.")
        with self.assertRaises(CompileError) as ctx:
            compile_text(sentence + "\n", LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_CLOSED_LEXICON)

    def test_negation_introduced_variable_renders_alpha_equivalently(self):
        # The reviewer's finding-1 input (a legal emission the old check
        # refused): a negated condition binds V2, and the kind-sorted text
        # mentions the higher-numbered V3 first.  The render is not
        # byte-identical-on-recompile (LE cannot mention a variable before
        # introducing it), but it recompiles to the SAME rule: every
        # variable keeps its (sort, ordinal) key and the renaming is
        # consistent -- the definition of alpha-equivalence.
        src = (
            "A day is a delivery day if it is not the case that the day is "
            "before a first day and the day is before a second day and the "
            "first day is before the second day.\n"
        )
        compiled = compile_text(src, LEGAL)
        sentence = render_rule(compiled.rules.strip(), LEGAL, template=BEFORE_PLAIN)
        # The ordinals come from the variable NUMBERS (the binder's own
        # keys), not the dl text order: V3 is met before V2 but holds the
        # FIRST ordinal key, so the pin below is the number-derived reading
        # and a walk-order reading renders a DIFFERENT sentence.
        self.assertEqual(
            sentence,
            "A day is a delivery day if the day is before a second day and "
            "a first day is before the second day and it is not the case "
            "that the day is before the first day.",
        )
        self.assertEqual(
            compile_text(sentence + "\n", LEGAL).rules,
            "delivery_day(V1) :- before(V1, V2), before(V3, V2), "
            "!before(V1, V3).\n",
        )

    def test_ground_derived_states_row_renders(self):
        # The fact leg: the meta-head rule states(V1, governs, V2,
        # isdaagreement) :- received(V1, V2). derives the fully ground row
        # below, which the forward compiler's own fold NEVER emits (a
        # constant subject is always a column).  The folded subject slot is
        # recovered from the mention VALUES, one sort-consistent reading
        # required.
        got = render_facts(
            ("states(acmeconfirmation, governs, acmetransaction, "
             "isdaagreement).",),
            LEGAL,
        )
        self.assertEqual(
            got,
            ("AcmeConfirmation of AcmeTransaction states that "
             "AcmeTransaction is governed by IsdaAgreement.",),
        )

    def test_ground_meta_row_ambiguity_is_loud(self):
        # Two same-sort mention values both fit the folded slot: loud, never
        # a silent pick (a constructed lexicon -- the LEGAL meta template's
        # mention sorts admit only one candidate value).
        lexicon = Lexicon.from_text(
            "[sorts]\nperson = person\n"
            "[names]\nAlice = person\nBob = person\nCarol = person\n"
            "[predicates]\n<a person> meets <a person> | meets | person, person\n"
            "[meta]\n<a person> of <a person> tells that <atom> | told | "
            "person, person, _\n"
        )
        with self.assertRaises(CompileError) as ctx:
            render_facts(("told(alice, meets, bob, carol).",), lexicon)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_AMBIGUOUS_TEMPLATE)

    def test_facts_path_takes_the_template_escape(self):
        # The render_facts/render_result asymmetry with render_rule is
        # closed: an ambiguous pred no longer fails the whole fact stream.
        got = render_facts(
            ("before(monday, wednesday).",), LEGAL, template=BEFORE_DAY
        )
        self.assertEqual(got, ("Monday is the day before Wednesday.",))
        # The override is checked for USE, as render_rule checks it: naming
        # a predicate the stream never mentions is loud, not silent.
        with self.assertRaises(CompileError) as ctx:
            render_facts(FACTS_DL, LEGAL, template=BEFORE_DAY)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_OVERRIDE)

    def test_render_result_takes_the_template_escape(self):
        result = compile_text(
            "Monday is before Wednesday.\n", LEGAL
        )
        got = render_result(result, LEGAL, template=BEFORE_PLAIN)
        self.assertEqual(got, ("Monday is before Wednesday.",))

    def test_non_dl_shape_is_loud(self):
        for bad in ("governs(V1) if commences(V1, V2).", "governs(V1)."):
            with self.subTest(bad=bad):
                with self.assertRaises(CompileError) as ctx:
                    render_rule(bad, LEGAL)
                self.assertEqual(ctx.exception.gap, GAP_RENDER_SHAPE)

    def test_wrong_arity_is_loud(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("commences(acmetransaction).",), LEGAL)
        self.assertIn("reverse render", ctx.exception.gap)


class TestConvenience(unittest.TestCase):
    def test_render_result_renders_the_fact_stream(self):
        result = compile_text(
            open(f"{EXAMPLES}/facts.le", encoding="utf-8").read(), LEGAL
        )
        self.assertEqual(render_result(result, LEGAL), FACTS_LE)


# -- slice 3: the live-db path ------------------------------------------------

try:  # pragma: no cover - environment dependent
    from fixtures import dlb as _dlb
except Exception:  # pragma: no cover - environment dependent
    _dlb = None

# NOTE: this import happens AFTER the fixtures bootstrap above (which honours
# $DLB_PATH), so on a
# dlb-less host the try block has already fallen through and render_db's
# own lazy import is never reached by these tests (they skip instead).
from le.render import render_db  # noqa: E402  (after the sys.path bootstrap)


@unittest.skipIf(_dlb is None, "dlb not importable")
class TestRenderDb(unittest.TestCase):
    """render_db against the REAL engine (skips when dlb is absent).

    MEASURED on this host: the engine's columns are u32 -- an interned
    symbol id (1-based, so 0 and any id sym_of() cannot de-intern are raw
    integers) -- and db.query returns those columns verbatim, including for
    a relation holding only STORED facts (no rules need be loaded).
    """

    def open_db(self):
        self.dir = tempfile.mkdtemp(prefix="le-render-db-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        return _dlb.Db.open(self.dir)

    def test_a_derived_row_renders_from_the_live_engine(self):
        # The full derived-facts leg: compile the rule, seed the facts, run
        # the query,
        # render the DERIVED tuple back -- it is the paper's sentence.  The
        # dates are INT columns (as in test_engine_roundtrip.py: a symbol
        # column cannot take the engine's >= comparison; MEASURED
        # 'column kind mismatch' on load), which also exercises the
        # sym/int discrimination in the rendered row's SYMBOL columns.
        db = self.open_db()
        try:
            result = compile_text(
                "A transaction is governed by IsdaAgreement if the "
                "transaction commences on a first day and IsdaAgreement is "
                "dated as of a second day and the first day is on or after "
                "the second day.",
                LEGAL,
            )
            db.declare_relation("commences", 2)
            db.declare_relation("dated", 2)
            tx, isa = db.intern("acmetransaction"), db.intern("isdaagreement")
            db.add_fact("commences", (tx, 100))
            db.add_fact("dated", (isa, 50))
            db.load_rules(result.rules)
            db.compile_rules()
            rows = [tuple(r) for r in db.query("governs", collect=True)]
            self.assertEqual(len(rows), 1)
            sentences = render_db(db, [("governs", rows[0])], LEGAL)
            self.assertEqual(
                sentences, ("AcmeTransaction is governed by IsdaAgreement.",)
            )
        finally:
            db.close()

    def test_a_stored_meta_row_renders_back_to_its_source_sentence(self):
        db = self.open_db()
        try:
            source = compile_text(
                "AcmeConfirmation of AcmeTransaction states that "
                "AcmeTransaction is governed by IsdaAgreement.",
                LEGAL,
            )
            db.declare_relation_variadic("states")
            from le.driver import intern_facts

            for pred, cols in intern_facts(db, source.facts):
                db.add_fact(pred, cols)
            rows = [tuple(r) for r in db.query("states", collect=True)]
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                render_db(db, [("states", rows[0])], LEGAL),
                (
                    "AcmeConfirmation of AcmeTransaction states that "
                    "AcmeTransaction is governed by IsdaAgreement.",
                ),
            )
        finally:
            db.close()

    def test_an_int_column_renders_as_digits_render_only(self):
        db = self.open_db()
        try:
            db.declare_relation("commences", 2)
            tx = db.intern("acmetransaction")
            db.add_fact("commences", (tx, 100))
            rows = [tuple(r) for r in db.query("commences", collect=True)]
            self.assertEqual(
                render_db(db, [("commences", rows[0])], LEGAL),
                ("AcmeTransaction commences on 100.",),
            )
        finally:
            db.close()

    def test_sym_of_is_the_discriminator_not_id_magnitude(self):
        # A raw integer BELOW the interned range must not masquerade as an
        # id: 0 de-interns to nothing (ids are 1-based) and renders as
        # digits; a real id renders as its symbol.
        db = self.open_db()
        try:
            db.declare_relation("commences", 2)
            tx = db.intern("acmetransaction")
            db.add_fact("commences", (tx, 0))
            rows = [tuple(r) for r in db.query("commences", collect=True)]
            self.assertEqual(rows, [(tx, 0)])
            self.assertEqual(
                render_db(db, [("commences", rows[0])], LEGAL),
                ("AcmeTransaction commences on 0.",),
            )
        finally:
            db.close()

    def test_unknown_relation_rows_still_render_through_the_lexicon(self):
        # The caller names the rows; the renderer reads the lexicon.  A
        # relation the lexicon has no template for is the loud no-template
        # gap -- never a silent skip.
        db = self.open_db()
        try:
            with self.assertRaises(CompileError) as ctx:
                render_db(db, [("w_orphaned", (1, 2))], LEGAL)
            self.assertEqual(ctx.exception.gap, GAP_RENDER_NO_TEMPLATE)
        finally:
            db.close()

    def test_a_non_db_object_is_a_type_error(self):
        with self.assertRaises(TypeError):
            render_db(object(), [("commences", (1, 2))], LEGAL)

    def test_a_non_u32_column_is_loud(self):
        db = self.open_db()
        try:
            with self.assertRaises(CompileError) as ctx:
                render_db(db, [("commences", ("acmetransaction", 2))], LEGAL)
            self.assertEqual(ctx.exception.gap, GAP_RENDER_SHAPE)
        finally:
            db.close()

    def test_an_empty_row_is_loud(self):
        db = self.open_db()
        try:
            with self.assertRaises(CompileError) as ctx:
                render_db(db, [("commences", ())], LEGAL)
            self.assertEqual(ctx.exception.gap, GAP_RENDER_SHAPE)
        finally:
            db.close()

    def test_db_rows_take_the_template_escape(self):
        # render_db mirrors render_facts's template= escape: an ambiguous
        # pred in a row stream no longer fails the whole call with no
        # remedy.
        db = self.open_db()
        try:
            db.declare_relation("before", 2)
            mon, wed = db.intern("monday"), db.intern("wednesday")
            db.add_fact("before", (mon, wed))
            rows = [tuple(r) for r in db.query("before", collect=True)]
            with self.assertRaises(CompileError) as ctx:
                render_db(db, [("before", rows[0])], LEGAL)
            self.assertEqual(ctx.exception.gap, GAP_RENDER_AMBIGUOUS_TEMPLATE)
            got = render_db(
                db, [("before", rows[0])], LEGAL, template=BEFORE_DAY
            )
            self.assertEqual(got, ("Monday is the day before Wednesday.",))
        finally:
            db.close()


class TestCanonicalTemplate(unittest.TestCase):
    """The [render] ini section and the canonical= map: the ambiguity
    escapes beyond template=.  Absent everywhere, nothing changes."""

    LEGAL_WITH_RENDER = None  # built once, below

    def test_ini_section_marks_the_canonical_template(self):
        text = open(EXAMPLES + "/lexicon.ini", encoding="utf-8").read()
        lexicon = Lexicon.from_text(
            text + "\n[render]\nbefore = <dayA> is before <dayB>\n"
        )
        self.assertEqual(
            lexicon.render_canonical, {"before": "<dayA> is before <dayB>"}
        )
        got = render_facts(("before(monday, wednesday).",), lexicon)
        self.assertEqual(got, ("Monday is before Wednesday.",))
        # The rules path honours it too, without template=.
        sentence = render_rule(DAY_CHAIN_DL, lexicon)
        self.assertEqual(compile_text(sentence, lexicon).rules, DAY_CHAIN_DL + "\n")

    def test_canonical_map_overrides_the_ini(self):
        text = open(EXAMPLES + "/lexicon.ini", encoding="utf-8").read()
        lexicon = Lexicon.from_text(
            text + "\n[render]\nbefore = <dayA> is before <dayB>\n"
        )
        got = render_facts(
            ("before(monday, wednesday).",), lexicon,
            canonical={"before": "<dayA> is the day before <dayB>"},
        )
        self.assertEqual(got, ("Monday is the day before Wednesday.",))

    def test_without_any_escape_the_ambiguity_stays_loud(self):
        with self.assertRaises(CompileError) as ctx:
            render_facts(("before(monday, wednesday).",), LEGAL)
        self.assertEqual(ctx.exception.gap, GAP_RENDER_AMBIGUOUS_TEMPLATE)

    def test_a_canonical_naming_an_undeclared_template_is_refused_at_load(self):
        text = open(EXAMPLES + "/lexicon.ini", encoding="utf-8").read()
        with self.assertRaises(CompileError) as ctx:
            Lexicon.from_text(text + "\n[render]\nbefore = <dayA> is after <dayB>\n")
        self.assertIn("not a declared template", str(ctx.exception))

    def test_old_inis_parse_byte_identically_without_render(self):
        # The two example lexicons have no [render] section; they must load
        # exactly as before (empty marking) and stay loud on 'before'.
        self.assertEqual(LEGAL.render_canonical, {})
        self.assertEqual(RPS.render_canonical, {})


if __name__ == "__main__":
    unittest.main()
