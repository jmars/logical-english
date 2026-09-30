"""Logical English -> datalog-dafsa compiler.

    from le import compile_text, CompileError, Lexicon

    result = compile_text(open("paper_example2.le").read(), "legal.ini")
    result.rules     # dl rule text, for dl_load_rules
    result.facts     # tuple of dl fact lines, for the add_fact path
    result.reactive  # tuple of ReactiveRule, for the reactive driver

WHAT THIS IS.  Logical English (Kowalski, LPOP 2020) is a controlled natural
language whose sentences translate mechanically into a logic program.  This
package implements three sentence forms:

* the DECLARATIVE subset -- `conclusion if condition and condition ... .`
  clauses and ground fact sentences, with article/ordinal variable binding, a
  declared lexicon of sorts and predicate templates, and the built-in
  comparison copulas;
* the REACTIVE form -- `If antecedent then consequent.` (paper example 1),
  compiled to an antecedent-detection rule plus a consequent schema that the
  driver in :mod:`le.driver` executes.  The driver needs the dlb engine
  binding and is therefore NOT imported here: use
  ``from le.driver import ReactiveDriver``.
* the META-LEVEL embedding -- `<subject noun phrases> states that <atom>`
  (paper example 2), the use/mention distinction of the paper's paragraph 18:
  the mentioned predicate is reduced to a TERM, so `governs` is USED as a
  predicate in the conclusion and MENTIONED as a symbol inside the reified
  relation.  The verb phrases come from the lexicon's `[meta]` section, so a
  second meta verb (`requires that ...`) costs a declaration, not code.

THE REVERSE DIRECTION.  ``le.render`` renders the compiler's own output back
into Logical English: ``render_facts``/``render_result`` for the fact
stream, ``render_rule`` for a rule line, and ``render_db`` for rows read out
of a live dlb database (that one imports dlb lazily, like the driver).

WHAT THIS IS NOT (each is a loud CompileError naming the gap -- the gap is
stated, never silently dropped):

* paper example (6)'s extended form -- the relative clause (`a day that is on
  or after the day as of which ...`) and the functional `the confirmation`;
* relative clauses and the functional `the`;
* existential head variables / skolemization            -- paper examples 4/5;
* plurals and quantifiers, conjunctive conclusions;
* `it becomes the case that` outside a reactive consequent, and the rest of
  the event calculus vocabulary (`when`, time stamps) -- paper example 3;
* ontology extension: the predicate set is CLOSED.  An undeclared predicate is
  a compile error, never a silent no-op -- inside a meta condition too.
  Proper nouns (symbol constants) must likewise be declared, with their sort,
  in the lexicon's `[names]` section, and the predicates a reactive consequence
  may perform must be declared in `[actions]`.
"""

from .binding import BindingState, Counter, Term
from .errors import CompileError, CompileResult
from .lexicon import Lexicon, NP, Template, match_atom
from .lower import GAP_CLOSED_LEXICON, compile_text
from .reactive import Arg, Consequent, ReactiveRule
from .render import render_db, render_facts, render_result, render_rule

__all__ = [
    "Arg",
    "BindingState",
    "CompileError",
    "CompileResult",
    "Consequent",
    "Counter",
    "GAP_CLOSED_LEXICON",
    "Lexicon",
    "NP",
    "ReactiveRule",
    "Template",
    "Term",
    "compile_text",
    "match_atom",
    "render_db",
    "render_facts",
    "render_result",
    "render_rule",
]
