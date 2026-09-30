# le — a Logical English → Datalog compiler

Logical English (LE; Kowalski, *Logical English*, LPOP 2020) is a controlled
natural language whose sentences translate **mechanically** into a logic
program. This package implements v1: the **declarative subset**, the
**reactive rule** form and the **meta-level embedding** (the use/mention
distinction of the paper's paragraph 18), compiling to
[datalog-dafsa](https://github.com/fixpoint-linux/datalog-dafsa) syntax, with a declared
lexicon driving a **closed** predicate set.

Python 3.11+, **stdlib only**. The dlb binding is imported by exactly one
`le` module — `le/driver.py`, lazily inside a function — and by the two
engine test files (`tests/test_engine_roundtrip.py`,
`tests/test_reactive_engine.py`), which *skip* when dlb or `libdatalog.so` is
absent. A binding that lives outside `sys.path` is found by setting
`DLB_PATH` to the directory that contains the `dlb` package (both the tests
and `le.render.render_db` honour it).

```python
from le import compile_text, CompileError, Lexicon, render_facts

result = compile_text(open("examples/paper_example2.le").read(), "examples/lexicon.ini")
result.rules     # dl rule text  -> dl_load_rules
result.facts     # ('commences(acmetransaction, wednesday).', ...) -> add_fact path
result.reactive  # (ReactiveRule(...),) -> le.driver.ReactiveDriver
render_facts(result.facts, lexicon)   # the reverse direction: back to LE
```

```console
$ python3 -m le examples/paper_example2.le
governs(V1, isdaagreement) :- commences(V1, V2), dated(isdaagreement, V3), V2 >= V3.
$ python3 -m le examples/facts.le --facts
commences(acmetransaction, wednesday).
dated(isdaagreement, monday).
$ python3 -m le examples/paper_example2_meta.le
governs(V1, isdaagreement) :- states(V2, governs, V1, isdaagreement), commences(V1, V3), dated(isdaagreement, V4), V3 >= V4.
$ python3 -m le examples/paper_example2.le --check
check: OK (1 rule(s), 0 fact(s))
$ python3 -m le examples/facts.le --json
{ "facts": [ ... ], "rules": "" }
$ python3 -m le examples/rps.le --reactive
{ "actions": ["plays", "receives"], "facts": [...], "reactive": [...], "rules": ... }
$ python3 -m le examples/facts.le --render
AcmeTransaction commences on Wednesday.
IsdaAgreement is dated as of Monday.
```

`--lexicon` defaults to the input path with a `.ini` suffix, else
`lexicon.ini` beside the input. `--check` runs a stdlib structural check of
the emitted rules (it does **not** load the engine; that verification lives in
the round-trip tests). `--reactive` prints the JSON driver manifest;
`--json` alone keeps exactly its two keys.

## The reverse direction (`le.render`: dl → Logical English)

The bridge is **bidirectional** by design (derived facts are
readable back in Logical English), and `le.render` is that half:

```python
from le import compile_text, render_facts, render_rule, render_result, render_db

result = compile_text(open("examples/facts.le").read(), "examples/lexicon.ini")
render_result(result, lexicon)
# ('AcmeTransaction commences on Wednesday.', 'IsdaAgreement is dated as of Monday.')
render_rule(result.rule_lines[0], lexicon)   # one dl rule line -> one sentence
```

* `render_facts(facts, lexicon, *, template=None, canonical=None)` —
  ground `pred(args).` lines → one sentence per line (`template=` mirrors
  `render_rule`'s escape: one fact stream, one pick for a shared
  predicate).
* `render_rule(rule_text, lexicon, *, template=None, canonical=None)` — one
  `head :- body.` line → `Conclusion if condition and condition.`
  The renderer **inverts the compiler's own binding walk**: head first, then
  the body atoms in the emitted (kind-sorted) order; a variable's first
  occurrence renders `a <noun>` and a later one `the <noun>`, and the
  ordinal comes from the variable's NUMBER — the counter is sequential, so
  its rank by number among the clause's same-sort variables IS the
  `(sort, ordinal)` binding key (`a first <noun>` for rank 2, …). That
  makes the inverse well-defined even when a negated or compared condition
  bound a variable that an earlier-EMITTED positive atom re-mentions
  (binding follows SOURCE order; the kind sort hides it): the sentence
  then recompiles to an alpha-equivalent rule — same atoms, one consistent
  renaming. Refused as non-emissions: a body outside the kind order, a
  variable only the conclusion mentions (the compiler's
  `GAP_EXISTENTIAL_HEAD`), a constant whose declared sort disagrees with
  its slot; a constant reverse-folds through `[names]` (or the caller's
  `names={symbol: surface}` display map); `!atom` becomes
  `it is not the case that …`; comparisons invert through the copula table
  (`>=` → `is on or after`); a `states(...)` row re-renders through its
  `[meta]` template, recovering the compiler's folded subject columns —
  from the mention's variables on a rule row, from the mention's VALUES on
  a ground DERIVED row (the evaluated meta-head rule), one
  sort-consistent reading required.
* `render_result(result, lexicon)` — the whole fact stream of a
  compilation.
* `render_db(db, rows, lexicon)` — **the derived-facts leg**: rows read out
  of a live dlb database (`(pred, cols)` pairs as `db.query(pred,
  collect=True)` returns them). The column→term step is its whole
  difference from `render_facts`: `db.sym_of(col)` de-interns a symbol id
  (ids are 1-based; `sym_of` is the discriminator, not the id's magnitude —
  a raw `0` or any id with no symbol renders as digits), and everything
  after that step **is** `render_facts`. It imports dlb **lazily**, inside
  the function, exactly like `le.driver` — `import le` stays stdlib-only.

Every refusal is a loud `CompileError` carrying a `GAP_RENDER_*` gap — a
predicate with no declared template (`le_antecedent_*` artifacts, `w_*`
witnesses, engine internals), an undeclared symbol (the renderer never
invents a proper noun), a fact line carrying a variable, a rule outside the
compiler's emission shape. **Known non-round-trippable cases**, documented
in the module docstring and pinned by tests:

* **Integers are render-only.** An int column renders as digits and the
  sentence displays fine, but the forward compiler refuses digits in a
  slot, so that row does not re-round-trip.
* **`another` / `the other` are information-lost in dl** (both lower to a
  fresh variable), so the canonical inverse is the ordinal form —
  `another day` renders back as `a first day`. The rendered sentence
  recompiles to the same dl; it is just not the original wording.

**Shared predicates** (`before` declares two templates in
`examples/lexicon.ini`) are a loud ambiguity — never a silent
first-declared pick — with three escapes, in override order:
`template=` (per call, on every entry point: `render_rule`,
`render_facts`, `render_result`, `render_db`), a `canonical={pred:
template text}` map, or the lexicon's ADDITIVE `[render]` section:

```ini
[render]
before = <dayA> is before <dayB>
```

(the value is the template's ini text, before the first `|`). Old inis —
no `[render]` — parse byte-identically and keep the loud behaviour; a
`[render]` value that is not a declared template of that predicate is
refused at load. `python -m le FILE --render` prints both compiled streams
read back as sentences (a reactive document renders its facts only: the
`le_antecedent_*` artifact has no LE sentence behind it — its source
sentence is the `--reactive` schema).

### Renderer tests (`tests/test_render.py`)

The strong pin is **dl → LE → dl identity** through the real
`compile_text` (an article swap is not a wrong-but-accepted render: the
compiler itself refuses it, so the oracle is loud by construction), plus
golden string pins and an LE → dl → LE stability pin — with each pin's
blind spot stated in the test docstrings. The db-path tests run against the
real engine and **skip** when dlb is absent, like the other engine tests.

## The language subset

    sentence  := clause '.'
    clause    := conclusion 'if' condition ('and' condition)*
    condition := [ negation ] atom | meta-condition
    negation  := 'it is not the case that'
    atom      := <template match> | <comparison copula>
    meta-cond := <subject noun phrases> <verb phrase> 'that' atom
    fact      := atom ('and' atom)*                # must be ground

* **Articles bind.** `a`/`an <noun>` introduces a variable at its first
  occurrence; `the <noun>` refers to it later; `first`/`second`/`third` and
  `another`/`the other` introduce further variables of the same sort.
  `first`/`second` are *discriminators*: `a first day` … `the first day` is one
  variable, `a second day` … `the second day` is another.
* **Nouns are sorts, verbs are predicates.** All of them come from the
  lexicon; a word that is not declared is a compile error.
* **Proper nouns are declared symbol constants** (`IsdaAgreement` →
  `isdaagreement`, `Wednesday` → `wednesday`) whose sort comes from the
  lexicon's `[names]` section, so a wrong-sort constant is a type clash rather
  than a silent misread. A proper noun the lexicon does not declare is a
  compile error. A *user variable name* is a letter followed by digits (`P1`,
  `C1`), and it is used verbatim as the dl variable.
* **Comparison copulas** are built in: `is before` → `<`, `is after` → `>`,
  `is on or after` → `>=`, `is on or before` → `<=`, and
  `<A> is <N> days before <B>` → the arithmetic producer `A = B - N`.
  A declared template takes precedence over a built-in copula, so a lexicon
  can give `is before` the semantics of a predicate instead (as
  `examples/lexicon.ini` does).
* Sentences are separated by `.` (each sentence must end with one); `#` starts
  a comment.

The compiler — not the source order — decides the body order: positive atoms
first (they bind every variable), then arithmetic producers in **dependency
order** (each producer's operand must be bound by a preceding atom, so
`… and the day is 3 days before a second day and the second day is 4 days
before a third day …` emits the second producer first), then negated atoms,
then comparisons. The engine requires a variable in a negated atom or a
comparison to be bound by a positive body atom first, so trusting the input
order would emit programs the engine rejects. Comparisons may be grounded by
arithmetic (the engine accepts it); negations may not (only a positive atom
grounds the engine's `unsafe negation` check).

**Two streams.** Facts never enter the rules text: `dl_load_rules` rejects
inline facts, so `CompileResult.facts` carries them for the `add_fact` path.

## Reactive rules (`If … then …`)

    react     := 'If' antecedent 'then' consequent '.'
    antecedent:= condition ('and' condition)*      # exactly a clause body
    consequent:= part ('and' part)*
    part      := <action atom>                     # deliberate: performed
               | 'it becomes the case that' atom   # fortuitous: state made true
               | 'it becomes the case that it is not the case that' atom

The paper (Kowalski 2020; example 1, the rock-paper-scissors game) says
reactive rules "represent goals that are made true by making their consequents
true whenever their antecedents **become** true", and that consequents are made
true "either deliberately by performing actions or fortuitously by observing
external events". That is a *transition*, and datalog-dafsa is a least-fixpoint
engine with no state transition — so a reactive rule compiles into two
artefacts and one driver:

* **the antecedent detection rule**, into `result.rules`, so everything the
  engine loads is still rule text:

      le_antecedent_1(P1, C1, P2, C2) :- plays(P1, C1), plays(P2, C2), beats(C1, C2), !over(rpsgame), P2 != P1.

  Its body is lowered exactly like a declarative clause body, and its head
  carries every variable the antecedent binds (explicit names in
  first-occurrence order, then `V`-names ascending) so a **row is the
  antecedent's substitution**.
* **the consequent schema**, into `result.reactive` — a `ReactiveRule` the
  driver executes. It is never engine text: an action is an output, a
  `becomes` is an `add_fact`/`delete_fact`.
* **`le.driver.ReactiveDriver`** executes it:

```python
import dlb
from le import compile_text, Lexicon
from le.driver import ReactiveDriver, intern_facts

lexicon = Lexicon.load("examples/rps.ini")
result = compile_text(open("examples/rps.le").read(), lexicon)
db = dlb.Db.open(directory)                      # the CALLER opens the database
driver = ReactiveDriver.from_compile(db, result, lexicon)
driver.enter()                                   # declare relations, load, compile
driver.seed(intern_facts(db, result.facts))      # initial state: the beats facts
step = driver.submit([("plays", (p1, rock)), ("plays", (p2, scissors))])
step.fired    # (FireRecord(1, (p1, rock, p2, scissors)),)
step.actions  # (ActionRecord('receives', (p1, prize)),)   <- deliberate action
step.added    # (('over', (game,)),)                       <- fortuitous update
step.deleted  # the two consumed plays facts
```

**A step is charged to the world's events.** `seed` adds initial-state facts
(no step); `submit(events)` runs one step; `run(batches, max_steps=…)` is the
event-driven convenience loop and raises `ValueError` rather than truncating
silently. The observation timeline starts at the empty state, so anything
already true when the first `submit` runs is an edge (call `submit([])` first
to make the seeded state the baseline).

**Edge, not level.** Per rule, the driver keeps the row set of the previous
observation; a row fires only when it was absent there — i.e. when the
antecedent *became* true. Firing also **consumes** the antecedent's event
facts (the `[actions]` predicates it read), so a held event cannot re-fire on
its own.

**A round reads first, then writes.** Within one round the driver queries
*every* rule's antecedent against the state the round started from and collects
the whole firing plan before applying anything. Two rules that read the same
`[actions]` predicate therefore both fire on the round's event — one rule's
firing cannot consume an event out from under a sibling rule that was entitled
to it.

**An event is consumed when it is processed.** Consumption is a state mutation
like any other, so a rule whose antecedent was false at the moment the event
was consumed never had a true antecedent: a guard that clears afterwards (a
later round of the same step, or a later step after the world changes) does not
resurrect the event. That is the LPS reading of an external event as a one-off
occurrence; keeping an event until every rule reading it has fired would need a
per-rule copy of each event and is not implemented (gap **g9**).

**Termination.** A `(rule, row)` pair fires at most once per step; a repeat
means the antecedent went false and true again inside one step, i.e. the
reaction is oscillating, and the step aborts loudly (`StepError`). The rounds
that **fire** are hard-bounded (100 by default) for genuinely growing cascades
— a cascade that settles in exactly 100 firing rounds completes, while one that
needs a 101st aborts loudly — also a `StepError`. On abort the driver restores
the pre-step state with a compensating transaction, the caller's submitted
events included, so an aborted step leaves nothing behind. There is no internal
loop that can hang.

### `[actions]`: the event vocabulary

```ini
[predicates]                        # static knowledge and state
<a choice> beats <a choice> | beats | choice, choice
<a game> is over            | over  | game

[actions]                           # external events: performed, then consumed
<a player> plays <a choice>    | plays    | player, choice
<a player> receives <a prize>  | receives | player, prize
```

Same line format as `[predicates]`. Membership is the single switch between
"state, which merely holds" and "event, which the driver consumes once a rule
that read it has fired"; the same predicate may not be declared in both
sections (loud). Consequent actions must be `[actions]` — a `[predicates]`
template cannot be performed (`GAP_CONSEQUENT_NOT_ACTION`) — and every
consequent term must be a variable bound in the antecedent or a declared
constant: a consequent may not introduce a term of its own
(`GAP_CONSEQUENT_SHAPE`), because its meaning would be existential and v1
refuses to guess.

Two binding refinements are confined to a reactive **antecedent** (declarative
output is untouched, byte for byte):

* `another <noun>` is a distinctness claim, so the reactive lowering emits
  `P2 != P1` guards (the engine accepts a variable-variable `!=`; the paper's
  "another player P2" is exactly this). Ordinals are not — `a choice C1 … a
  choice C2` are two variables, not two guaranteed-different values, which is
  what lets both players play the same choice;
* a **second named** `a <noun>` of an already-bound sort is allowed (the
  paper's own "a choice C1 … a choice C2"), where an anonymous repeat stays the
  loud `GAP_REPEAT_A` usage error.

### Paper fidelity

**Implemented** from the paper: the reactive sentence form (example 1, modulo
the rendering of the two noun phrases below), edge-triggered firing when the
antecedent becomes true, consequents as deliberate actions *and* as
observation/fluent updates, destructive updates of a single current state held
as engine facts, the emergent frame axiom (a fluent persists until deleted —
nothing re-derives it), and a potentially infinite event stream driven by the
caller (`run`).

**Not implemented, stated rather than hidden:**

* **g1 cross-sentence anaphora.** The paper's example (1) writes "the game" and
  "the prize" with no antecedent in the clause; LE's documented binding rules
  are per clause, and v1 does not read across sentences. `examples/rps.le`
  therefore declares `RpsGame`/`RpsPrize` in `[names]` and uses them — a
  declared constant instead of a guessed referent.
* **g2 consequent forms.** Only the three forms above; no nested reactive
  composition, no action arguments that are themselves new terms.
* **g3 no priority/conflict resolution** beyond document order: rules fire in
  document order, rows in the engine's deterministic order, consequent parts in
  declaration order, and simultaneous add/delete of one fact resolves
  last-wins inside the round's single transaction.
* **g4 no time-stamped history.** Only the current state plus the per-rule
  previous-observation sets are kept; there is no event log or model of states.
* **g5 no external-event validation.** Any fact of a declared `[actions]`
  predicate is accepted as an event; the world-check (that the event really
  happened) is the host application's, still unwired. An
  `ActionRecord` mirrors `ActionCall(predicate, args)` for a one-line adapter,
  and the driver does **not** write the action's own fact.
* **g6** `when` conditions and the rest of the event calculus (paper ex. 3).
* **g7** no arithmetic producers that reference consequent-added variables
  (subsumed by the consequent-shape rule).
* **g8** the antecedent rule is query-only: there is no separate public
  "detect" API.
* **g9 no per-reader event copies.** An event fact is consumed when the round
  that processes it commits, so a rule whose guard was false at that moment
  never fires on it — the event is gone and no new edge arrives. Keeping an
  event until every rule that mentions it has fired (the other LPS reading)
  would require a per-rule copy of each event fact. Measured reading: the
  event is a one-off occurrence, consumed when processed.
* **g10 the driver's magic/top-down queries over a variadic program.**
  `ReactiveDriver.from_compile` now declares the lexicon's `[meta]` reifying
  relations variadic itself, so a reactive rule whose **antecedent** contains a
  meta condition needs nothing from the caller (measured end to end in
  `tests/test_reactive_engine.py`). What remains is the engine limitation the
  plan recorded: magic-set/top-down queries over a program that contains a
  variadic relation are rejected — the driver only ever calls `db.query` (full
  fixpoint, `driver.py`) and `query_rules_ro`, both measured working over
  variadics.

This feature deviates from the obvious reading of the paper in several
places, each because the engine or the paper said otherwise; they are listed
under **Design decisions and divergences** below.

## Meta-level conditions (`states that …`)

Paper example (2) puts an *object-level sentence inside a meta-level one*:

> A transaction is governed by IsdaAgreement if **a confirmation of the
> transaction states that the transaction is governed by IsdaAgreement** …

Paragraph 18 of the paper says this embedding "is represented in Prolog by
translating the phrase *is governed by* **both** into a predicate symbol, which
is *used* at the object-level in the conclusion of the sentence, **and** into a
function symbol, which is *mentioned* at the meta-level in a condition of the
sentence." That is the classic **use/mention** distinction, and confusing the
two is the silent-wrong failure of this feature: a compiler that emits
`states(Conf) :- governs(T, A)` (the mention as another body atom of the outer
clause) inverts the semantics while looking reasonable.

**The encoding.** datalog-dafsa is function-free, so the *mentioned* sentence
is reified into **flat columns** — a predicate-symbol constant plus the
mentioned arguments:

```
'A transaction is governed by IsdaAgreement if a confirmation of the
 transaction states that the transaction is governed by IsdaAgreement and …'
        |->
governs(V1, isdaagreement) :- states(V2, governs, V1, isdaagreement), commences(V1, V3), dated(isdaagreement, V4), V3 >= V4.
```

Column order is `states(<subject terms>, <mentioned predicate>, <its
arguments>)`, so reading column 2 as the function symbol recovers the paper's
`states(Conf, governs(T, A))`. Two `Conf`s are the same relation, so the reader
sees one row per mention. The values are ordinary u32 columns: a mention can be
**posted as a fact** through `add_fact`, which is what makes the encoding
reachable from the existing binding (an interned list/term representation is
not: dlb exposes no term constructor).

**Shared variables.** The mention is resolved against the *same*
`BindingState` as the outer clause, so `the transaction` inside the mention is
the head's own `V1` — the paper's sharing, not a capture. A subject **variable**
the mention already carries back is not repeated as a column (that is why the
row above has one `V1`, not two); every other subject slot is a column of its
own. The fold keys on *variable identity* only — one column per entity — and a
**constant** subject (a proper noun) is always kept as a column, even when the
mention carries the same constant back, because folding it would let two
different `[meta]` verb phrases reify onto the same row:

```
'A transaction is governed by IsdaAgreement if a party alongside a
 confirmation states that the transaction is governed by IsdaAgreement.'
        |->
governs(V1, isdaagreement) :- states(V2, V3, governs, V1, isdaagreement).
```

**The `[meta]` declaration.** The verb phrases live in the lexicon, so a second
one costs a declaration, never code (`tests/test_meta.py` measures
`requires that` compiling the same way):

```ini
[meta]
<a confirmation> of <a transaction> states that <atom> | states | confirmation, transaction, _
<a party> requires that <atom>                        | requires | party, _
```

An entry is `<subject slots> <verb phrase> that <atom> | pred | <subject sorts>,
_`: every slot of a normal template except the last, which must be the single
`<atom>` slot (the mentioned sentence), declared `_` because the mention's
arguments are typed by the template it mentions. `pred` names the reifying
relation; that relation is **variadic** (one column count per mentioned
predicate), so `le.cli.validate_dl` exempts `lexicon.meta_preds` from its
one-arity-per-predicate rule — `--check` passes a document that mentions
`governs`/2 and `over`/1 in the same relation.

**The mention is matched by the ordinary machinery.** The predicate after
`states that` must be a *declared* template — the closed lexicon holds inside a
mention, so an invented predicate is still `GAP_CLOSED_LEXICON` — and its
arguments type-check against *its own* template's sorts. A meta condition is
tried **only after** the ordinary matcher has failed for the whole span, so
every non-meta sentence takes exactly the path it took before this feature
existed (the four pre-existing example documents compile byte-identically).

**Refused, loudly** (`tests/test_meta.py`): a mention that is not a positive
declared atom (`GAP_META_SHAPE` — a comparison, an arithmetic form, a negated
mention, a mention of a mention, a subject that does not match its `[meta]`
form), a mention whose reified row would need more than the engine's 8 columns
(`GAP_META_ARITY`, whose message names the numbers), and a `[meta]` declaration
whose verb phrase is a variable (`GAP_META_VARIABLE_PRED`).

**A coordination is never absorbed into a mention.** `and` splits the *condition
list* before the meta match runs, and a meta condition consumes exactly ONE atom
(the atom bounded by the `and`), so a mention can never swallow a second
conjunct. That split is what the paper's example (2) itself relies on — its meta
condition is followed by three further conjuncts that are ordinary body atoms.
The consequence is stated rather than hidden: when the conjunct after the
mention is itself a declared atom it compiles as an ordinary condition of the
clause (and the mention reifies the FIRST atom only — a mention *of a
coordination* has no encoding in v1), while a fragment that is not an atom is
refused by name (`GAP_CLOSED_LEXICON`/`GAP_META_SHAPE`). Both halves are pinned
in `tests/test_meta.py::TestMentionShape`.

**Residual (still a loud gap):** the paper's extended form, example (6) — the
relative clause (`a day that is on or after the day as of which …`) and the
functional `the confirmation`. A meta condition can also be negated as a whole
(`it is not the case that … states that …` → `!states(…)`, with the ordinary
negation-grounding rules), but the *mentioned* sentence may not be. A meta
condition is legal in a **head** as well as in a body or a fact — it lowers to an
ordinary positive atom, so a rule can *conclude* a mention
(`… states that … :- the confirmation of the transaction is received.` →
`states(V1, governs, V2, isdaagreement) :- received(V1, V2).`) — and the driver
declares the reifying relation variadic itself, so a reactive rule whose
antecedent reads a mention needs nothing from the caller (g10 now covers only
the engine's magic/top-down limitation over variadics).

## The lexicon (`[sorts]`, `[names]`, `[predicates]`, `[actions]`, `[meta]`)

```ini
[sorts]
transaction = transaction   # noun surface -> sort symbol
day                         # bare = identity

[names]
AcmeTransaction = transaction   # proper noun -> its declared sort
Wednesday = day

[predicates]
<a transaction> commences on <a day> | commences | transaction, day
<a confirmation> of <a transaction> is accepted | accepted | confirmation, transaction
```

One `[predicates]` line is `<template> | pred | arg sorts`. The template
interleaves literal words with `<slot>` markers (a slot marker is a *name* for
the slot; `of` chains are written as `of` between two slots). The declared
sorts — in slot order, `_` meaning "any sort" — are what type-checks a
compilation. If two declared templates match the same span, the compilation
refuses with an ambiguity error rather than picking one silently. Predicate
names must be lowercase-initial dl identifiers: the engine parses an uppercase
head as a *variable* and rejects the program.

`[names]` gives every proper noun used in a sentence its sort, so the slot
sorts type-check constants exactly as they type-check variables. It is the
option that fits the existing file format (another `section` of `key = value`
lines) and it keeps a proper noun usable in any typed slot, where the
alternative — requiring the slot to be `_` — would make constants unusable in
every declared slot. An undeclared proper noun is `GAP_UNDECLARED_NAME`.

`[actions]` uses the same line format and is described under
**Reactive rules** above: membership is the one thing that separates state
from a consumable external event, and a predicate may not be in both sections.

`[meta]` declares the meta-level verb phrases and is described under
**Meta-level conditions** below. Its entries end with the single `<atom>`
slot, and the literal run before that slot (the verb phrase) must end in
`that` — so `that` is reserved in every other section, where a template
containing it could not be told apart from a relative clause.

## What v1 refuses (every one is a loud `CompileError` naming the gap)

| gap | example |
|---|---|
| malformed reactive rule | `If a player P1 plays a choice C1.` (no `then`), an empty half, a second `then`/`if` |
| a consequent that is not an action | `… then P1 receives a prize.` when `receives` is `[predicates]` |
| a consequent that introduces a term | `… then P1 receives a prize.` (a new prize variable), `… then it is not the case that a game is over.` |
| fluent update / event calculus | `It becomes the case that a requirement is defaulted on a day.` (paper ex. 3) — anywhere except as a reactive consequent prefix |
| meta mention that is not a declared atom | a comparison (`… states that the first day is on or after the second day.`), arithmetic, a negated mention, a mention of a mention, a subject that does not match the `[meta]` declaration |
| meta mention too wide | subject slots + 1 predicate column + the mentioned arity > 8 (`… states that a day is logged in a week in a month in a year in an era in a second year.`) |
| a variable mentioned predicate | a `[meta]` template with a slot where the verb phrase belongs (`<a day> states <a phrase> <atom>`) |
| relative clauses | `… a day that is on or after the day as of which IsdaAgreement is dated.` (paper ex. 6) |
| `when` conditions | `A day is before Wednesday when a day is before Monday.` |
| plurals and quantifiers | `… if all days are before Wednesday.` |
| conjunctive conclusions | `A player receives a prize and a prize is delivered if …` (a *declarative* clause; the reactive consequent is the form that sequences them) |
| **ontology extension** | any undeclared predicate — the predicate set is closed |
| undeclared proper noun | `Foo commences on Wednesday.` unless `Foo` is in `[names]`; a declared name of the wrong sort is a type clash |
| existential head variables | a variable in the conclusion that no positive condition binds (paper ex. 4/5, skolemization) |
| unstratified negation | a clause whose body is only negation; a negated atom whose variables no *positive* atom binds (an arithmetic producer does not ground a negation) |
| arithmetic producers | a result variable produced twice (the engine would silently answer with one of the two rows), a cycle or self-reference (`the day is 4 days before the day`), an operand no preceding atom binds |
| `the other` with two `another`s in scope | `… before another day and another day is before Monday and the other day is before Tuesday.` (which one it denotes is undecidable) |
| symbol constants in ordering comparisons | `… if the day is on or after Wednesday.` (the engine rejects it) |
| ground-fact violation | a variable inside a fact sentence |

Usage errors are equally loud: `the day` with no antecedent, `a day` twice in
one clause (`a second day` twice as well), a bare common noun, an explicit name
used before its noun, a comparison between two different sorts, a missing
final `.`.

`CompileError.gap` carries the gap name, so a caller can branch on it.

## Design decisions and divergences

An early draft of the binding design read `the other <noun>` as introducing a
*fresh* binding.
That reading contradicts the paper, which writes "Monday is the day before
**another** day and **the other** day is before Wednesday" — the second clause
must chain through the first variable. Implemented as the paper says:
`the other <noun>` refers to the most recent `another <noun>` of that sort, and
is an error when there is none. (`tests/test_paper_examples.py` measures the
chain.)

Two further deliberate choices, both documented in the code: proper nouns fold
to lowercase bare dl symbols (the paper's `Monday` → `monday`), and a
compilation that would fold two distinct proper nouns onto one symbol is
refused rather than silently merged — the fold map is per *document*, so the
check also covers two sentences apart.

### Meta-level conditions

1. **One column per distinct term, and the cap counts the FOLDED row.** The
   column order is `states(<subject slots>, <pred>, <mention args>)`
   and the acceptance emission is `states(V2, governs, V1, isdaagreement)`
   for the paper's example, whose `[meta]` template has *two* subject slots
   (`<a confirmation> of <a transaction>`). Emitting one column per subject slot
   would give `states(V2, V1, governs, V1, isdaagreement)` — the same variable in
   two columns — so a subject **variable** the mention carries back is not
   repeated (the paper's `states(Conf, governs(T, A))`). The fold is keyed on
   variable identity, so it never applies to a constant subject: a proper noun
   stays a column even when the mention repeats it, which departs from the
   draft's own expected *fact*
   (`states(conf1, governs, acmetransaction, isdaagreement)`) but is what keeps
   two different `[meta]` verb phrases from reifying onto the same row (measured
   pre-fix: both emitted `states(governs, acmetransaction, isdaagreement)`, the
   subject column gone). The engine's arity cap is checked on the row that is
   actually emitted — the folded column count — so a mention that carries a
   subject term back still fits: a 6-argument mention whose first argument is the
   meta condition's own transaction emits 8 columns and compiles, where counting
   the declared subject slots refused it with "9 columns". One consequence,
   stated rather than hidden: a mention whose arguments already carry *every*
   subject **variable** yields a row with no subject column
   (`states(pred, args…)`) — rules and facts fold identically, so the program
   stays consistent, and the variadic relation holds both shapes.

### Reactive rules

1. **A step is one transaction per ROUND, not per step.** A naive reading of
   the feature would use one
   transaction per step; measured, a read (`query`/`lookup`/`count`) inside an
   open transaction cannot see that transaction's writes, so round 2 could not
   observe round 1 and an internally cascading reaction would never fire. Each
   round is still all-or-nothing, and an aborted step restores the pre-step
   state with one compensating transaction (an already-committed round cannot
   be rolled back by the engine). The sole-writer contract makes that exact.
2. **Only `another` claims distinctness, and the head carries only what the
   antecedent BINDS.** `P2 != P1` comes from `another player P2` (the paper's
   own words, and example (1)'s expected output); ordinals introduce further
   *variables* without an inequality, which is what lets both players play the
   same choice. A variable that appears only inside a negation is not a head
   column, so the engine's stratified-negation check reports
   `GAP_NEG_BINDING` instead of a head-variable error.
3. **`receives` is `[actions]` in `examples/rps.ini`.** A first draft of the
   example left it in `[predicates]`, but a consequent check requires a
   performed consequent to be an action — the paper calls "P1 receives the
   prize" a deliberate action — so the [predicates] reading would not compile
   example (1) at all. For the same reason an action is **recorded, not
   written**: an earlier draft said both "action record + `add_fact(receives)`"
   and, in
   its final decision (and its end-to-end expectation), that a bare action
   consequent only records. The final decision wins — performing an action is
   the world's job, so the driver must not fabricate its effect.
4. **Two binding refinements, reactive-antecedent only.** A second *named*
   `a <noun>` of an already-bound sort is allowed (example (1)'s expected
   output has `a choice C1 … a choice C2`, which the declarative
   `GAP_REPEAT_A` rule refuses); an anonymous repeat stays loud. The paper
   does not mention this — its own sentence forced it.
5. **A variable-free antecedent is padded.** A naive encoding assumes an
   ordinary rule
   per reactive rule, but the engine rejects a 0-arity rule head (measured:
   `head arity 0 for variadic`), so `If RpsGame is over then …` emits
   `le_antecedent_1(0) :- over(rpsgame).` and the driver binds no variables.
6. **"the game"/"the prize" are declared constants.** An earlier draft
   resolved the
   game this way and did not mention the prize; g1 covers both.
7. **No `ReactiveProgram` type, and the CLI tests live in `test_cli.py`.** An
   early sketch mentioned a manifest type once and never defined it — the CLI
   builds the
   manifest from `CompileResult` plus the `Lexicon` — and the CLI tests sit
   with the other CLI tests instead of in `test_reactive.py`.

## Tests

```console
python3 -m unittest discover -s tests -v
```

272 tests, all passing on the development host (Python 3.13.14, libdatalog.so
ABI 1) — including the engine tests that load the compiled output (and run a
compiled reactive rule) against the real engine. Redirecting `HOME` so
the engine cannot be found turns exactly those into skips, so the suite is
green (and honestly labelled) on a host without dlb.

| id | file | what it measures |
|---|---|---|
| T1 | `test_lexicon.py` | lexicon loading ([sorts]/[names]/[predicates]); the closed predicate set (drift control); proper-noun sort typing |
| T2 | `test_paper_examples.py` | paper example (2), VERBATIM (with the meta-level `states that` condition) → the acceptance emission |
| T3 | `test_paper_examples.py` | example (3)'s relational rewrite: `another`/`the other` (and the two-`another` ambiguity), the `of` chain |
| T4 | `test_comparisons.py` | copula lowering (`>=`, `<`, `>`), body reordering, `N days before` incl. multi-producer chains, dependency order, refusals |
| T5 | `test_negation.py` | `it is not the case that` → `!atom`, reordered after the positives; negation refusals (incl. arithmetic-bound vars) |
| T6 | `test_errors.py` | the error battery (binding errors, declared gaps, facts, names, the document-wide fold check) |
| T7 | `test_engine_roundtrip.py` | **dlb only, skips if absent**: rules load and run; facts round-trip through `add_fact`; symbol identity; negation; arithmetic-grounded head var; a reordered multi-producer chain; the reified mention (derives from a *posted* mention, three controls, two mention arities in one variadic relation) |
| T8 | `test_determinism.py` | same input → byte-identical output, independent of `PYTHONHASHSEED` |
| T9 | `test_cli.py` | CLI smoke (`--facts`, `--json`, `--check`, `--reactive`, `--render`, manifest keys, error exit) + the `validate_dl` checker |
| T10 | `test_reactive.py` | the reactive surface: example (1)'s antecedent rule and schema, the `another` guards, the consequent forms and every one of their refusals, the `[actions]` lexicon, reactive determinism under `PYTHONHASHSEED` |
| T11 | `test_reactive_engine.py` | **dlb only, skips if absent**: example (1) end to end (one firing, the action record, the state update, the consumed plays), edge vs level, a shared event reaching every rule that reads it, one report per change of a fact, the consumed-event reading (no resurrection by a later guard clear), document-order firing and last-wins collisions, the stratified-negation flip, a reactive antecedent that reads a **mention** (the driver declaring the variadic relation itself), and every termination guard (oscillation abort + state restore incl. a same-round collision and the caller's own submitted events, the firing-round bound at its boundary, `run`'s `max_steps`) |
| T12 | `test_meta.py` | the meta-level `states that` embedding: the verbatim paper emission, the use/mention boundary (textual + the closed lexicon inside a mention), the shared variables, the variable-identity fold (a constant subject stays a column; the no-subject-column extreme), the folded arity cap, every mention shape refusal, a coordination as a condition list, a meta condition as a conclusion, negated meta conditions, meta facts, a meta condition in a reactive antecedent (and a meta consequent refused), the `[meta]` declaration diagnoses, a second meta verb, and hash-seed determinism |
| T13 | `test_render.py` | the REVERSE direction: dl → LE → dl identity (the loud oracle), golden sentence pins, LE → dl → LE stability, every `GAP_RENDER_*` refusal, the live-db path against the real engine (**skips if absent**), the `[render]` canonical section and the `canonical=` map |

Claim marking: the compiled outputs quoted above and the test
results are **MEASURED**; the descriptions of *why* a gap exists (LE/LPS
semantics) are **INTERPRETATION** from the paper; anything about how a
generator would behave under this grammar is **PROJECTION** and is not claimed
here.

## Licence

MIT (see `LICENSE`). The implementation is released under it; this repository
implements the grammar of Logical English as set out in Robert Kowalski,
*Logical English*, LPOP 2020 (a position paper). The paper is its author's,
and the examples reproduced above are quoted to specify the language this
compiler accepts.
