"""Article/ordinal variable binding for one Logical English clause.

The model (Kowalski 2020, "Logical English", LPOP 2020 position paper,
https://www.doc.ic.ac.uk/~rak/papers/LPOP.pdf):
"the scope of the universally quantified variables is limited to the clause";
``a``/``an <noun>`` introduces the variable at its first occurrence, ``the``
refers to it later, and further variables of the same type are introduced by
``first``/``second``/``another``/``the other``.

Design consequences, all of them loud rather than silent (state the gap;
never silently drop):

* ``a day`` twice in one clause is a USAGE error, not a quiet second entity --
  the language resolves a second entity with ``another``/``a second``.
* ``the day`` with no antecedent is an error, not a fresh variable.
* ``the other day`` refers to the most recent ``another day`` (the paper's
  "Monday is the day before another day and the other day is before
  Wednesday").  NOTE: an early draft of this design read ``the other`` as a
  fresh binding; that reading contradicts the paper and the test battery, so
  it is not followed here -- the paper's chain reading is implemented.
* Proper nouns are declared symbol constants: their sort comes from the
  lexicon's ``[names]`` section, so a wrong-sort constant is a loud type clash
  rather than a silent misread.  A proper noun the lexicon does not declare is
  refused.  The declared symbol is the lowercased surface when that is a legal
  dl bare symbol, otherwise a double-quoted string constant.

REACTIVE MODE (``BindingState(..., reactive=True)``, used by
:mod:`le.reactive` for the antecedent of an LPS reactive rule).  Two
refinements, both confined to reactive antecedents so the declarative output
is untouched:

* ``another <noun>`` is a distinctness claim: the paper's example (1) writes
  "a player P1 ... another player P2" precisely to say P2 is NOT P1.  In
  reactive mode every ``another`` introduction records a ``new != old`` pair
  for each earlier variable of the same sort already bound in the clause; the
  reactive lowering turns those into ``!=`` guards (MEASURED: the engine
  accepts a var-var ``!=`` and filters exactly the self-pair rows).
* a second unanchored ``a <noun>`` of an already-bound sort is normally the
  loud GAP_REPEAT_A error, because a silent second entity is a misread.  The
  paper's own example (1) writes "a choice C1 ... a choice C2", though: when
  BOTH the bound and the new noun phrase carry a DISTINCT explicit name, the
  two entities are explicit rather than silent, so reactive mode allows the
  second binding (an anonymous repeat stays loud).  A later ``the <noun>``
  then refers to the most recent introduction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import CompileError
from .lexicon import GAP_UNDECLARED_NAME, Lexicon, NP, is_auto_var

_SYMBOL = re.compile(r"[a-z][a-z0-9_]*\Z")

GAP_REPEAT_A = (
    "'a <noun>' introduces one entity per sort and ordinal; use 'another "
    "<noun>' or a NEW ordinal ('a third <noun>') for a distinct one"
)
GAP_UNBOUND_THE = "unbound 'the <noun>' (introduce it with 'a <noun>' first)"
GAP_THE_OTHER = "'the other <noun>' with no preceding 'another <noun>'"
GAP_THE_OTHER_AMBIGUOUS = (
    "'the other <noun>' with more than one 'another <noun>' of that sort in "
    "scope -- which one it refers to is undecidable"
)
GAP_BARE_NOUN = "a common noun needs an article ('a day'), an ordinal ('a first day') or a name"
GAP_NAME_UNBOUND = "an explicit variable name used before the noun it names"
GAP_NAME_REUSED = "an explicit variable name bound twice in one clause"
GAP_NAME_AUTO = "an explicit variable name that collides with auto-generated V-names"
GAP_SYMBOL_COLLISION = "two proper nouns folding to the same dl symbol"
GAP_NAME_ON_REF = "an explicit name on a back-reference ('the <noun> P1')"


@dataclass(frozen=True)
class Term:
    """A resolved dl term: a variable or a symbol/integer constant."""

    kind: str  # 'var' | 'const'
    text: str
    sort: str | None
    surface: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


class Counter:
    """The compilation-wide variable counter (deterministic per input)."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


class BindingState:
    """Per-clause binding state: (sort, discriminator) -> dl variable name."""

    def __init__(
        self,
        lexicon: Lexicon,
        counter: Counter | None = None,
        consts: dict | None = None,
        *,
        reactive: bool = False,
    ) -> None:
        self.lexicon = lexicon
        self.counter = counter if counter is not None else Counter()
        self.bindings: dict = {}
        self.named: dict = {}
        self.others: dict = {}
        self.reactive = reactive
        # Variables introduced per sort, in introduction order, plus the
        # (sort, new, old) distinctness pairs 'another' records (reactive mode).
        self.sort_vars: dict = {}
        self.distinct_pairs: list = []
        # The fold map is shared across the whole compilation (one per
        # document, not per clause) so that two proper nouns folding onto one
        # dl symbol are caught even when they sit in different sentences.
        self.consts = consts if consts is not None else {}

    # -- helpers -----------------------------------------------------------

    def _fresh(self, name: str | None, *, sentence, index) -> str:
        if name is None:
            self.counter.n += 1
            return f"V{self.counter.n}"
        if is_auto_var(name):
            raise CompileError(
                f"explicit variable name '{name}' collides with the "
                "auto-generated V-names",
                gap=GAP_NAME_AUTO,
                sentence=sentence,
                index=index,
            )
        if name in self.named:
            raise CompileError(
                f"explicit variable name '{name}' is bound twice in one clause",
                gap=GAP_NAME_REUSED,
                sentence=sentence,
                index=index,
            )
        return name

    def _constant(self, surface: str, *, sentence, index) -> str:
        low = surface.lower()
        if not _SYMBOL.match(low):
            return '"' + surface.replace("\\", "\\\\").replace('"', '\\"') + '"'
        previous = self.consts.get(low)
        if previous is not None and previous != surface:
            raise CompileError(
                f"proper nouns '{previous}' and '{surface}' both fold to the "
                f"dl symbol '{low}'",
                gap=GAP_SYMBOL_COLLISION,
                sentence=sentence,
                index=index,
            )
        self.consts[low] = surface
        return low

    # -- resolution --------------------------------------------------------

    def resolve_np(self, np: NP, *, sentence=None, index=None) -> Term:
        """Resolve one noun phrase to a dl term, binding if it introduces one."""
        if np.proper is not None:
            sort = self.lexicon.sort_of_name(np.proper)
            if sort is None:
                raise CompileError(
                    f"the proper noun '{np.proper}' has no declared sort; add "
                    f"'{np.proper} = <sort>' to the lexicon's [names] section "
                    "(a constant with no sort cannot be type-checked)",
                    gap=GAP_UNDECLARED_NAME,
                    sentence=sentence,
                    index=index,
                )
            return Term(
                "const", self._constant(np.proper, sentence=sentence, index=index),
                sort, np.proper,
            )

        if np.noun is None:
            assert np.name is not None
            if np.name not in self.named:
                raise CompileError(
                    f"'{np.name}' is used before it is introduced (write "
                    f"'a <noun> {np.name}' at its first occurrence)",
                    gap=GAP_NAME_UNBOUND,
                    sentence=sentence,
                    index=index,
                )
            sort, var = self.named[np.name]
            return Term("var", var, sort, np.name)

        if np.bare:
            raise CompileError(
                f"bare common noun '{np.noun}' needs a determiner",
                gap=GAP_BARE_NOUN,
                sentence=sentence,
                index=index,
            )

        sort = self.lexicon.sort_of(np.noun)
        assert sort is not None
        art, ordinal = np.art, np.ordinal
        key = (sort, ordinal)

        if np.name is not None and art in ("the", "the_other"):
            raise CompileError(
                f"'{np.text}' names a variable that is already bound",
                gap=GAP_NAME_ON_REF,
                sentence=sentence,
                index=index,
            )

        if art == "a":
            if key in self.bindings:
                existing = self.bindings[key]
                # Reactive antecedents may introduce a second entity of an
                # already-bound sort when both noun phrases carry a distinct
                # explicit name (the paper's "a choice C1 ... a choice C2"):
                # the names make the two entities explicit, not silent.
                if not (
                    self.reactive and np.name is not None and existing in self.named
                ):
                    if ordinal is None:
                        detail = (
                            f"'{np.text}' is a second unanchored binding of sort "
                            f"'{sort}' in this clause"
                        )
                    else:
                        detail = (
                            f"'{np.text}' re-uses the ordinal '{ordinal}' already "
                            f"bound for sort '{sort}' in this clause"
                        )
                    raise CompileError(
                        detail, gap=GAP_REPEAT_A, sentence=sentence, index=index
                    )
            return self._bind(key, sort, np, sentence=sentence, index=index)

        if art == "the":
            if key not in self.bindings:
                raise CompileError(
                    f"'{np.text}' has no antecedent in this clause",
                    gap=GAP_UNBOUND_THE,
                    sentence=sentence,
                    index=index,
                )
            return Term("var", self.bindings[key], sort, np.text)

        if art == "another":
            var = self._fresh(np.name, sentence=sentence, index=index)
            if np.name is not None:
                self.named[np.name] = (sort, var)
            if self.reactive:
                # 'another' claims a distinct entity: guard it against each
                # earlier variable of the same sort in this clause.
                for other in self.sort_vars.get(sort, ()):
                    self.distinct_pairs.append((sort, var, other))
            self.sort_vars.setdefault(sort, []).append(var)
            self.others.setdefault(sort, []).append(var)
            return Term("var", var, sort, np.text)

        if art == "the_other":
            candidates = self.others.get(sort) or []
            if not candidates:
                raise CompileError(
                    f"'{np.text}' has no preceding 'another {np.noun}'",
                    gap=GAP_THE_OTHER,
                    sentence=sentence,
                    index=index,
                )
            if len(candidates) > 1:
                raise CompileError(
                    f"'{np.text}' is ambiguous: {len(candidates)} 'another "
                    f"{np.noun}' of sort '{sort}' are in scope in this clause, "
                    "so which one 'the other' refers to is undecidable",
                    gap=GAP_THE_OTHER_AMBIGUOUS,
                    sentence=sentence,
                    index=index,
                )
            return Term("var", candidates[-1], sort, np.text)

        # Bare ordinal: 'first day' with no article -- refer if bound, else
        # introduce (the paper only uses the 'a first day'/'the first day'
        # pair, and this is the lenient reading of that pair).
        if key in self.bindings:
            return Term("var", self.bindings[key], sort, np.text)
        return self._bind(key, sort, np, sentence=sentence, index=index)

    def _bind(self, key, sort, np, *, sentence, index) -> Term:
        var = self._fresh(np.name, sentence=sentence, index=index)
        self.bindings[key] = var
        self.sort_vars.setdefault(sort, []).append(var)
        if np.name is not None:
            self.named[np.name] = (sort, var)
        return Term("var", var, sort, np.text)
