"""T8 -- determinism: the same input must compile to identical bytes.

MEASURES (a) that two compilations of the same text against the same lexicon
produce byte-identical rules and facts, and (b) that the output does not
depend on the process hash seed (so no set/dict iteration order leaks into the
emitted text).
"""

import os
import subprocess
import sys
import unittest

from fixtures import LEGAL_INI, ROOT

from le import Lexicon, compile_text

SOURCES = [
    "A transaction is governed by IsdaAgreement if the transaction commences "
    "on a first day and IsdaAgreement is dated as of a second day and the "
    "first day is on or after the second day.",
    "A day is before Wednesday if the day is before another day and the other "
    "day is before Wednesday.",
    "A player is eligible if it is not the case that the player is excluded "
    "and the player plays a choice.",
    "AcmeTransaction commences on Wednesday.",
    "A confirmation of a transaction is accepted if the confirmation of the "
    "transaction is received.",
]


class TestDeterminism(unittest.TestCase):
    def test_same_input_twice_is_byte_identical(self):
        for source in SOURCES:
            first = compile_text(source, LEGAL_INI)
            second = compile_text(source, LEGAL_INI)
            self.assertEqual(first, second, source)

    def test_fresh_lexicon_load_is_identical(self):
        lexicon = Lexicon.load(LEGAL_INI)
        for source in SOURCES:
            self.assertEqual(
                compile_text(source, lexicon), compile_text(source, LEGAL_INI)
            )

    def test_output_is_independent_of_the_hash_seed(self):
        script = (
            "import sys; sys.path.insert(0, %r);"
            "from le import compile_text;"
            "import json;"
            "print(json.dumps(["
            "[compile_text(s, %r).rules, list(compile_text(s, %r).facts)]"
            " for s in %r]))"
        ) % (ROOT, LEGAL_INI, LEGAL_INI, SOURCES)
        outputs = []
        for seed in ("0", "1", "12345"):
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


if __name__ == "__main__":
    unittest.main()
