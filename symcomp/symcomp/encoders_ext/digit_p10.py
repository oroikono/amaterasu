"""digit_p10 -- Charton-style base-10 digit numerals in the prefix skeleton.

Coefficient-numeral-tokenization axis (minimal pair vs `lample_charton`):
coefficients are quantized to the SAME 8 bins FIRST (via
symcomp.encoders._q -- do NOT skip pre-binning), then the single bin token
of the enc_lample_charton prefix skeleton is replaced by a fixed 5-token
sign/mantissa/exponent numeral: constant sign token `+N`, three base-10
digit tokens of round(bin*100) zero-padded to width 3, and constant
exponent token `E-2`. Only the numeral FORM varies; binning, tree
structure, and term order are identical to the baseline.

Normative examples (from the arm spec, matched exactly):
  advection:1.0
    -> [*, c, +N, 1, 0, 0, E-2, d1, u]                        (9 tokens)
  advection:1.0+diffusion:0.5+dispersion:0.3   (0.3 -> bin 0.25 -> 025)
    -> [+, *, c, +N, 1, 0, 0, E-2, d1, u,
        +, *, c, +N, 0, 5, 0, E-2, d2, u,
           *, c, +N, 0, 2, 5, E-2, d3, u]                     (29 tokens)

Max length: 9 tokens/term + 1 '+' between terms -> 29 for a 3-term
operator (<= 48 budget). Sequence length is a covariate vs the 17-token
lample_charton baseline -- record it in analysis.

Vocab: {+, *, c, u, +N, E-2} + {d0..d4} + realized digits. The spec's
expected_vocab_size=21 counts all ten digits 0-9; the 8 bins only realize
mantissas {025,050,075,100,125,150,175,200} -> digits {0,1,2,5,7}, so the
vocab actually built over the Stage-A universe is 16. Not a deviation from
the token scheme, just the spec's expectation counting the full digit
alphabet. Injectivity on the universe follows from that of the baseline
skeleton (bin token -> digit block is a bijection).

Pure deterministic function of the Operator: integer arithmetic on exactly
representable bin values, no hashing, no randomness, no IO.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator, PRIMITIVES

KEY = "digit_p10"

_SIGN = "+N"   # constant sign token (all universe coefficients positive)
_EXP = "E-2"   # constant exponent token: mantissa * 10^-2


def _numeral(bin_tok: str) -> list[str]:
    """Bin token (e.g. '0.5') -> [+N, d, d, d, E-2] fixed 3-digit mantissa."""
    mantissa = int(round(float(bin_tok) * 100.0))
    return [_SIGN, *f"{mantissa:03d}", _EXP]


def encode(op: Operator) -> list[str]:
    toks: list[str] = []
    names = op.names()
    for i, n in enumerate(names):
        if i < len(names) - 1:
            toks.append("+")
        order = PRIMITIVES[n]["order"]
        toks += ["*", "c", *_numeral(_q(op.coeffs[n])), f"d{order}", "u"]
    return toks
