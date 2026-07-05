"""fourier_symbol -- semantic-ceiling arm: tokenize what L DOES, not its syntax.

Syntax-vs-semantics axis. Instead of any expression-tree serialization, we
sample the Fourier multiplier L_hat(xi) of the (coefficient-binned) operator at
xi in {1, 2, 4, 8} and tokenize sign + log2-magnitude of the real and imaginary
parts. If a representation's compositional win is about *structure alignment*
rather than *semantic content*, this arm bounds what pure semantics buys.

Scheme (normative formula from the arm spec):
  1. Pre-bin coefficients with the shared quantizer ``symcomp.encoders._q`` and
     build a binned Operator (sorted mechanism order, so summation order in
     ``fourier_symbol`` is canonical).
  2. Evaluate ``Operator.fourier_symbol`` at xi = (1, 2, 4, 8) -- the repo's
     sign conventions are the source of truth for the emitted values.
  3. Per xi emit 5 tokens:  ``xi{k} re <tok> im <tok>``  where for value v
        <tok> = 'z'                      if |v| < 1e-12
              = '<s>m<b>'                otherwise, s in {p, n} = sign(v),
                                         b = clip(floor(log2 |v|) + 3, 0, 16)
     (1-octave magnitude bins; a 2-octave variant was audited non-injective).

Always exactly 20 tokens per operator. Design vocabulary: 4 xi markers + 're'
+ 'im' + 'z' + 34 fused sign-magnitude tokens (pm0..pm16, nm0..nm16) = 41.

Spec-interpretation note (documented deviation-avoidance, not a deviation):
the spec's phrase "sign token ('p'/'n') followed by magnitude-bin token
'm<b>'" is realized as ONE fused token per value (e.g. 'pm5', 'nm3'), not two
tokens. This is the only reading consistent with the spec's own normative
arithmetic: max_tokens_3term = 20 = 4 xi * 5 tokens, and expected_vocab_size
= 41 = 4 + 2 + 1 + 2*17. Two separate tokens would give 20-28 variable-length
sequences and vocab 26.

Purity: deterministic arithmetic only -- no hash(), no RNG, no IO; stable
under any PYTHONHASHSEED. Injectivity on the Stage-A universe (5 linear
mechanisms, singletons/pairs/triples at the default coefficients) is verified
by the arm self-test with 1-octave bins.
"""
from __future__ import annotations

import numpy as np

from symcomp.encoders import _q
from symcomp.operators import Operator

KEY = "fourier_symbol"

_XI = (1, 2, 4, 8)
_ZERO_TOL = 1e-12
_B_MIN, _B_MAX = 0, 16


def _val_tok(v: float) -> str:
    """'z' for (numerically) zero, else fused sign + 1-octave log2-mag bin."""
    if abs(v) < _ZERO_TOL:
        return "z"
    s = "p" if v > 0.0 else "n"
    b = int(np.clip(np.floor(np.log2(abs(v))) + 3.0, _B_MIN, _B_MAX))
    return f"{s}m{b}"


def encode(op: Operator) -> list[str]:
    """Tokenize sign/log-magnitude-binned samples of L_hat(xi), xi in _XI."""
    # bin coefficients with the shared quantizer; sorted names() makes the
    # accumulation order inside fourier_symbol canonical (order-independent).
    binned = Operator({n: float(_q(op.coeffs[n])) for n in op.names()})
    sym = binned.fourier_symbol(np.asarray(_XI, dtype=float))
    toks: list[str] = []
    for k, v in zip(_XI, sym):
        toks.append(f"xi{k}")
        toks.append("re")
        toks.append(_val_tok(float(v.real)))
        toks.append("im")
        toks.append(_val_tok(float(v.imag)))
    return toks
