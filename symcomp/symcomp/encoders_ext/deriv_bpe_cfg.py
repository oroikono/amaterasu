"""deriv_bpe_cfg -- deriv_cfg with the R_DX tower run-length chunked (BPE-1).

Single-chunk tower arm (exploratory Stage AX). BPE induction over the
deriv_cfg tower strings converges in one merge, (R_DX, R_DX) -> R_DX2, so a
derivative tower of order n is emitted in greedy normal form

    [R_DX2] * (n // 2) + [R_DX] * (n % 2)

i.e. the semigroup presentation with the derived generator D2. This is the
ONE AXIS varied against deriv_cfg: the spine (R_EQ, R_SUM->PROD /
R_SUM->SUM+PROD), R_PROD, COEF_<_q bin>, R_U, term order (alphabetical via
Operator.names()) and coefficient binning are all byte-identical, and
order-0/1 terms (reaction, advection) are token-identical to deriv_cfg.

Normative examples (from the arm spec, matched exactly):
  advection:1.0 -> [R_EQ, R_SUM->PROD, R_PROD, COEF_1, R_DX, R_U]
  advection:1.0+diffusion:0.5+dispersion:0.3 ->
    [R_EQ, R_SUM->PROD, R_PROD, COEF_1, R_DX, R_U,
     R_SUM->SUM+PROD, R_PROD, COEF_0.5, R_DX2, R_U,
     R_SUM->SUM+PROD, R_PROD, COEF_0.25, R_DX2, R_DX, R_U]   (17 tokens)
  (dispersion's 0.3 quantizes to the 0.25 bin via encoders._q.)

Injectivity on the linear universe: greedy binary/unary normal form is a
bijection on derivative orders (0 -> [], 1 -> [R_DX], 2 -> [R_DX2],
3 -> [R_DX2, R_DX], 4 -> [R_DX2, R_DX2]), and the linear PRIMITIVES have
pairwise distinct orders, so each term's tower uniquely identifies its
mechanism; coefficient bins ride along per term.

Worst 3-term (diffusion+dispersion+hyperdiffusion) =
1 + (4+1) + (4+2) + (4+2) = 18 tokens <= 48 budget.

Scope: linear (constant-coefficient) mechanisms only; nonlinear mechanisms
raise ValueError (unreachable in the linear Stage universe). Implemented as a
byte-identical reimplementation of deriv_cfg's skeleton (the spec's allowed
alternative to importing the sibling module) so the import surface stays
within symcomp.encoders / symcomp.operators.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator, PRIMITIVES

KEY = "deriv_bpe_cfg"


def encode(op: Operator) -> list[str]:
    """Pure deterministic tokenizer: Operator -> list of string tokens."""
    toks: list[str] = ["R_EQ"]
    for i, name in enumerate(op.names()):  # names() is sorted alphabetically
        if name not in PRIMITIVES:
            raise ValueError(
                f"deriv_bpe_cfg: nonlinear mechanism {name!r} unsupported"
            )
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append("R_PROD")
        toks.append(f"COEF_{_q(op.coeffs[name])}")
        order = int(PRIMITIVES[name]["order"])
        toks.extend(["R_DX2"] * (order // 2))  # greedy normal form: D2 chunks
        toks.extend(["R_DX"] * (order % 2))    # + at most one residual D
        toks.append("R_U")
    return toks
