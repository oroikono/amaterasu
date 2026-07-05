"""unary_order -- derivative order in UNARY (exploratory Stage AX arm).

Primitive-decomposability axis: instead of opaque per-order symbols (d1, d2,
d3, ...), the derivative order of each mechanism is spelled out as a stack of
repeated single 'd' tokens, so higher-order mechanisms are literal
compositions of one primitive token (dispersion = d d d) rather than atoms.

Token scheme (normative examples from the arm spec):
  Per term, terms in alphabetical mechanism order (Operator.names()), joined
  by a '+' separator token:
      [coef, <_q coefficient bin>, 'd' * order, u]
  reaction (order 0) has no 'd' tokens: [coef, <bin>, u].

  advection:1.0                       -> [coef, 1, d, u]
  advection:1.0+diffusion:0.5+dispersion:0.3 ->
      [coef, 1, d, u, +, coef, 0.5, d, d, u, +, coef, 0.25, d, d, d, u]
  (17 tokens; dispersion's 0.3 quantizes to the 0.25 bin via encoders._q).

Injectivity on the linear universe: the linear PRIMITIVES have pairwise
distinct derivative orders (0..4), so the unary 'd' run-length of each term
uniquely identifies its mechanism and the (mechanism -> coefficient-bin) map
is recoverable from the token string.

Deviation note: the spec quotes "max_tokens_3term": 18 (hyperdiffusion worst
case), but the actual worst 3-term operator in the universe is
diffusion+dispersion+hyperdiffusion = 5 + 6 + 7 term tokens + 2 separators
= 20 tokens. The token scheme itself is unchanged (the normative examples are
matched exactly); only the spec's worst-case arithmetic was off. 20 <= 48, so
the budget contract holds.

Scope: linear (constant-coefficient) mechanisms only; nonlinear primitives
have no derivative order in PRIMITIVES and raise ValueError.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator, PRIMITIVES

KEY = "unary_order"


def encode(op: Operator) -> list[str]:
    """Pure deterministic tokenizer: Operator -> list of string tokens."""
    toks: list[str] = []
    for i, name in enumerate(op.names()):  # names() is sorted alphabetically
        if name not in PRIMITIVES:
            raise ValueError(
                f"unary_order encoder supports linear primitives only, got {name!r}"
            )
        if i > 0:
            toks.append("+")
        toks.append("coef")
        toks.append(_q(op.coeffs[name]))
        toks.extend(["d"] * int(PRIMITIVES[name]["order"]))
        toks.append("u")
    return toks
