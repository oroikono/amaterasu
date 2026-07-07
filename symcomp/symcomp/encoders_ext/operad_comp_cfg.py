"""operad_comp_cfg -- CFG with EXPLICIT BINARY COMPOSITION morphisms over DX.

Minimal pair against `deriv_cfg`. That arm serializes a derivative of order n
as an implicit unary chain of n R_DX productions. This arm instead makes the
composition morphism explicit: the single generator is dx, and higher-order
derivatives are built by a binary COMP operad node,

    DERIV -> COMP(DERIV, DERIV) | dx

prefix-serialized under a FIXED balanced bracketing (normative; do not deviate):

    T(0) = []                                (reaction emits an empty tower;
                                              no ID token)
    T(1) = [DX]
    T(n) = [COMP] + T(ceil(n/2)) + T(floor(n/2))

so  T(2) = [COMP, DX, DX]
    T(3) = [COMP, COMP, DX, DX, DX]          (T(2) then T(1))
    T(4) = [COMP, COMP, DX, DX, COMP, DX, DX] ([COMP] + T(2) + T(2))

Everything else matches deriv_cfg exactly (the ONE varied axis is explicit
binary composition vs implicit unary chaining):

    EQ    -> u_t = SUM               R_EQ
    SUM   -> PROD | SUM + PROD       R_SUM->PROD / R_SUM->SUM+PROD
    per term: [spine, R_PROD, COEF_<_q bin>, tower T(order), R_U]

Terms in canonical (alphabetical mechanism) order; coefficient bins shared
with all symbolic arms (_q). Nonlinear mechanisms raise (unreachable in the
linear Stage universe).

Normative examples:
  advection:1.0 -> [R_EQ, R_SUM->PROD, R_PROD, COEF_1, DX, R_U]
  advection:1.0+diffusion:0.5+dispersion:0.3 ->
    [R_EQ, R_SUM->PROD, R_PROD, COEF_1, DX, R_U,
     R_SUM->SUM+PROD, R_PROD, COEF_0.5, COMP, DX, DX, R_U,
     R_SUM->SUM+PROD, R_PROD, COEF_0.25, COMP, COMP, DX, DX, DX, R_U]
    (22 tokens)

Worst 3-term (hyperdiffusion+dispersion+diffusion) = 1 + 11 + 9 + 7 = 28
tokens. Injective on the linear universe: mechanism orders are pairwise
distinct, so each term's tower uniquely identifies its mechanism.
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "operad_comp_cfg"


def _tower(n: int) -> list:
    """Prefix serialization of the fixed balanced composition tree for dx^n."""
    if n == 0:
        return []
    if n == 1:
        return ["DX"]
    return ["COMP"] + _tower((n + 1) // 2) + _tower(n // 2)


def encode(op):
    toks = ["R_EQ"]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(
                f"operad_comp_cfg: nonlinear mechanism {n} unsupported")
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append("R_PROD")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        toks.extend(_tower(int(PRIMITIVES[n]["order"])))
        toks.append("R_U")
    return toks
