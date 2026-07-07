"""deriv_cd_cfg -- deriv_cfg with COUNTDOWN-indexed derivative towers.

Tower positional-readability arm. Identical spine / rules / coefficient bins
/ term order / lengths to `deriv_cfg`; the ONLY change is the derivative
tower: the unary run of R_DX (order repetitions of one token) becomes an
indexed COUNTDOWN

    DX_order, DX_{order-1}, ..., DX_1

so the descent is deterministic once the tower-opening token (DX_order) is
seen -- entropy is concentrated at the tower opening instead of spread over
a run-length decision. The COUNTDOWN direction is load-bearing: always
DX_3, DX_2, DX_1 -- never count-up.

Grammar (as in deriv_cfg):
    EQ    -> u_t = SUM              R_EQ
    SUM   -> PROD | SUM + PROD      R_SUM->PROD / R_SUM->SUM+PROD
    PROD  -> COEF * DERIV           R_PROD
    DERIV -> indexed dx tower | u   DX_k countdown / R_U
    COEF  -> quantized bin          COEF_<bin>

Per term: [spine, R_PROD, COEF_<_q bin>, DX_order, ..., DX_1, R_U].
Order-0 (reaction) has no DX tokens. Terms in canonical (alphabetical
mechanism) order; R_EQ prefix. Nonlinear mechanisms raise (unreachable in
the linear Stage universe).

advection:1.0 -> [R_EQ, R_SUM->PROD, R_PROD, COEF_1, DX_1, R_U]
advection:1.0+diffusion:0.5+dispersion:0.3 ->
  [R_EQ, R_SUM->PROD, R_PROD, COEF_1, DX_1, R_U,
   R_SUM->SUM+PROD, R_PROD, COEF_0.5, DX_2, DX_1, R_U,
   R_SUM->SUM+PROD, R_PROD, COEF_0.25, DX_3, DX_2, DX_1, R_U]  (19 tokens)

Worst 3-term (orders 4+3+2) = 1 + (4+4) + (4+3) + (4+2) = 22 tokens.
Vocab: 5 rule tokens + 8 COEF bins + DX_1..DX_4 = 17 (universe-dependent).
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_cd_cfg"


def encode(op):
    toks = ["R_EQ"]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(f"deriv_cd_cfg: nonlinear mechanism {n} unsupported")
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append("R_PROD")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        order = int(PRIMITIVES[n]["order"])
        toks.extend(f"DX_{k}" for k in range(order, 0, -1))
        toks.append("R_U")
    return toks
