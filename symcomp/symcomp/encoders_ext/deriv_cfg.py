"""deriv_cfg -- CFG over DERIVATIVE OPERATORS, not mechanism names.

The core `grammar` arm treats mechanisms as opaque terminals (MECH_advection).
Physically, mechanisms are derivative expressions sharing substructure:
advection = c*dx(u), diffusion = nu*dx(dx(u)), hyperdiffusion = dx^4(u).
This arm serializes the leftmost derivation of the equation under

    EQ    -> u_t = SUM              R_EQ
    SUM   -> PROD | SUM + PROD      R_SUM->PROD / R_SUM->SUM+PROD
    PROD  -> COEF * DERIV           R_PROD
    DERIV -> dx ( DERIV ) | u       R_DX / R_U
    COEF  -> quantized bin          COEF_<bin>

so every mechanism is a tower of the SAME R_DX production and composites
share subsequences with their parts. Terms in canonical (alphabetical
mechanism) order; coefficient bins shared with all symbolic arms (_q).
Reaction (order 0) is COEF * u. Nonlinear mechanisms are out of scope and
raise (unreachable in the linear Stage universe).

advection:1.0 -> [R_EQ, R_SUM->PROD, R_PROD, COEF_1, R_DX, R_U]
advection:1.0+diffusion:0.5 ->
  [R_EQ, R_SUM->PROD, R_PROD, COEF_1, R_DX, R_U,
   R_SUM->SUM+PROD, R_PROD, COEF_0.5, R_DX, R_DX, R_U]
Worst 3-term (with hyperdiffusion) = 1 + 3 terms * <=8 = ~22 tokens.
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_cfg"


def encode(op):
    toks = ["R_EQ"]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(f"deriv_cfg: nonlinear mechanism {n} unsupported")
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append("R_PROD")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        toks.extend(["R_DX"] * int(PRIMITIVES[n]["order"]))
        toks.append("R_U")
    return toks
