"""deriv_infix -- the equation tokenized as plain infix math text over
derivative operators: how a paper (or Lample-Charton's original math corpus)
would write it, with dx applications spelled out and no rule tokens.

advection:1.0 -> [u_t, =, 1, *, dx, u]
advection:1.0+diffusion:0.5 -> [u_t, =, 1, *, dx, u, +, 0.5, *, dx, dx, u]

Same _q coefficient bins as every symbolic arm; canonical term order;
reaction (order 0) is <bin> * u. Worst 3-term ~ 2 + 3*7 = 23 tokens.
Nonlinear mechanisms raise (unreachable in the linear Stage universe).
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_infix"


def encode(op):
    toks = ["u_t", "="]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(f"deriv_infix: nonlinear mechanism {n} unsupported")
        if i > 0:
            toks.append("+")
        toks.append(str(_q(op.coeffs[n])))
        toks.append("*")
        toks.extend(["dx"] * int(PRIMITIVES[n]["order"]))
        toks.append("u")
    return toks
