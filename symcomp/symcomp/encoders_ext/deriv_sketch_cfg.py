"""deriv_sketch_cfg -- deriv_cfg grammar with a SPINE-FIRST (sketch) AR order.

Zero-covariate linearization arm: identical grammar, vocabulary, token
multiset, and length to `deriv_cfg`; ONLY the autoregressive factorization
order changes. Instead of interleaving the SUM-spine productions with the
term blocks (the rightmost-expansion order deriv_cfg uses), this arm emits
the TRUE LEFTMOST derivation of

    EQ    -> u_t = SUM              R_EQ
    SUM   -> SUM + PROD | PROD      R_SUM->SUM+PROD / R_SUM->PROD
    PROD  -> COEF * DERIV           R_PROD
    DERIV -> dx ( DERIV ) | u       R_DX / R_U
    COEF  -> quantized bin          COEF_<bin>

for a LEFT-leaning SUM spine: the whole spine plan comes first -- (n-1)
copies of R_SUM->SUM+PROD, then one R_SUM->PROD, right after R_EQ -- and
only then the n per-term blocks [R_PROD, COEF_<bin>, R_DX * order, R_U] in
canonical (alphabetical mechanism) order. The model must therefore commit
to the number of terms ("the sketch") before generating any term content.
Per-term blocks contain NO spine tokens. 1-term operators are byte-identical
to deriv_cfg. Coefficient bins are shared with all symbolic arms (_q).
Reaction (order 0) is COEF * u. Nonlinear mechanisms are out of scope and
raise (unreachable in the linear Stage universe).

advection:1.0 ->
  [R_EQ, R_SUM->PROD, R_PROD, COEF_1, R_DX, R_U]
advection:1.0+diffusion:0.5+dispersion:0.3 ->
  [R_EQ, R_SUM->SUM+PROD, R_SUM->SUM+PROD, R_SUM->PROD,
   R_PROD, COEF_1, R_DX, R_U,
   R_PROD, COEF_0.5, R_DX, R_DX, R_U,
   R_PROD, COEF_0.25, R_DX, R_DX, R_DX, R_U]      (19 tokens)

Worst 3-term (diffusion+dispersion+hyperdiffusion) = 4 spine/eq tokens
+ (5 + 6 + 7) term tokens = 22 tokens.
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_sketch_cfg"


def encode(op):
    names = op.names()  # canonical alphabetical order
    for n in names:
        if n not in PRIMITIVES:
            raise ValueError(
                f"deriv_sketch_cfg: nonlinear mechanism {n} unsupported")
    # Spine plan first: (n-1) continuation rules, THEN the single terminator.
    toks = ["R_EQ"]
    toks.extend(["R_SUM->SUM+PROD"] * (len(names) - 1))
    toks.append("R_SUM->PROD")
    # Term blocks: no spine tokens inside.
    for n in names:
        toks.append("R_PROD")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        toks.extend(["R_DX"] * int(PRIMITIVES[n]["order"]))
        toks.append("R_U")
    return toks
