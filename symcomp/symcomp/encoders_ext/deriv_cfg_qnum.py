"""deriv_cfg_qnum -- deriv_cfg with COEFFICIENTS AS COMPOSITIONAL UNARY NUMERALS.

ONE-AXIS minimal pair vs `deriv_cfg`: everything byte-identical EXCEPT that the
atomic coefficient terminal COEF_<bin> is replaced by a grammar-derived unary
numeral. The shared quantized bin (via symcomp.encoders._q) is bin = k * 0.25
for k in 1..8, and we emit exactly round(bin / 0.25) repetitions of the single
token N_Q. Smallest grammar of the round: 7 token types total
{R_EQ, R_SUM->PROD, R_SUM->SUM+PROD, R_PROD, N_Q, R_DX, R_U}.

Derivation grammar (leftmost derivation serialized, as in deriv_cfg):

    EQ    -> u_t = SUM              R_EQ
    SUM   -> PROD | SUM + PROD      R_SUM->PROD / R_SUM->SUM+PROD
    PROD  -> COEF * DERIV           R_PROD
    COEF  -> Q | Q COEF             N_Q  (unary numeral, value = count * 0.25)
    DERIV -> dx ( DERIV ) | u       R_DX / R_U

Per term (alphabetical mechanism order, R_EQ prefix):
    [spine, R_PROD, N_Q * round(bin/0.25), R_DX * order, R_U]

Injectivity: each linear mechanism has a distinct derivative order, so the
R_DX run length identifies the mechanism; the N_Q run length identifies the
bin (bins are 0.25..2.0 step 0.25, so counts 1..8, never zero); runs are
delimited by distinct neighboring tokens. NORMATIVE examples:

advection:1.0 ->
  [R_EQ, R_SUM->PROD, R_PROD, N_Q, N_Q, N_Q, N_Q, R_DX, R_U]            (9)
advection:1.0+diffusion:0.5+dispersion:0.3 ->
  [R_EQ, R_SUM->PROD, R_PROD, N_Q, N_Q, N_Q, N_Q, R_DX, R_U,
   R_SUM->SUM+PROD, R_PROD, N_Q, N_Q, R_DX, R_DX, R_U,
   R_SUM->SUM+PROD, R_PROD, N_Q, R_DX, R_DX, R_DX, R_U]                 (23)

The N_Q count uses the BINNED value (round(bin/0.25)), not the raw
coefficient. Nonlinear mechanisms are out of scope and raise (unreachable in
the linear Stage universe). Worst case in the default universe is 24 tokens;
spec bound 43 (<= max_len 48) holds for any 3-term operator over the bins.
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_cfg_qnum"

_Q_STEP = 0.25


def encode(op):
    toks = ["R_EQ"]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(f"deriv_cfg_qnum: nonlinear mechanism {n} unsupported")
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append("R_PROD")
        k = round(float(_q(op.coeffs[n])) / _Q_STEP)
        toks.extend(["N_Q"] * k)
        toks.extend(["R_DX"] * int(PRIMITIVES[n]["order"]))
        toks.append("R_U")
    return toks
