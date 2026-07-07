"""typed_sketch_cd_cfg -- push-the-number arm stacking THREE axes onto the
current leader deriv_typed_cfg, completing the factorial with deriv_sketch_cfg
and deriv_cd_cfg:

  1. ALGEBRA TYPING (from deriv_typed_cfg): every term derives inside the
     subgrammar of its operator-algebra class, and the dx terminals are
     class-typed. Class function is IDENTICAL to deriv_typed_cfg:
       src  = order 0 (reaction; scalar source, NO DX tokens)
       skew = odd order (advection dx^1, dispersion dx^3)
       sym  = even order >= 2 (diffusion dx^2, hyperdiffusion dx^4)
  2. PLAN-FIRST SPINE (sketch): ALL additive-spine productions are emitted
     up front, before any term body -- the model reads/writes the arity plan
     of the sum before deriving any term:
       [R_EQ] + (n-1) x [R_SUM->SUM+PROD] + [R_SUM->PROD]
  3. TYPED COUNTDOWN TOWERS (cd): the derivative tower of an order-k term of
     class c is spelled DX_<c>_k, DX_<c>_{k-1}, ..., DX_<c>_1 -- each dx
     token carries both its algebraic type and how many applications remain.

Terms appear in alphabetical mechanism order (op.names()); coefficient bins
shared via symcomp.encoders._q. Nonlinear mechanisms raise (unreachable in
the linear universe).

Normative examples (from the arm spec, reproduced exactly):
  advection:1.0 ->
    [R_EQ, R_SUM->PROD, R_CLASS_skew, COEF_1, DX_skew_1, R_U]
  advection:1.0 + diffusion:0.5 + dispersion:0.3 ->
    [R_EQ, R_SUM->SUM+PROD, R_SUM->SUM+PROD, R_SUM->PROD,
     R_CLASS_skew, COEF_1,    DX_skew_1, R_U,
     R_CLASS_sym,  COEF_0.5,  DX_sym_2, DX_sym_1, R_U,
     R_CLASS_skew, COEF_0.25, DX_skew_3, DX_skew_2, DX_skew_1, R_U]
    (19 tokens; 0.3 quantizes to the 0.25 bin)
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "typed_sketch_cd_cfg"


def _cls(order):
    if order == 0:
        return "src"
    return "skew" if order % 2 == 1 else "sym"


def encode(op):
    names = op.names()
    # plan-first spine: full arity sketch of the sum before any term body
    toks = ["R_EQ"] + ["R_SUM->SUM+PROD"] * (len(names) - 1) + ["R_SUM->PROD"]
    for n in names:
        if n not in PRIMITIVES:
            raise ValueError(f"typed_sketch_cd_cfg: nonlinear {n} unsupported")
        order = int(PRIMITIVES[n]["order"])
        c = _cls(order)
        toks.append(f"R_CLASS_{c}")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        toks.extend(f"DX_{c}_{k}" for k in range(order, 0, -1))
        toks.append("R_U")
    return toks
