"""deriv_typed_cfg -- derivative-level CFG whose production families mirror
the OPERATOR ALGEBRA's grading (the "subgrammars per subalgebra" arm).

Constant-coefficient 1D operators decompose as L = D + A + S:
  D  even-order derivatives  (self-adjoint / dissipative subalgebra):
     diffusion (dx^2), hyperdiffusion (dx^4)
  A  odd-order derivatives   (skew-adjoint / conservative-dispersive):
     advection (dx), dispersion (dx^3)
  S  order zero              (scalar/source): reaction

Each term derives inside its class's subgrammar, and the derivative tower
uses CLASS-TYPED dx terminals, so the token stream carries the algebraic
type of every derivative application:

    EQ    -> u_t = SUM                       R_EQ
    SUM   -> PROD | SUM + PROD               R_SUM->PROD / R_SUM->SUM+PROD
    PROD  -> CLASS_sym | CLASS_skew | CLASS_src   R_CLASS_<c>
    CLASS_c -> COEF DERIV_c                  (implicit in R_CLASS_<c>)
    DERIV_c -> dx_c DERIV_c | u              DX_<c> ... R_U

advection:1.0 -> [R_EQ, R_SUM->PROD, R_CLASS_skew, COEF_1, DX_skew, R_U]
advection+diffusion(0.5) -> [..., R_SUM->SUM+PROD, R_CLASS_sym, COEF_0.5,
                             DX_sym, DX_sym, R_U]

Compare against deriv_cfg (same structure, UNTYPED dx) to isolate the value
of algebra typing, and against deriv_infix (L&C-style flat vocabulary) to
isolate rule tokens vs plain math text. Coefficient bins shared (_q);
canonical term order; nonlinear mechanisms raise (unreachable here).
"""
from symcomp.encoders import _q
from symcomp.operators import PRIMITIVES

KEY = "deriv_typed_cfg"


def _cls(order):
    if order == 0:
        return "src"
    return "skew" if order % 2 == 1 else "sym"


def encode(op):
    toks = ["R_EQ"]
    for i, n in enumerate(op.names()):
        if n not in PRIMITIVES:
            raise ValueError(f"deriv_typed_cfg: nonlinear {n} unsupported")
        order = int(PRIMITIVES[n]["order"])
        toks.append("R_SUM->PROD" if i == 0 else "R_SUM->SUM+PROD")
        toks.append(f"R_CLASS_{_cls(order)}")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
        toks.extend([f"DX_{_cls(order)}"] * order)
        toks.append("R_U")
    return toks
