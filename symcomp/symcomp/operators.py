"""operators.py -- mechanism primitives, symbolic operators, and the commutator.

A PDE operator here is L = sum_k c_k * P_k, where each P_k is a *mechanism
primitive* (advection, diffusion, dispersion, reaction, hyperdiffusion). On a
1D periodic grid every constant-coefficient primitive is diagonal in Fourier
space, so the generator L acts as multiplication by a symbol L_hat(xi) and the
exact solution semigroup is u_hat(t) = exp(t * L_hat(xi)) * u0_hat(xi).

KEY FACTS this module encodes (the theory spine of the paper):
  * Constant-coefficient primitives all COMMUTE: [P_i, P_j] = 0. So for the
    commuting stratum exp(t(A+B)) = exp(tA) exp(tB) exactly, and composition is
    fully determined by the additive generator -- which is exactly what the
    symbol channel carries. ||[A,B]|| = 0 here.
  * VARIABLE coefficients break commutativity: [a(x) d_x, nu(x) d_xx] != 0, and
    the commutator magnitude grows with the spatial variation. This gives a
    *continuous knob* (epsilon) to sweep ||[A,B]|| while staying linear.
  * NONLINEAR advection (Burgers) is the strongly-non-commuting / singular rung.

The commutator is what we will correlate zero-shot compositional error against.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

# ----------------------------------------------------------------------------
# Mechanism primitives (constant-coefficient): name -> (order, fourier symbol)
# On a periodic domain, d/dx -> i*xi, so d^n/dx^n -> (i*xi)^n.
# We store the Fourier multiplier as a function of wavenumber xi.
# ----------------------------------------------------------------------------
PRIMITIVES: dict[str, dict] = {
    "advection":     {"order": 1, "mult": lambda xi: (1j * xi) ** 1, "sign": -1.0, "linear": True},
    "diffusion":     {"order": 2, "mult": lambda xi: (1j * xi) ** 2, "sign": +1.0, "linear": True},
    "dispersion":    {"order": 3, "mult": lambda xi: (1j * xi) ** 3, "sign": +1.0, "linear": True},
    "reaction":      {"order": 0, "mult": lambda xi: np.ones_like(xi, dtype=complex), "sign": +1.0, "linear": True},
    "hyperdiffusion":{"order": 4, "mult": lambda xi: (1j * xi) ** 4, "sign": -1.0, "linear": True},
}

# Nonlinear primitives (S3 stratum): NOT diagonal in Fourier space; handled by
# the time-stepping solver, not the exact spectral exponential. Registered here
# so the symbolic encoders and split generator can reason about them.
NONLINEAR_PRIMITIVES: dict[str, dict] = {
    "burgers":      {"linear": False, "desc": "u * u_x (nonlinear advection)"},
    "cubic":        {"linear": False, "desc": "u - u^3 (Fisher/Allen-Cahn source)"},
    "quadratic":    {"linear": False, "desc": "u^2 source"},
}
ALL_MECHANISMS = list(PRIMITIVES) + list(NONLINEAR_PRIMITIVES)
# 'sign' encodes the physically stable convention, e.g. diffusion uses +nu*d_xx
# (note (i*xi)^2 = -xi^2, so +nu*(i*xi)^2 = -nu*xi^2 decays -> stable). The
# advection sign is conventional; with real coefficient it stays skew (no
# growth). hyperdiffusion uses -gamma*d_xxxx -> -gamma*xi^4 (stable).


@dataclass(frozen=True)
class Operator:
    """A linear constant-coefficient operator L = sum_k coeffs[name_k] * P_k."""
    coeffs: dict[str, float]  # mechanism name -> real coefficient

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.coeffs))

    def fourier_symbol(self, xi: np.ndarray) -> np.ndarray:
        """Return L_hat(xi), the (complex) Fourier multiplier of the generator."""
        out = np.zeros_like(xi, dtype=complex)
        for name, c in self.coeffs.items():
            p = PRIMITIVES[name]
            out = out + c * p["sign"] * p["mult"](xi)
        return out

    def canonical_str(self) -> str:
        """Canonical (sorted) human form, e.g. 'advection:1.00+diffusion:0.50'."""
        return "+".join(f"{n}:{self.coeffs[n]:.4g}" for n in self.names())

    def add(self, other: "Operator") -> "Operator":
        merged = dict(self.coeffs)
        for n, c in other.coeffs.items():
            merged[n] = merged.get(n, 0.0) + c
        return Operator(merged)


def commutator_constcoeff(A: Operator, B: Operator) -> float:
    """[A, B] for constant-coefficient operators == 0 exactly (they all commute).

    Returned as a float (0.0) for API symmetry with the variable-coeff case.
    This is not a numerical approximation: constant-coefficient Fourier
    multipliers are scalars per mode, and scalars commute.
    """
    return 0.0


# ----------------------------------------------------------------------------
# Variable-coefficient commutator: the continuous knob.
# For A = a(x) d_x and B = nu(x) d_xx, the commutator [A,B] is itself a
# differential operator; we measure its operator 2-norm on the grid as a
# scalar ||[A,B]|| that grows with the coefficient variation epsilon.
# ----------------------------------------------------------------------------
def _spectral_derivative_matrix(N: int, L: float, order: int) -> np.ndarray:
    """Dense matrix of the periodic spectral derivative d^order/dx^order."""
    xi = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)  # wavenumbers
    F = np.fft.fft(np.eye(N), axis=0)
    mult = (1j * xi) ** order
    D = np.fft.ifft(mult[:, None] * F, axis=0)
    return D.real if order % 2 == 0 else D  # keep complex for odd orders


def variable_coeff_commutator_norm(
    a_field: np.ndarray, nu_field: np.ndarray, N: int, L: float
) -> float:
    """||[a(x) d_x, nu(x) d_xx]|| as a spectral operator 2-norm on the grid.

    a_field, nu_field are the (real) coefficient profiles sampled on the grid.
    """
    D1 = _spectral_derivative_matrix(N, L, 1)
    D2 = _spectral_derivative_matrix(N, L, 2)
    A = np.diag(a_field) @ D1
    B = np.diag(nu_field) @ D2
    C = A @ B - B @ A
    # operator 2-norm = largest singular value
    return float(np.linalg.svd(C, compute_uv=False)[0])


def make_varcoeff_profile(N: int, L: float, base: float, epsilon: float,
                          k: int = 1, rng: np.random.Generator | None = None
                          ) -> np.ndarray:
    """Coefficient profile  c(x) = base + epsilon * cos(k * 2pi x / L).

    epsilon = 0 -> constant coefficient (commuting limit). Increasing epsilon
    monotonically increases ||[A,B]||. This is the sweep variable.
    """
    x = np.linspace(0.0, L, N, endpoint=False)
    return base + epsilon * np.cos(k * 2.0 * np.pi * x / L)
