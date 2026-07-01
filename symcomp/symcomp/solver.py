"""solver.py -- generate numerical trajectories for operators.

Constant-coefficient linear operators are solved EXACTLY in Fourier space:
    u_hat(t) = exp(t * L_hat(xi)) * u0_hat(xi)
No timestepping error -> the commuting-stratum data is ground-truth clean, which
is what makes the rung-1/2 result near-theorem-quality.

Variable-coefficient and nonlinear (Burgers) operators use ETDRK4
(exponential time differencing, Cox-Matthews) on the linear part + spectral
treatment of the nonlinearity. These are the strata where ||[A,B]|| > 0.
"""
from __future__ import annotations
import numpy as np
from .operators import Operator


def grid(N: int, Ldom: float):
    x = np.linspace(0.0, Ldom, N, endpoint=False)
    xi = 2.0 * np.pi * np.fft.fftfreq(N, d=Ldom / N)
    return x, xi


def random_initial_condition(N: int, Ldom: float, n_modes: int = 6,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Smooth random periodic IC: superposition of low Fourier modes."""
    rng = rng or np.random.default_rng()
    x = np.linspace(0.0, Ldom, N, endpoint=False)
    u = np.zeros(N)
    for k in range(1, n_modes + 1):
        a, b = rng.normal(), rng.normal()
        u += (a * np.cos(k * 2 * np.pi * x / Ldom) +
              b * np.sin(k * 2 * np.pi * x / Ldom)) / k
    return u / (np.std(u) + 1e-12)


def solve_constcoeff(op: Operator, u0: np.ndarray, t_eval: np.ndarray,
                     N: int, Ldom: float) -> np.ndarray:
    """EXACT solve for constant-coefficient linear op. Returns (T, N) array."""
    _, xi = grid(N, Ldom)
    Lhat = op.fourier_symbol(xi)
    u0h = np.fft.fft(u0)
    out = np.empty((len(t_eval), N))
    for i, t in enumerate(t_eval):
        out[i] = np.fft.ifft(np.exp(t * Lhat) * u0h).real
    return out


def solve_varcoeff_advdiff(a_field: np.ndarray, nu_field: np.ndarray,
                           u0: np.ndarray, t_eval: np.ndarray,
                           N: int, Ldom: float, n_sub: int = 2000) -> np.ndarray:
    """Variable-coefficient advection-diffusion via explicit spectral RK4.

    u_t = -a(x) u_x + nu(x) u_xx  (note: stable diffusion sign).
    Coefficient fields multiply in PHYSICAL space; derivatives in Fourier.
    """
    _, xi = grid(N, Ldom)
    ik = 1j * xi
    ik2 = (1j * xi) ** 2
    t0, t1 = float(t_eval[0]), float(t_eval[-1])
    dt = (t1 - t0) / n_sub

    def rhs(u):
        uh = np.fft.fft(u)
        ux = np.fft.ifft(ik * uh).real
        uxx = np.fft.ifft(ik2 * uh).real
        return -a_field * ux + nu_field * uxx

    # integrate, sampling at t_eval
    out = np.empty((len(t_eval), N))
    targets = list(t_eval)
    u = u0.copy()
    t = t0
    si = 0
    if abs(targets[0] - t0) < 1e-12:
        out[0] = u; si = 1
    for step in range(1, n_sub + 1):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t0 + step * dt
        while si < len(targets) and t + 1e-12 >= targets[si]:
            out[si] = u; si += 1
    while si < len(targets):
        out[si] = u; si += 1
    return out


def solve_burgers(nu: float, u0: np.ndarray, t_eval: np.ndarray,
                  N: int, Ldom: float, n_sub: int = 4000) -> np.ndarray:
    """Viscous Burgers  u_t = -u u_x + nu u_xx  via ETDRK4 (Kassam-Trefethen).

    The nu -> 0 singular limit (pure advection / shock formation) is the
    asymmetry test in the decompose direction.
    """
    _, xi = grid(N, Ldom)
    Llin = nu * (1j * xi) ** 2          # linear part multiplier (real, <=0)
    t0, t1 = float(t_eval[0]), float(t_eval[-1])
    h = (t1 - t0) / n_sub
    E = np.exp(h * Llin); E2 = np.exp(h * Llin / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * Llin[:, None] + r[None, :]
    Q  = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    ik = 1j * xi

    def Nl(vh):  # nonlinear term -(1/2) d_x (u^2)
        u = np.fft.ifft(vh).real
        return -0.5 * ik * np.fft.fft(u * u)

    v = np.fft.fft(u0)
    out = np.empty((len(t_eval), N)); targets = list(t_eval)
    si = 0; t = t0
    if abs(targets[0] - t0) < 1e-12:
        out[0] = np.fft.ifft(v).real; si = 1
    for step in range(1, n_sub + 1):
        Nv = Nl(v)
        a = E2 * v + Q * Nv
        Na = Nl(a)
        b = E2 * v + Q * Na
        Nb = Nl(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = Nl(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        t = t0 + step * h
        while si < len(targets) and t + 1e-12 >= targets[si]:
            out[si] = np.fft.ifft(v).real; si += 1
    while si < len(targets):
        out[si] = np.fft.ifft(v).real; si += 1
    return out
