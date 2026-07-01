"""test_physics.py -- validate the load-bearing physics claims."""
import numpy as np
from symcomp.operators import (Operator, variable_coeff_commutator_norm,
                               make_varcoeff_profile)
from symcomp.solver import (solve_constcoeff, random_initial_condition, grid)

N, Ldom = 128, 2 * np.pi
t_eval = np.linspace(0, 0.5, 8)
rng = np.random.default_rng(0)
u0 = random_initial_condition(N, Ldom, rng=rng)

# 1) COMMUTING IDENTITY: exp(t(A+B)) == exp(tA)exp(tB) for const-coeff.
adv = Operator({"advection": 1.0})
dif = Operator({"diffusion": 0.5})
advdif = Operator({"advection": 1.0, "diffusion": 0.5})

# solve full composite directly
y_full = solve_constcoeff(advdif, u0, t_eval, N, Ldom)
# solve by sequential application exp(tB)exp(tA) (split) -- should match exactly
_, xi = grid(N, Ldom)
u0h = np.fft.fft(u0)
Ahat = adv.fourier_symbol(xi); Bhat = dif.fourier_symbol(xi)
y_split = np.empty_like(y_full)
for i, t in enumerate(t_eval):
    y_split[i] = np.fft.ifft(np.exp(t * Bhat) * np.exp(t * Ahat) * u0h).real
err = np.max(np.abs(y_full - y_split)) / (np.max(np.abs(y_full)) + 1e-12)
print(f"[1] commuting split identity max rel err = {err:.2e}  (expect ~1e-13)")
assert err < 1e-10, "commuting identity broken!"

# 2) VARIABLE-COEFF COMMUTATOR grows monotonically with epsilon.
print("[2] ||[a(x)d_x, nu(x)d_xx]|| vs epsilon:")
prev = -1
for eps in (0.0, 0.15, 0.3, 0.5, 0.8):
    af = make_varcoeff_profile(N, Ldom, 1.0, eps, k=1)
    nf = make_varcoeff_profile(N, Ldom, 0.5, eps, k=2)
    c = variable_coeff_commutator_norm(af, nf, N, Ldom)
    print(f"    eps={eps:.2f}  ||[A,B]||={c:.4f}")
    assert c >= prev - 1e-9, "commutator not monotone in epsilon!"
    prev = c
print("    -> monotone increasing. commutator knob works.")
print("PHYSICS OK")
