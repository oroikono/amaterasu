"""slot_vector -- positional-binding/anonymization arm (discrete coeff_vector twin).

Mechanism identity is carried PURELY BY POSITION in a fixed 5-slot sequence
over the canonical MECHANISMS order [advection, diffusion, dispersion,
reaction, hyperdiffusion]. Slot k holds the token ``c_<_q(coeff)>`` if the
k-th mechanism is present in the operator, else ``c_0``. Exactly 5 tokens for
every operator; no mechanism-name tokens exist, so identity rides entirely on
the learned positional embedding.

Examples (normative, from the arm spec):
  advection:1.0                              -> [c_1, c_0, c_0, c_0, c_0]
  advection:1.0+diffusion:0.5+dispersion:0.3 -> [c_1, c_0.5, c_0.25, c_0, c_0]

Injectivity on the Stage-AX universe: each mechanism has a single fixed
coefficient (configs/default.yaml), so slot k is c_0 iff the mechanism is
absent and takes one fixed nonzero token otherwise; distinct operators
(distinct mechanism subsets) map to distinct 5-tuples. No spec deviations.
Operators containing mechanisms outside the 5 linear PRIMITIVES (e.g. the
nonlinear S3 rung) have no slot and raise ValueError rather than silently
encoding non-injectively.
"""
from __future__ import annotations

from ..encoders import MECHANISMS, _q
from ..operators import Operator

KEY = "slot_vector"


def encode(op: Operator) -> list[str]:
    """Exactly 5 tokens: slot k = k-th mechanism in MECHANISMS order."""
    unknown = set(op.coeffs) - set(MECHANISMS)
    if unknown:
        raise ValueError(f"slot_vector has no slot for mechanisms: {sorted(unknown)}")
    return [f"c_{_q(op.coeffs[m])}" if m in op.coeffs else "c_0"
            for m in MECHANISMS]
