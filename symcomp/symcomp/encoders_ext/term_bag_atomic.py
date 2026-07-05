"""term_bag_atomic -- holistic-lexicon endpoint of the structure-vs-bag axis.

One FUSED atomic token per term: the mechanism abbreviation and the binned
coefficient are welded into a single indivisible lexical item, e.g. "ADV@1".
The operator is rendered as a bag of these tokens in canonical alphabetical
order (Operator.names()), with ZERO intra-term syntax: no separators, no
coef/mech role markers, no expression-tree scaffolding.

Normative examples (from the arm spec):
    advection:1.0                              -> ["ADV@1"]
    advection:1.0+diffusion:0.5+dispersion:0.3 -> ["ADV@1", "DIF@0.5", "DSP@0.25"]

Coefficients are binned with the shared quantizer symcomp.encoders._q (8 bins
on [0.25, 2.0]); note _q(0.3) -> "0.25", hence DSP@0.25 above. Mechanism
abbreviations: ADV, DIF, DSP, REA, HYP for advection, diffusion, dispersion,
reaction, hyperdiffusion. Full token space: 5 mechanisms x 8 coefficient bins
= 40 tokens.

KNOWN LIMITATION (documented per spec "risks"): in Stage A every mechanism has
exactly ONE fixed coefficient (configs/default.yaml coeffs), so each mechanism
maps to a single fused token and the representation collapses to a
mechanism-ID bag -- only 5 of the 40 possible tokens are exercised, and the
coefficient half of the fused token carries no discriminative information.
The arm is still injective on the Stage A universe because operators there
differ exactly in their mechanism sets.

Purity: encode is a pure deterministic function of the Operator -- no hashing,
no randomness, no file IO; token order comes from Operator.names() (sorted),
so it is PYTHONHASHSEED-stable.

Nonlinear mechanisms (burgers, cubic, quadratic) are outside this arm's spec;
encountering one raises KeyError rather than inventing an unspecified token.
"""
from __future__ import annotations

from ..encoders import _q
from ..operators import Operator

KEY = "term_bag_atomic"

_ABBR = {
    "advection": "ADV",
    "diffusion": "DIF",
    "dispersion": "DSP",
    "reaction": "REA",
    "hyperdiffusion": "HYP",
}


def encode(op: Operator) -> list[str]:
    """Operator -> bag of fused mechanism@coeff tokens, alphabetical order."""
    return [f"{_ABBR[n]}@{_q(op.coeffs[n])}" for n in op.names()]
