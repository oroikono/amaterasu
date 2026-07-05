"""physics_typed_tags -- typed/physics-information exploratory arm (Stage AX).

Each term carries explicit PHYSICAL TYPE TAGS (dynamical class, derivative
order, parity) instead of an opaque mechanism ID. Tests whether exposing the
physics typing axis (dissipative / dispersive / conservative / source) helps
compositional generalization relative to mechanism-ID symbolic arms.

Token scheme (normative examples from the arm spec), per term in alphabetical
mechanism order (Operator.names()):

    [<class>, ord<order>, <parity>, coef, <_q bin>]

with
    class  in {dissip, disper, conserv, source}:
             diffusion, hyperdiffusion -> dissip; dispersion -> disper;
             advection -> conserv; reaction -> source
    parity in {even, odd}  (derivative order mod 2)

Examples (match spec exactly):
    advection:1.0
        -> [conserv, ord1, odd, coef, 1]
    advection:1.0 + diffusion:0.5 + dispersion:0.3
        -> [conserv, ord1, odd, coef, 1,
            dissip,  ord2, even, coef, 0.5,
            disper,  ord3, odd, coef, 0.25]      (15 tokens)

Deviation from spec header: the spec's `max_tokens_3term` field said 17, but
the scheme is exactly 5 tokens per term, so a 3-term operator is 15 tokens
(the spec's own normative 3-term example has 15 tokens and instructs to
recompute). Final count: 5 * n_terms, max 15 on the Stage A universe.

Injectivity: mechanism is recoverable from (class, ord<order>) -- the two
dissip mechanisms differ in order (diffusion=ord2, hyperdiffusion=ord4).
Asserted at import time over all linear PRIMITIVES. Terms are fixed-width
(5 tokens) so term boundaries are unambiguous without separators.

Pure deterministic function of the Operator: no hashing, no randomness,
no file IO; PYTHONHASHSEED-stable.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator, PRIMITIVES

KEY = "physics_typed_tags"

# mechanism -> dynamical class tag (linear constant-coefficient primitives)
_CLASS: dict[str, str] = {
    "advection": "conserv",
    "diffusion": "dissip",
    "dispersion": "disper",
    "reaction": "source",
    "hyperdiffusion": "dissip",
}

# (class, order) must uniquely identify the mechanism, else the arm is
# non-injective by construction. Guard at import time.
_tag_pairs = {n: (_CLASS[n], PRIMITIVES[n]["order"]) for n in PRIMITIVES}
assert len(set(_tag_pairs.values())) == len(_tag_pairs), (
    f"physics_typed_tags: (class, order) tags not injective: {_tag_pairs}"
)


def encode(op: Operator) -> list[str]:
    """Encode operator as per-term typed physics tags (5 tokens per term)."""
    toks: list[str] = []
    for n in op.names():
        if n not in _CLASS:
            raise KeyError(
                f"physics_typed_tags: no class tag for mechanism {n!r} "
                "(only linear constant-coefficient primitives supported)"
            )
        order = PRIMITIVES[n]["order"]
        parity = "even" if order % 2 == 0 else "odd"
        toks += [_CLASS[n], f"ord{order}", parity, "coef", _q(op.coeffs[n])]
    return toks
