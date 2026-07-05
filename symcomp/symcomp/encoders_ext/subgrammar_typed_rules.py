"""subgrammar_typed_rules -- exploratory Stage AX arm: subgrammar factorization.

Hypothesis axis (user-flagged): instead of one monolithic TERM production
(`R_TERM->COEF_MECH` in the base `grammar` arm), factor the grammar into
separate production FAMILIES per mechanism CLASS, so the rule token itself is
typed by the physical subgrammar it belongs to:

    transport   : advection            (odd order 1, hyperbolic flavor)
    dissipation : diffusion, hyperdiffusion (even order, decaying)
    dispersion  : dispersion           (odd order 3, oscillatory)
    source      : reaction             (order 0, local)

Token scheme (normative example from the arm spec, followed EXACTLY):

    advection:1.0 ->
        [R_OP->TERM, R_transport::TERM->COEF_MECH, MECH_advection, COEF_1]

Each term contributes exactly FOUR tokens:
    1. spine rule       : "R_OP->TERM" for the first term, "R_OP->OP+TERM"
                          for every subsequent term (identical additive spine
                          and alphabetical `Operator.names()` term order as
                          `enc_grammar`, preserving minimal-pair validity);
    2. typed term rule  : "R_<class>::TERM->COEF_MECH";
    3. mechanism token  : "MECH_<name>";
    4. coefficient bin  : "COEF_<bin>" using the shared `_q` quantizer.

DOCUMENTED DEVIATION: one line of the spec estimated ~9 tokens for a 3-term
operator, but the normative per-term example implies 4 tokens/term, i.e.
3 terms -> 12 tokens. Per the spec's own instruction ("follow your exact
scheme, document"), this module follows the example: max 12 tokens for a
3-term operator (well under the 48-token budget). Expected corpus vocab on
the Stage A universe: 2 spine rules + 4 typed term rules + 5 MECH + 4 COEF
bins = 15 tokens.

Nonlinear S3 mechanisms (not in the Stage A universe) are mapped for forward
compatibility: burgers -> transport (nonlinear advection), cubic/quadratic ->
source. They contribute no vocab unless present in the corpus.

Pure, deterministic, PYTHONHASHSEED-stable: a fixed dict lookup plus the
shared deterministic `_q` binning; no hashing, randomness, or IO.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator

KEY = "subgrammar_typed_rules"

# mechanism -> subgrammar class (fixed, exhaustive over ALL_MECHANISMS)
_CLASS_OF: dict[str, str] = {
    "advection": "transport",
    "diffusion": "dissipation",
    "hyperdiffusion": "dissipation",
    "dispersion": "dispersion",
    "reaction": "source",
    # S3 forward-compatibility (outside the Stage A universe):
    "burgers": "transport",
    "cubic": "source",
    "quadratic": "source",
}


def encode(op: Operator) -> list[str]:
    """Typed-subgrammar derivation sequence for `op` (4 tokens per term)."""
    toks: list[str] = []
    for i, name in enumerate(op.names()):  # alphabetical, as in enc_grammar
        toks.append("R_OP->TERM" if i == 0 else "R_OP->OP+TERM")
        toks.append(f"R_{_CLASS_OF[name]}::TERM->COEF_MECH")
        toks.append(f"MECH_{name}")
        toks.append(f"COEF_{_q(op.coeffs[name])}")
    return toks
