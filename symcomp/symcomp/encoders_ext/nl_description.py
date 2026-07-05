"""nl_description -- natural-language description arm (exploratory, Stage AX).

Natural-language axis: encodes an operator the way a human would describe it,
as English word-tokens with function words. Tests whether NL-style tokens
(including constant "overhead" function words) behave differently from formal
symbolic tokens under held-out composition.

Token scheme (normative, per spec examples): terms in alphabetical mechanism
order (Operator.names()), joined by the word 'plus'; each term is
    [<mechname>, with, strength, <_q coefficient bin>]
Examples:
    advection:1.0                 -> [advection, with, strength, 1]
    advection:1.0 + diffusion:0.5 + dispersion:0.3 ->
        [advection, with, strength, 1, plus,
         diffusion, with, strength, 0.5, plus,
         dispersion, with, strength, 0.25]        (14 tokens)

Deviation from spec metadata (documented per instructions): the arm spec's
`max_tokens_3term` field said 16, but the normative token-scheme example
yields 4 tokens per term + 2 'plus' separators = 14 tokens for a 3-term
operator. The examples are normative, so we follow the scheme; the true
3-term length is 14 (well under the 48-token budget).

Function words ('with', 'strength', 'plus') are constant across operators --
pure overhead tokens carrying no operator information. That is deliberate:
it IS the axis under test.

Injectivity on the universe follows exactly as for enc_grammar: the encoding
is a bijective re-rendering of the (mechanism name, _q(coeff)) term sequence,
and _q does not collide distinct coefficients used in the Stage A universe.
Pure and deterministic: no hashing, no randomness, no file IO.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator

KEY = "nl_description"


def encode(op: Operator) -> list[str]:
    """Render op as an English description token sequence (see module doc)."""
    toks: list[str] = []
    for i, name in enumerate(op.names()):
        if i > 0:
            toks.append("plus")
        toks += [name, "with", "strength", _q(op.coeffs[name])]
    return toks
