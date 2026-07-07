"""bpe_cfg -- corpus-induced-parsimony CONTROL: frozen BPE over deriv_cfg.

Pre-registered prediction: should NOT beat deriv_cfg. BPE merges are chosen
by corpus frequency, not constituent structure, so several merges cross
constituent boundaries (e.g. U_SUMP glues the END of one term to the START
of the next; DX_U_SUMP glues a derivative from term i to the SUM/PROD rules
of term i+1). If this arm matched deriv_cfg, "parsimony" rather than
"constituent alignment" would explain the grammar effect.

encode(op) = greedy BPE applied to deriv_cfg(op)'s token list under the
FROZEN 8-merge table below (hardcoded, never re-induced). Application
semantics: for each merge in rank order, apply greedy left-to-right passes
over the whole sequence until that rank reaches a fixpoint, then move to the
next rank.

FROZEN MERGE TABLE (rank order):
  1: (R_DX, R_DX)            -> DXDX
  2: (R_SUM->SUM+PROD, R_PROD) -> SUMP
  3: (R_U, SUMP)             -> U_SUMP
  4: (R_EQ, R_SUM->PROD)     -> EQP
  5: (EQP, R_PROD)           -> EQPP
  6: (COEF_0.25, DXDX)       -> C25DXDX
  7: (R_DX, U_SUMP)          -> DX_U_SUMP
  8: (C25DXDX, DXDX)         -> C25DX4

Normative examples (verified):
  advection:1.0 -> [EQPP, COEF_1, R_DX, R_U]
  advection:1.0+diffusion:0.5+dispersion:0.3 ->
    [EQPP, COEF_1, DX_U_SUMP, COEF_0.5, DXDX, U_SUMP, C25DXDX, R_DX, R_U]

Injectivity: every merged token has a unique expansion back to deriv_cfg
base tokens, so the map deriv_cfg(op) -> bpe_cfg(op) is invertible and the
arm inherits deriv_cfg's injectivity on the universe. Nonlinear mechanisms
raise (via deriv_cfg). No deviations from the spec were needed.
"""
from symcomp.encoders_ext.deriv_cfg import encode as _deriv_cfg_encode

KEY = "bpe_cfg"

# FROZEN -- do not re-induce. (pair, merged_token) in rank order.
MERGES: tuple[tuple[tuple[str, str], str], ...] = (
    (("R_DX", "R_DX"), "DXDX"),
    (("R_SUM->SUM+PROD", "R_PROD"), "SUMP"),
    (("R_U", "SUMP"), "U_SUMP"),
    (("R_EQ", "R_SUM->PROD"), "EQP"),
    (("EQP", "R_PROD"), "EQPP"),
    (("COEF_0.25", "DXDX"), "C25DXDX"),
    (("R_DX", "U_SUMP"), "DX_U_SUMP"),
    (("C25DXDX", "DXDX"), "C25DX4"),
)


def _merge_pass(toks: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    """One greedy left-to-right pass replacing adjacent `pair` with `merged`."""
    out: list[str] = []
    i = 0
    n = len(toks)
    while i < n:
        if i + 1 < n and toks[i] == pair[0] and toks[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def encode(op) -> list[str]:
    toks = list(_deriv_cfg_encode(op))
    for pair, merged in MERGES:
        while True:
            new = _merge_pass(toks, pair, merged)
            if new == toks:
                break
            toks = new
    return toks
