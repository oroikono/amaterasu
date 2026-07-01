"""model.py -- matched-capacity multimodal operator learner.

Predicts the trajectory u(.,t) from (initial condition data, symbolic operator).
Swappable along the two controlled axes:
  symbol channel : 'lample_charton' | 'prose_tree' | 'grammar' | 'coeff_vector'
                   | 'none' (data-only floor)
  fusion         : 'concat' | 'xattn' | 'film' | 'none'

Capacity is held ~constant across configs by fixing d_model / depth and (for the
data-only floor) widening the data branch to match parameter count.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class DataEncoder(nn.Module):
    """Encodes the IC (and a few input timesteps) into a token set over the grid.

    `hidden_override` lets the capacity harness set the branch width directly
    (fractional matching) instead of only the integer `width_mult`. Used to match
    coeff_vector / data_only arms to the symbolic arms within tolerance (A1).
    """
    def __init__(self, n_grid, d_model, n_in_steps=1, depth=2, width_mult=1,
                 hidden_override=None):
        super().__init__()
        h = hidden_override if hidden_override is not None else d_model * width_mult
        h = int(round(h / 4) * 4)  # keep divisible by nhead
        self.proj = nn.Linear(n_in_steps, h)
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_grid, h))
        layer = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=4 * h,
                                           batch_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(h, d_model)

    def forward(self, x):  # x: (B, N, n_in_steps)
        h = self.proj(x) + self.pos
        h = self.tf(h)
        return self.out(h)  # (B, N, d_model)


class SymbolEncoder(nn.Module):
    def __init__(self, kind, vocab_size, d_model, n_mech=5, depth=2, max_len=32):
        super().__init__()
        self.kind = kind
        if kind == "coeff_vector":
            self.mlp = nn.Sequential(nn.Linear(n_mech, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        elif kind == "none":
            self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.emb = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(0.02 * torch.randn(1, max_len, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead=4,
                                               dim_feedforward=4 * d_model,
                                               batch_first=True)
            self.tf = nn.TransformerEncoder(layer, depth)

    def forward(self, sym, mask=None):
        if self.kind == "coeff_vector":
            return self.mlp(sym).unsqueeze(1)            # (B,1,d)
        if self.kind == "none":
            return self.null.expand(sym.shape[0], -1, -1)  # (B,1,d) zeros
        h = self.emb(sym) + self.pos[:, : sym.shape[1]]
        kpm = (mask < 0.5) if mask is not None else None
        h = self.tf(h, src_key_padding_mask=kpm)
        return h                                          # (B, L, d)


class Fusion(nn.Module):
    def __init__(self, kind, d_model):
        super().__init__()
        self.kind = kind
        if kind == "xattn":
            self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.ln = nn.LayerNorm(d_model)
        elif kind == "film":
            self.to_gamma_beta = nn.Linear(d_model, 2 * d_model)
        elif kind == "concat":
            self.ln = nn.LayerNorm(d_model)

    def forward(self, data_tok, sym_tok):
        if self.kind == "none":
            return data_tok
        if self.kind == "concat":
            # broadcast-pool symbol, add to each grid token
            s = sym_tok.mean(1, keepdim=True)
            return self.ln(data_tok + s)
        if self.kind == "film":
            s = sym_tok.mean(1)
            gamma, beta = self.to_gamma_beta(s).chunk(2, dim=-1)
            return data_tok * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if self.kind == "xattn":
            a, _ = self.attn(data_tok, sym_tok, sym_tok)
            return self.ln(data_tok + a)


class OperatorLearner(nn.Module):
    """Maps (IC data, symbol) -> trajectory (prediction) and/or operator (discovery).

    Two heads:
      * prediction head: fused repr -> (T_out, N) rollout.
      * discovery head: data-only repr -> symbolic operator. For symbolic reps
        this is an autoregressive decoder over the rep's vocab (TODO: wire on
        Euler); a mechanism-multilabel + coeff-regression head is provided here
        as the matched, rep-agnostic discovery baseline that runs today.
    """
    def __init__(self, n_grid, T_out, vocab_size, d_model=128,
                 symbol_kind="grammar", fusion="xattn", n_mech=5,
                 depth=2, max_len=32, n_discovery_mech=8, width_mult=None,
                 data_hidden_override=None):
        super().__init__()
        # width_mult lets the harness tune the data branch so EVERY arm matches
        # params (defense A1). Defaults: floor widens, coeff_vector also widens a
        # bit to compensate for its tiny (MLP) symbol branch.
        if width_mult is None:
            width_mult = {"none": 2, "coeff_vector": 1}.get(symbol_kind, 1)
        self.data_enc = DataEncoder(n_grid, d_model, depth=depth, width_mult=width_mult,
                                    hidden_override=data_hidden_override)
        self.sym_enc = SymbolEncoder(symbol_kind, vocab_size, d_model, n_mech,
                                     depth, max_len)
        self.fusion = Fusion(fusion if symbol_kind != "none" else "none", d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, T_out))
        self.discovery_mech = nn.Linear(d_model, n_discovery_mech)
        self.discovery_coef = nn.Linear(d_model, n_discovery_mech)
        self.symbol_kind = symbol_kind

    def forward(self, ic, sym, sym_mask=None, ablate_symbol=False):
        d = self.data_enc(ic)
        if ablate_symbol or self.symbol_kind == "none":
            s = torch.zeros_like(d[:, :1, :])
            fused = self.fusion(d, s)
        else:
            s = self.sym_enc(sym, sym_mask)
            fused = self.fusion(d, s)
        return self.decoder(fused).transpose(1, 2)  # (B, T_out, N)

    def discover(self, ic):
        """Recover the operator from DATA alone. Returns (mech_logits, coefs)."""
        d = self.data_enc(ic).mean(1)        # pool grid
        return self.discovery_mech(d), self.discovery_coef(d)

    def symbol_embedding(self, sym, sym_mask=None):
        s = self.sym_enc(sym, sym_mask)
        return s.mean(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def assert_matched_capacity(models: dict, tol: float = 0.02):
    """Defense A1: assert all representation arms have params within `tol`.

    `models` maps rep_name -> model. Raises with a readable table if any arm is
    more than `tol` (fractional) away from the median param count.
    """
    counts = {k: count_params(v) for k, v in models.items()}
    med = sorted(counts.values())[len(counts) // 2]
    bad = {k: c for k, c in counts.items() if abs(c - med) / med > tol}
    report = "\n".join(f"  {k:18s} {c:,}  ({(c-med)/med:+.1%})"
                       for k, c in counts.items())
    if bad:
        raise AssertionError(
            f"Capacity mismatch > {tol:.0%} vs median {med:,}:\n{report}\n"
            f"Offenders: {list(bad)}. Adjust width_mult/depth to match.")
    return counts, report
