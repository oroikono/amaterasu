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


class ARDecoder(nn.Module):
    """Autoregressive symbolic decoder for the DISCOVERY task (H3).

    Decodes the operator's token sequence in the REPRESENTATION'S OWN
    vocabulary, conditioned on the pooled data-branch embedding of the
    observed trajectory prefix. Two ids are reserved beyond the encoder
    vocab: BOS = vocab_size, EOS = vocab_size + 1 (internal to this head;
    the encoder vocab used for CONDITIONING is untouched, so prediction-side
    comparability with non-decoder stages is preserved).
    """

    def __init__(self, vocab_size, d_model, max_len, depth=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos = vocab_size
        self.eos = vocab_size + 1
        self.emb = nn.Embedding(vocab_size + 2, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, max_len + 1, d_model))
        layer = nn.TransformerDecoderLayer(d_model, nhead=4,
                                           dim_feedforward=4 * d_model,
                                           batch_first=True)
        self.tf = nn.TransformerDecoder(layer, depth)
        self.out = nn.Linear(d_model, vocab_size + 2)

    def forward(self, memory, tgt_ids):
        """Teacher-forced logits. memory (B,1,d); tgt_ids (B,L) encoder-vocab
        ids. Returns (B, L+1, V+2) predicting [tok_0..tok_{L-1}, EOS] from
        [BOS, tok_0..tok_{L-1}]."""
        B, L = tgt_ids.shape
        inp = torch.cat([torch.full((B, 1), self.bos, dtype=torch.long,
                                    device=tgt_ids.device), tgt_ids], dim=1)
        h = self.emb(inp) + self.pos[:, : L + 1]
        causal = torch.triu(torch.ones(L + 1, L + 1, device=tgt_ids.device,
                                       dtype=torch.bool), diagonal=1)
        h = self.tf(h, memory, tgt_mask=causal)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, memory, max_steps):
        """Greedy decode; (B, <=max_steps) ids, EOS-filled after stopping."""
        B = memory.shape[0]
        ids = torch.full((B, 1), self.bos, dtype=torch.long,
                         device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        outs = []
        for _ in range(max_steps):
            h = self.emb(ids) + self.pos[:, : ids.shape[1]]
            causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1],
                                           device=memory.device,
                                           dtype=torch.bool), diagonal=1)
            h = self.tf(h, memory, tgt_mask=causal)
            nxt = self.out(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            done = done | (nxt == self.eos)
            outs.append(nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            if bool(done.all()):
                break
        return torch.stack(outs, dim=1)


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
                 data_hidden_override=None, n_in_steps=1, use_ar_decoder=False,
                 ar_vocab_size=None):
        super().__init__()
        # n_in_steps: how many observed trajectory frames feed the data branch.
        # MUST be > 1 for the discovery head to be identifiable: a single IC
        # frame carries no information about the operator, so discovery from
        # n_in_steps=1 is structurally at chance.
        # width_mult lets the harness tune the data branch so EVERY arm matches
        # params (defense A1). Defaults: floor widens, coeff_vector also widens a
        # bit to compensate for its tiny (MLP) symbol branch.
        if width_mult is None:
            width_mult = {"none": 2, "coeff_vector": 1}.get(symbol_kind, 1)
        self.data_enc = DataEncoder(n_grid, d_model, n_in_steps=n_in_steps,
                                    depth=depth, width_mult=width_mult,
                                    hidden_override=data_hidden_override)
        self.sym_enc = SymbolEncoder(symbol_kind, vocab_size, d_model, n_mech,
                                     depth, max_len)
        self.fusion = Fusion(fusion if symbol_kind != "none" else "none", d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, T_out))
        self.discovery_mech = nn.Linear(d_model, n_discovery_mech)
        self.discovery_coef = nn.Linear(d_model, n_discovery_mech)
        # H3: AR decoder. Default: over the rep's OWN vocab (token arms only).
        # ar_vocab_size decouples the DECODE vocabulary from the INPUT rep
        # (e.g. condition on coeff_vector, name in the typed derivative CFG),
        # in which case any symbol_kind may carry a decoder.
        if ar_vocab_size is None and symbol_kind not in ("coeff_vector", "none"):
            ar_vocab_size = vocab_size
        self.ar_decoder = (ARDecoder(ar_vocab_size, d_model, max_len)
                           if use_ar_decoder and ar_vocab_size is not None
                           else None)
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

    def discover_ar(self, ic, tgt_ids=None, max_steps=None):
        """AR discovery (H3): teacher-forced logits when tgt_ids given, else
        greedy-decoded ids. Conditions ONLY on the data branch."""
        assert self.ar_decoder is not None
        memory = self.data_enc(ic).mean(1, keepdim=True)   # (B,1,d)
        if tgt_ids is not None:
            return self.ar_decoder(memory, tgt_ids)
        return self.ar_decoder.greedy(memory, max_steps)

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
