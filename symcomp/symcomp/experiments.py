"""experiments.py -- the four interventions that constitute the paper.

E1 Composition curve : zero-shot rel-L2 vs ||[A,B]||, per representation.
                       (the headline: gap to flat widens with commutator)
E2 Channel masking   : mask the symbol channel at test. If error barely moves,
                       the data did the work; if it explodes, symbols carry it.
E3 Counterfactual swap: feed symbol(A+B) but data(A). Does output move toward
                       the composite (symbol causal) or stay at A (symbol passive)?
E4 Embedding additivity: test emb(A+B) ~= emb(A) (+) emb(B) in the symbol latent,
                       and regress the residual against ||[A,B]||.
"""
from __future__ import annotations
import numpy as np
import torch
from .train import evaluate, OpDataset
from .encoders import enc_coeff_vector, encode_ids, build_vocab
from .operators import Operator


def E1_composition_curve(model, bench, encoder, vocab, max_len, device="cpu"):
    rows = evaluate(model, bench, bench.test_compose_idx, encoder, vocab, max_len, device)
    # aggregate by rounded commutator bin
    bins = {}
    for r in rows:
        key = round(r["commutator"], 3)
        bins.setdefault(key, []).append(r["rel_l2"])
    curve = sorted((k, float(np.mean(v)), float(np.std(v)), len(v)) for k, v in bins.items())
    return {"per_sample": rows, "curve": curve}


def E2_channel_masking(model, bench, encoder, vocab, max_len, device="cpu"):
    on = evaluate(model, bench, bench.test_compose_idx, encoder, vocab, max_len,
                  device, ablate_symbol=False)
    off = evaluate(model, bench, bench.test_compose_idx, encoder, vocab, max_len,
                   device, ablate_symbol=True)
    # symbol leverage = how much worse without the symbol, per stratum
    out = {}
    for st in set(r["stratum"] for r in on):
        e_on = np.mean([r["rel_l2"] for r in on if r["stratum"] == st])
        e_off = np.mean([r["rel_l2"] for r in off if r["stratum"] == st])
        out[st] = {"err_with_symbol": float(e_on),
                   "err_masked": float(e_off),
                   "symbol_leverage": float(e_off - e_on)}
    return out


@torch.no_grad()
def E3_counterfactual_swap(model, bench, encoder, vocab, max_len,
                           primitive_op: Operator, composite_op: Operator,
                           device="cpu", n=16):
    """Feed data from `primitive_op` but symbol from `composite_op`.
    Measure whether prediction moves toward composite ground truth.
    """
    from .solver import random_initial_condition, solve_constcoeff
    rng = np.random.default_rng(123)
    N, Ldom, t_eval = bench.N, bench.Ldom, bench.t_eval
    shifts = []
    model.eval()
    for _ in range(n):
        u0 = random_initial_condition(N, Ldom, rng=rng)
        y_prim = solve_constcoeff(primitive_op, u0, t_eval, N, Ldom)
        y_comp = solve_constcoeff(composite_op, u0, t_eval, N, Ldom)
        ic = torch.tensor(u0[:, None], dtype=torch.float32, device=device)[None]
        def sym_for(op):
            if encoder == "coeff_vector":
                return torch.tensor(enc_coeff_vector(op), device=device)[None], None
            ids, m = encode_ids(op, encoder, vocab[encoder], max_len)
            return (torch.tensor(ids, device=device)[None],
                    torch.tensor(m, device=device)[None])
        s_comp, m_comp = sym_for(composite_op)
        pred = model(ic, s_comp, m_comp).cpu().numpy()[0]
        # how far did pred move from primitive-truth toward composite-truth?
        d_to_prim = np.linalg.norm(pred - y_prim)
        d_to_comp = np.linalg.norm(pred - y_comp)
        sep = np.linalg.norm(y_comp - y_prim) + 1e-9
        # 1.0 => fully at composite (symbol fully causal); 0 => stayed at primitive
        shifts.append(float((d_to_prim) / (d_to_prim + d_to_comp)))
    return {"symbol_causal_fraction_mean": float(np.mean(shifts)),
            "symbol_causal_fraction_std": float(np.std(shifts))}


@torch.no_grad()
def E4_embedding_additivity(model, bench, encoder, vocab, max_len, device="cpu"):
    """emb(A+B) vs emb(A) (+) emb(B); regress residual on ||[A,B]||.

    Uses the trained symbol encoder's pooled embedding. The composition op (+)
    is fit as the best linear map (ridge) from [emb(A);emb(B)] -> emb(A+B),
    following the language-embedding compositionality methodology.
    """
    if encoder in ("coeff_vector", "none"):
        return {"applicable": False,
                "note": "additivity probe is for symbolic encoders"}

    # gather (A, B, A+B) triples from constant-coeff samples
    from .encoders import MECHANISMS
    triples = []
    seen = {}
    for s in bench.samples:
        seen[s.op.canonical_str()] = s.op
    ops = list(seen.values())
    singles = [o for o in ops if len(o.coeffs) == 1]
    for o in ops:
        if len(o.coeffs) == 2:
            names = o.names()
            a = next((x for x in singles if x.names() == (names[0],)
                      and abs(list(x.coeffs.values())[0] - o.coeffs[names[0]]) < 1e-6), None)
            b = next((x for x in singles if x.names() == (names[1],)
                      and abs(list(x.coeffs.values())[0] - o.coeffs[names[1]]) < 1e-6), None)
            if a is not None and b is not None:
                triples.append((a, b, o))

    def emb(op):
        ids, m = encode_ids(op, encoder, vocab[encoder], max_len)
        return model.symbol_embedding(
            torch.tensor(ids, device=device)[None],
            torch.tensor(m, device=device)[None]).cpu().numpy()[0]

    if len(triples) < 3:
        return {"applicable": True, "n_triples": len(triples),
                "note": "need more 2-term composites for a stable fit"}

    EA = np.stack([emb(a) for a, b, c in triples])
    EB = np.stack([emb(b) for a, b, c in triples])
    EC = np.stack([emb(c) for a, b, c in triples])
    X = np.concatenate([EA, EB], axis=1)
    # ridge fit  EC ~ X W
    lam = 1e-2
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ EC)
    pred = X @ W
    resid = np.linalg.norm(EC - pred, axis=1)
    # simple additive baseline (sum)
    add_resid = np.linalg.norm(EC - (EA + EB), axis=1)
    r2 = 1 - (np.sum((EC - pred) ** 2) / (np.sum((EC - EC.mean(0)) ** 2) + 1e-9))
    return {"applicable": True, "n_triples": len(triples),
            "ridge_R2": float(r2),
            "mean_ridge_residual": float(resid.mean()),
            "mean_additive_residual": float(add_resid.mean())}
