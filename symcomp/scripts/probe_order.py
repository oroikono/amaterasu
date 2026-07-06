"""probe_order.py -- post-hoc symbol-algebra probes on saved checkpoints.

No retraining: loads every stage-AD/AX run's model.pt and evaluates two
"other side" questions on the held-out variants:

  P1 ORDER SWAP (commutativity of the symbol algebra): re-encode each
     held-out composite with REVERSED term order (dif+adv instead of the
     canonical adv+dif -- token vocabulary identical, order unseen in
     training) and measure prediction rel_l2 against the same targets.
     Rows: metric rel_l2_ordswap (paired with the run's original rel_l2).
     Order-free representations (coeff_vector, slot_vector, fourier_symbol)
     are invariant by construction and serve as the sanity anchor.
     NOTE grammar_scrambled: its shuffle is seeded from canonical_str(), so
     the "reversed" encoding is a wholly different sequence, not a mere
     reversal -- interpret its gap as encoding-perturbation, not order.

  P2 DECOMPOSE NAMING (stage AD checkpoints only): greedy-decode the pure
     decompose held-outs (laws seen ONLY inside trained composites, never
     alone) -- the naming analogue of H5. Rows: metric exact_match_decompose.

Emits rows under stage=ORD<src> (ORDAD / ORDAX) via the registry, one run
per probed cell, run_id provenance preserved.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symcomp import registry, shards
from symcomp.encoders import MECHANISMS, build_vocab, encode_ids
from symcomp.model import OperatorLearner
from symcomp.operators import ALL_MECHANISMS, Operator
from symcomp.splits import make_split


class ReversedOp(Operator):
    """Same operator, term iteration order reversed (anti-alphabetical).
    Every sequence encoder iterates op.names(), so this re-serializes any
    representation without touching encoder code."""

    def names(self):
        return tuple(sorted(self.coeffs, reverse=True))


def build_cell(cfg, split_seed, data_dir):
    sp = cfg["split"]
    mechs = list(cfg["mechanisms"]["linear"])
    co = {m: float(cfg["coeffs"][m]) for m in mechs}
    man = make_split(mechs, co, split_seed, held_frac=sp["held_frac"],
                     max_terms=sp["max_terms"],
                     triples_sample=sp["triples_sample"],
                     protect=frozenset({"advection", "diffusion"}))
    anchor = Operator({"advection": co["advection"],
                       "diffusion": co["diffusion"]})
    ac = anchor.canonical_str()
    if ac not in {o.canonical_str() for o in man.test_compose}:
        man.train = [o for o in man.train if o.canonical_str() != ac]
        man.test_compose.append(anchor)
    n_tr = int(cfg["data"]["n_ic_train"])
    vocab = build_vocab([o for o in man.train + man.test_compose
                         + man.test_decompose])
    return man, anchor, vocab, n_tr


def load_model(run_dir, rep, cfg, vocab, use_ar):
    mcfg = cfg["model"]
    vs = max(len(vocab.get(rep, {})), 4)
    from symcomp.capacity import resolve_hidden_overrides
    reps = list(mcfg["reps"])
    vsz = {r: max(len(vocab.get(r, {})), 4) for r in reps}
    overrides, _, _ = resolve_hidden_overrides(
        reps, int(cfg["data"]["N"]), int(cfg["data"]["T"]), vsz,
        int(mcfg["d_model"]), len(MECHANISMS), int(mcfg["max_len"]),
        tol=float(cfg["train"]["match_capacity_tol"]),
        n_in_steps=int(mcfg.get("n_in_steps", 4)), use_ar_decoder=use_ar)
    m = OperatorLearner(
        int(cfg["data"]["N"]), int(cfg["data"]["T"]), vs,
        d_model=int(mcfg["d_model"]), symbol_kind=rep, fusion=mcfg["fusion"],
        n_mech=len(MECHANISMS), max_len=int(mcfg["max_len"]),
        n_discovery_mech=len(ALL_MECHANISMS),
        data_hidden_override=overrides[rep],
        n_in_steps=int(mcfg.get("n_in_steps", 4)), use_ar_decoder=use_ar)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"),
                                 map_location="cpu"))
    return m


@torch.no_grad()
def eval_symbolled(model, rep, vocab, max_len, n_in, op_for_symbol, key,
                   data_dir, n_tr, device):
    """rel_l2 on an entry's eval ICs with an explicit symbol operator."""
    u0, traj = shards.load_shard(data_dir, key, 0.0)
    traj = traj[n_tr:]
    ic = torch.tensor(np.ascontiguousarray(traj[:, :n_in].transpose(0, 2, 1)),
                      dtype=torch.float32, device=device)
    y = torch.tensor(traj, dtype=torch.float32, device=device)
    if rep == "coeff_vector":
        from symcomp.encoders import enc_coeff_vector
        sym = torch.tensor(np.stack([enc_coeff_vector(op_for_symbol)] * len(y)),
                           device=device)
        mask = None
    else:
        ids, mk = encode_ids(op_for_symbol, rep, vocab[rep], max_len)
        sym = torch.tensor(np.stack([ids] * len(y)), device=device)
        mask = torch.tensor(np.stack([mk] * len(y)), device=device)
    pred = model(ic, sym, mask)
    num = torch.linalg.norm((pred - y).reshape(len(y), -1), dim=1)
    den = torch.linalg.norm(y.reshape(len(y), -1), dim=1) + 1e-9
    return float((num / den).mean())


@torch.no_grad()
def decode_exact(model, rep, vocab, max_len, n_in, op, key, data_dir, n_tr,
                 device):
    u0, traj = shards.load_shard(data_dir, key, 0.0)
    traj = traj[n_tr:]
    ids, mk = encode_ids(op, rep, vocab[rep], max_len)
    L = int(mk.sum())
    target = torch.tensor(ids[:L])
    ic = torch.tensor(np.ascontiguousarray(traj[:, :n_in].transpose(0, 2, 1)),
                      dtype=torch.float32, device=device)
    dec = model.discover_ar(ic, max_steps=L + 1).cpu()
    hits = 0
    for b in range(dec.shape[0]):
        seq = dec[b]
        stop = (seq == model.ar_decoder.eos).nonzero()
        Ld = int(stop[0]) if len(stop) else seq.shape[0]
        hits += int(Ld == L and bool((seq[:L] == target).all()))
    return hits / dec.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--data_dir",
                    default=os.environ.get("SCRATCH", ".") + "/symcomp/data")
    ap.add_argument("--stages", default="AD,AX")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = a.workdir or registry.work_dir()

    cfgs = {"AD": yaml.safe_load(open(os.environ.get("ORD_CFG_AD",
                                                     "configs/stageAD.yaml"))),
            "AX": yaml.safe_load(open(os.environ.get("ORD_CFG_AX",
                                                     "configs/stageAX.yaml")))}
    cells = {}   # (stage, rep, split, init) -> latest run dir
    for rid in registry.list_runs(root):
        rd = os.path.join(root, "runs", rid)
        mf = os.path.join(rd, "manifest.json")
        if not (os.path.exists(mf) and os.path.exists(os.path.join(rd, "model.pt"))):
            continue
        m = json.load(open(mf))
        if m.get("stage") not in a.stages.split(","):
            continue
        k = (m["stage"], m["rep"], int(m["split_seed"]), int(m["init_seed"]))
        if k not in cells or rid > cells[k][0]:
            cells[k] = (rid, rd)
    print(f"probing {len(cells)} checkpointed cells on {device}", flush=True)

    by_split = {}
    for (stage, rep, s, i), (rid, rd) in sorted(cells.items()):
        cfg = cfgs[stage]
        if (stage, s) not in by_split:
            by_split[(stage, s)] = build_cell(cfg, s, a.data_dir)
        man, anchor, vocab, n_tr = by_split[(stage, s)]
        use_ar = bool(cfg["train"].get("ar_decoder", False))
        try:
            model = load_model(rd, rep, cfg, vocab, use_ar).to(device).eval()
        except Exception as e:
            print(f"  SKIP {stage}/{rep}/s{s}i{i}: {e}", flush=True)
            continue
        run = registry.Run.create(
            {"probe": "order+decompose-naming", "source_run": rid},
            cell_tag=f"ord-{stage}-{rep}-s{s}i{i}", root=root,
            stage=f"ORD{stage}", rep=rep, split_seed=s, init_seed=i)
        base = {"stage": f"ORD{stage}", "encoder": rep,
                "fusion": cfg["model"]["fusion"],
                "backbone": cfg["model"]["backbone"], "split_seed": s,
                "init_seed": i, "task": "intervention", "params": 0}
        rows = []
        n_in = int(cfg["model"].get("n_in_steps", 4))
        ml = int(cfg["model"]["max_len"])
        if rep != "none":
            for op in [o for o in man.test_compose if len(o.coeffs) >= 2]:
                key = op.canonical_str()
                canon = eval_symbolled(model, rep, vocab, ml, n_in, op, key,
                                       a.data_dir, n_tr, device)
                swap = eval_symbolled(model, rep, vocab, ml, n_in,
                                      ReversedOp(op.coeffs), key,
                                      a.data_dir, n_tr, device)
                rows.append({**base, "commutator": 0.0,
                             "metric_name": "rel_l2_canon", "metric_value": canon})
                rows.append({**base, "commutator": 0.0,
                             "metric_name": "rel_l2_ordswap", "metric_value": swap})
        if use_ar and getattr(model, "ar_decoder", None) is not None:
            for op in man.test_decompose:
                em = decode_exact(model, rep, vocab, ml, n_in, op,
                                  op.canonical_str(), a.data_dir, n_tr, device)
                rows.append({**base, "task": "discovery", "commutator": 0.0,
                             "metric_name": "exact_match_decompose",
                             "metric_value": em})
        if rows:
            run.append_rows(rows)
        print(f"  {stage}/{rep:22s} s{s}i{i}: {len(rows)} rows", flush=True)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
