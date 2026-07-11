"""run_task.py -- one SLURM array task = one (rep, split_seed, init_seed) cell.

Maps a flat --task_index to the config grid, trains BOTH heads (prediction +
discovery baseline), evaluates on the held-out compose/decompose sets across
the commutator strata, and appends rows to the durable registry
(run-local rows.csv + file-locked master CSV, schema registry.MASTER_SCHEMA).

IMPLEMENTED (was a spec stub). Pipeline per task:
  1. parse yaml config; mechanisms + coeffs from it.
  2. idx -> (rep, split_seed, init_seed) via the fixed GRID below.
  3. split manifest via symcomp.splits.make_split(split_seed). The S2 anchor
     composite (advection+diffusion) is FORCED into test_compose (matching the
     pre-registered S2 epsilon sweep, which only measures zero-shot
     composition if the anchor is held out); leakage is re-asserted after.
  4. load precomputed shards from --data_dir (gen_data.py wrote them);
     ICs [:n_ic_train] feed training, [n_ic_train:] feed eval (IC hygiene).
  5. resolve data_hidden_override via symcomp.capacity and assert ALL arms
     match params within config tolerance (defense A1).
  6. joint training: MSE on the trajectory rollout + discovery_weight * (BCE
     mechanism multilabel + MSE coefficient regression), AdamW, linear-warmup
     + cosine schedule, early stop on plateaued train loss.
  7. evaluate per held-out variant: rel_l2 (prediction; compose and decompose
     -- decompose rows use metric_name rel_l2_decompose) and mech_f1/coef_mae
     (discovery baseline, compose). exact_match is NOT emitted: it requires
     the autoregressive decoder, which is not implemented yet.
  8. rows + resolved config + git SHA + seeds + data-manifest hash -> registry;
     small artifacts copied to the home archive.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symcomp import registry, shards
from symcomp.capacity import resolve_hidden_overrides
from symcomp.dataset import Benchmark, Sample
from symcomp.encoders import MECHANISMS, build_vocab
from symcomp.model import OperatorLearner, assert_matched_capacity
from symcomp.operators import ALL_MECHANISMS as ALL_MECHANISMS_DISCOVERY
from symcomp.operators import Operator
from symcomp.splits import make_split, _assert_no_leakage
from symcomp.train import OpDataset, evaluate

SPLIT_SEEDS = [0, 1, 2, 3, 4]
INIT_SEEDS = [0, 1, 2]


def build_grid(reps):
    """Flat index -> (rep, split_seed, init_seed), rep-major.

    reps comes from config model.reps. APPEND-ONLY discipline: new arms go at
    the END of the list so existing task indices keep their meaning within a
    stage (Stage A = the first 6 reps = indices 0-89; Stage AX extensions
    start at 90).
    """
    return list(itertools.product(reps, SPLIT_SEEDS, INIT_SEEDS))


# ---------------------------------------------------------------------------
# benchmark assembly from shards
# ---------------------------------------------------------------------------
def build_split(cfg, split_seed):
    sp = cfg["split"]
    mechs = list(cfg["mechanisms"]["linear"])
    coeffs = {m: float(cfg["coeffs"][m]) for m in mechs}
    if sp.get("compose_transfer"):
        # Stage SYM ("symbols must make the difference"): train = all pure
        # laws + composites over DONOR mechanisms only (disjoint from the
        # anchor pair); test = the anchor composite. Combined with
        # n_in_steps=1 (IC-only input, operator-independent), the symbol
        # channel is the SOLE carrier of operator identity at test time —
        # composition must happen by algebra over symbols, not data.
        from symcomp.splits import enumerate_operators, SplitManifest
        import numpy as np
        rng = np.random.default_rng(split_seed)
        universe = enumerate_operators(mechs, coeffs, sp["max_terms"],
                                       sp["triples_sample"], rng)
        donors = {"dispersion", "reaction", "hyperdiffusion"}
        train = [o for o in universe
                 if len(o.coeffs) == 1 or set(o.names()) <= donors]
        anchor = Operator({"advection": coeffs["advection"],
                           "diffusion": coeffs["diffusion"]})
        man = SplitManifest(seed=split_seed, train=train,
                            test_compose=[anchor], test_decompose=[],
                            meta={"mode": "compose_transfer",
                                  "donors": sorted(donors)})
        return man, anchor
    if sp.get("decompose_anchor"):
        # ADEC mode ("train the mixture, find the pure laws"): the anchor
        # primitives' singletons are held out; every composite (incl. the
        # anchor) trains. No compose direction, no S2 variants -- the eval is
        # purely mixture -> pure-law zero-shot (H5 pulled forward). split_seed
        # only varies nothing here (deterministic split); init seeds carry
        # the variance.
        from symcomp.splits import enumerate_operators, SplitManifest
        import numpy as np
        rng = np.random.default_rng(split_seed)
        universe = enumerate_operators(mechs, coeffs, sp["max_terms"],
                                       sp["triples_sample"], rng)
        held = {"advection", "diffusion"}
        train = [o for o in universe
                 if len(o.coeffs) >= 2 or o.names()[0] not in held]
        dec = [o for o in universe
               if len(o.coeffs) == 1 and o.names()[0] in held]
        man = SplitManifest(seed=split_seed, train=train, test_compose=[],
                            test_decompose=dec,
                            meta={"mode": "decompose_anchor"})
        anchor = Operator({"advection": coeffs["advection"],
                           "diffusion": coeffs["diffusion"]})
        return man, anchor
    # anchor primitives are protected: their singletons must stay in train so
    # the forced-heldout anchor still satisfies the compose premise
    man = make_split(mechs, coeffs, split_seed, held_frac=sp["held_frac"],
                     max_terms=sp["max_terms"],
                     triples_sample=sp["triples_sample"],
                     protect=frozenset({"advection", "diffusion"}))
    # make_split can hold the same primitive out of two composites -> dedup so
    # no operator is double-weighted in the decompose aggregation
    seen: set = set()
    man.test_decompose = [o for o in man.test_decompose
                          if not (o.canonical_str() in seen
                                  or seen.add(o.canonical_str()))]
    # force the S2 anchor into the held-out compose set (see docstring step 3)
    anchor = Operator({"advection": coeffs["advection"],
                       "diffusion": coeffs["diffusion"]})
    ac = anchor.canonical_str()
    if ac not in {o.canonical_str() for o in man.test_compose}:
        man.train = [o for o in man.train if o.canonical_str() != ac]
        man.test_compose.append(anchor)
        man.meta["anchor_forced_heldout"] = True
        _assert_no_leakage(man)
    return man, anchor


def load_benchmark(cfg, man, anchor, data_dir):
    """Assemble a Benchmark from shards. Returns (bench, variants) where
    variants = [{label, stratum, commutator, direction, idxs}] for eval."""
    d = cfg["data"]
    n_tr, n_te = int(d["n_ic_train"]), int(d["n_ic_test"])
    noise = float(d["noise_levels"][0])   # Stage A trains/evals at clean data
    mani = shards.load_manifest(data_dir)
    t_eval = np.linspace(0.0, float(d["t_max"]), int(d["T"]))

    # completeness: refuse to run against a partial/stale data dir, otherwise
    # cells silently evaluate different variant sets and stop being comparable
    dec_anchor = (bool(cfg["split"].get("decompose_anchor"))
                  or bool(cfg["split"].get("compose_transfer")))
    need = {o.canonical_str() for o in
            man.train + man.test_compose + man.test_decompose}
    if not dec_anchor:
        need |= {f"{anchor.canonical_str()}@eps{float(e):g}"
                 for e in d["epsilons"] if float(e) > 0}
    missing = need - set(mani)
    assert not missing, f"data_dir manifest missing entries: {sorted(missing)}"
    if not dec_anchor:
        s2_have = {k for k, r in mani.items() if r["stratum"] == "S2"}
        s2_want = {k for k in need if "@eps" in k}
        assert s2_have == s2_want, \
            f"S2 variants on disk {sorted(s2_have)} != config {sorted(s2_want)}"

    samples, train_idx, test_c, test_d, variants = [], [], [], [], []

    def add(op, key, stratum, comm, role, rows):
        # shard-vs-config consistency (config drift would breach IC hygiene)
        with open(os.path.join(data_dir, shards.entry_dirname(key),
                               "sidecar.json")) as f:
            side = json.load(f)
        for field, want in (("n_ic_train", n_tr), ("n_ic_test", n_te),
                            ("N", int(d["N"])), ("T", int(d["T"])),
                            ("t_max", float(d["t_max"]))):
            assert side[field] == want, \
                f"{key}: shard {field}={side[field]} != config {want}"
        u0, traj = shards.load_shard(data_dir, key, noise)
        assert u0.shape[0] == n_tr + n_te, f"{key}: shard IC count mismatch"
        sl = slice(0, n_tr) if rows == "train" else slice(n_tr, None)
        idxs = []
        for i in range(*sl.indices(u0.shape[0])):
            samples.append(Sample(op, stratum, comm, traj[i], u0[i], role))
            idxs.append(len(samples) - 1)
        return idxs

    for op in man.train:
        train_idx += add(op, op.canonical_str(), "S1", 0.0, "train", "train")
    for op in man.test_compose:
        idxs = add(op, op.canonical_str(), "S1", 0.0, "composite", "eval")
        test_c += idxs
        variants.append({"label": op.canonical_str(), "stratum": "S1",
                         "commutator": 0.0, "direction": "compose",
                         "idxs": idxs, "op": op})
    for op in man.test_decompose:
        idxs = add(op, op.canonical_str(), "S1", 0.0, "primitive", "eval")
        test_d += idxs
        variants.append({"label": op.canonical_str(), "stratum": "S1",
                         "commutator": 0.0, "direction": "decompose",
                         "idxs": idxs, "op": op})
    # S2 epsilon sweep on the (held-out) anchor -- not applicable in
    # decompose_anchor mode, where the anchor composite is TRAINED
    for key, rec in sorted(mani.items()):
        if dec_anchor or rec["stratum"] != "S2":
            continue
        comm = float(rec["commutator"])
        idxs = add(anchor, key, "S2", comm, "composite", "eval")
        test_c += idxs
        variants.append({"label": key, "stratum": "S2", "commutator": comm,
                         "direction": "compose", "idxs": idxs, "op": anchor})

    bench = Benchmark(samples, t_eval, int(d["N"]), 2 * np.pi,
                      train_idx, test_c, test_d)
    return bench, variants


# ---------------------------------------------------------------------------
# joint two-head training
# ---------------------------------------------------------------------------
def split_rep(rep, use_ar):
    """'input@decode' pseudo-arms decouple the conditioning representation
    from the naming vocabulary (e.g. 'coeff_vector@deriv_typed_cfg': best
    prediction channel in, best naming grammar out). Returns
    (input_rep, decode_rep-or-None)."""
    if "@" in rep:
        in_rep, dec_rep = rep.split("@", 1)
        return in_rep, (dec_rep if use_ar else None)
    return rep, (rep if use_ar and rep not in ("coeff_vector", "none") else None)


def build_joint_tensors(bench, idxs, encoder, vocab, max_len, n_in,
                        decode_rep=None):
    """Precompute the whole training set as stacked tensors.

    Per-item Dataset.__getitem__ tensor construction costs ~seconds/step under
    some runtimes (measured 5.5 s/step vs 43 ms of actual GPU compute); the
    train set is only ~150 MB, so materializing it once removes the entire
    input pipeline from the step time.
    """
    from symcomp.encoders import encode_ids
    inner = OpDataset(bench, idxs, encoder, vocab, max_len, n_in=n_in)
    ics, syms, masks, ys, dsyms, dmasks = [], [], [], [], [], []
    mechs = torch.zeros(len(idxs), len(ALL_MECHANISMS_DISCOVERY))
    coefs = torch.zeros(len(idxs), len(ALL_MECHANISMS_DISCOVERY))
    for i in range(len(idxs)):
        ic, sym, mask, y, _, _ = inner[i]
        ics.append(ic); syms.append(sym); masks.append(mask); ys.append(y)
        op = bench.samples[idxs[i]].op
        if decode_rep is not None:
            di, dm = encode_ids(op, decode_rep, vocab[decode_rep], max_len)
            dsyms.append(torch.tensor(di)); dmasks.append(torch.tensor(dm))
        else:  # unused placeholders (keeps the batch tuple fixed-width)
            dsyms.append(torch.zeros(max_len, dtype=torch.long))
            dmasks.append(torch.zeros(max_len))
        for n, c in op.coeffs.items():
            j = ALL_MECHANISMS_DISCOVERY.index(n)
            mechs[i, j], coefs[i, j] = 1.0, float(c)
    return torch.utils.data.TensorDataset(
        torch.stack(ics), torch.stack(syms), torch.stack(masks),
        torch.stack(ys), mechs, coefs, torch.stack(dsyms), torch.stack(dmasks))


def train_joint(model, bench, encoder, vocab, cfg, init_seed, device,
                decode_rep=None):
    tr = cfg["train"]
    torch.manual_seed(init_seed)
    ds = build_joint_tensors(bench, bench.train_idx, encoder, vocab,
                             int(cfg["model"]["max_len"]),
                             int(cfg["model"].get("n_in_steps", 4)),
                             decode_rep=decode_rep)
    dl = torch.utils.data.DataLoader(ds, batch_size=int(tr["batch"]),
                                     shuffle=True, drop_last=False,
                                     pin_memory=(device == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                            weight_decay=float(tr["weight_decay"]))
    epochs = int(tr["epochs"])
    steps = max(1, epochs * len(dl))
    warm = max(1, int(float(tr["warmup_frac"]) * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    mse, bce = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
    w_disc = float(tr.get("discovery_weight", 0.5))
    w_ar = float(tr.get("ar_weight", 0.5))
    patience = int(tr["early_stop_patience"])
    best, bad, ep_done = float("inf"), 0, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for ic, sym, mask, y, mech, coef, dsym, dmask in dl:
            ic, sym, mask, y = (t.to(device) for t in (ic, sym, mask, y))
            mech, coef = mech.to(device), coef.to(device)
            opt.zero_grad()
            pred = model(ic, sym, mask)
            mlog, cpred = model.discover(ic)
            loss = (mse(pred, y)
                    + w_disc * (bce(mlog, mech) + mse(cpred, coef)))
            if model.ar_decoder is not None:
                # H3: teacher-forced CE over the DECODE representation's
                # token sequence (== the input rep unless decoupled via '@').
                tgt, tmask = ((dsym, dmask) if decode_rep is not None
                              else (sym, mask))
                tgt, tmask = tgt.to(device), tmask.to(device)
                logits = model.discover_ar(ic, tgt_ids=tgt)
                lens = tmask.sum(1).long()
                labels = torch.full((tgt.shape[0], tgt.shape[1] + 1), -100,
                                    dtype=torch.long, device=device)
                labels[:, : tgt.shape[1]] = torch.where(
                    tmask > 0.5, tgt, torch.full_like(tgt, -100))
                labels[torch.arange(len(lens)), lens] = model.ar_decoder.eos
                loss = loss + w_ar * torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        tot /= len(dl)
        ep_done = ep + 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} joint_loss={tot:.4e}", flush=True)
        if tot < best - 1e-5:
            best, bad = tot, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at ep{ep} (plateau {patience})", flush=True)
                break
    return ep_done, best


@torch.no_grad()
def e3_symbol_swap(model, bench, variants, anchor, rep, vocab, max_len, n_in,
                   cfg, device):
    """Counterfactual swap on the anchor's S1 eval samples (see caller)."""
    from symcomp.encoders import enc_coeff_vector, encode_ids
    from symcomp.solver import solve_constcoeff_batch
    v = next((v for v in variants if v["stratum"] == "S1"
              and v["op"].canonical_str() == anchor.canonical_str()), None)
    if v is None:
        return None
    samples = [bench.samples[i] for i in v["idxs"]]
    pure = Operator({"advection": anchor.coeffs["advection"]})
    U0 = np.stack([s.u0 for s in samples])
    y_prim = solve_constcoeff_batch(pure, U0, bench.t_eval, bench.N, bench.Ldom)
    y_comp = np.stack([s.traj for s in samples])
    ic = torch.tensor(np.stack([s.traj[:n_in].T for s in samples]),
                      dtype=torch.float32, device=device)
    if rep == "coeff_vector":
        sym = torch.tensor(np.stack([enc_coeff_vector(pure)] * len(samples)),
                           device=device)
        mask = None
    else:
        ids, m = encode_ids(pure, rep, vocab[rep], max_len)
        sym = torch.tensor(np.stack([ids] * len(samples)), device=device)
        mask = torch.tensor(np.stack([m] * len(samples)), device=device)
    pred = model(ic, sym, mask).cpu().numpy()
    d_prim = np.linalg.norm((pred - y_prim).reshape(len(samples), -1), axis=1)
    d_comp = np.linalg.norm((pred - y_comp).reshape(len(samples), -1), axis=1)
    return float(np.mean(d_comp / (d_comp + d_prim + 1e-12)))


@torch.no_grad()
def ar_exact_match(model, bench, idxs, rep, vocab, max_len, n_in, device):
    """H3 metric: greedy-decoded token sequence == the operator's canonical
    encoding, exactly (incl. stopping at the right length). All encoders are
    injective on the universe, so sequence match == operator match; a
    non-canonical decode that would parse to the right operator counts as a
    MISS (strict; noted in the memo)."""
    from symcomp.encoders import encode_ids
    model.eval()
    op = bench.samples[idxs[0]].op
    ids, m = encode_ids(op, rep, vocab[rep], max_len)
    true_len = int(m.sum())
    target = torch.tensor(ids[:true_len])
    ics = torch.tensor(np.stack([bench.samples[i].traj[:n_in].T for i in idxs]),
                       dtype=torch.float32).to(device)
    dec = model.discover_ar(ics, max_steps=true_len + 1).cpu()
    hits = 0
    for b in range(dec.shape[0]):
        seq = dec[b]
        stop = (seq == model.ar_decoder.eos).nonzero()
        L = int(stop[0]) if len(stop) else seq.shape[0]
        hits += int(L == true_len and bool((seq[:L] == target).all()))
    return hits / dec.shape[0]


@torch.no_grad()
def discovery_metrics(model, bench, idxs, device, n_in):
    """Baseline-head discovery metrics for one variant's eval samples.

    Input is the OBSERVED trajectory prefix (n_in frames) -- the IC alone is
    operator-independent, so discovery from it would be structurally at chance.
    """
    model.eval()
    ics = torch.tensor(np.stack([bench.samples[i].traj[:n_in].T
                                 for i in idxs]), dtype=torch.float32).to(device)
    mlog, cpred = model.discover(ics)
    probs = torch.sigmoid(mlog).cpu().numpy()
    cpred = cpred.cpu().numpy()
    op = bench.samples[idxs[0]].op
    true = np.zeros(len(ALL_MECHANISMS_DISCOVERY))
    coef = np.zeros(len(ALL_MECHANISMS_DISCOVERY))
    for n, c in op.coeffs.items():
        j = ALL_MECHANISMS_DISCOVERY.index(n)
        true[j], coef[j] = 1.0, float(c)
    pred = (probs > 0.5).astype(float)
    inter = (pred * true).sum(1)
    f1 = np.where(pred.sum(1) + true.sum() > 0,
                  2 * inter / (pred.sum(1) + true.sum() + 1e-12), 1.0)
    present = true > 0.5
    mae = np.abs(cpred[:, present] - coef[present]).mean(1)
    return float(f1.mean()), float(mae.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--task_index", type=int, required=True)
    ap.add_argument("--data_dir",
                    default=os.environ.get("SCRATCH", ".") + "/symcomp/data")
    ap.add_argument("--workdir", default=None,
                    help="durable output root; defaults to $SYMCOMP_WORK_DIR "
                         "via symcomp.registry (refuses scratch on clusters)")
    a = ap.parse_args()
    t_start = time.time()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    REPS = list(cfg["model"]["reps"])
    rep, split_seed, init_seed = build_grid(REPS)[a.task_index]
    cell = {"rep": rep, "split_seed": split_seed, "init_seed": init_seed,
            "stage": a.stage, "task_index": a.task_index}
    print("RESOLVED CELL:", json.dumps(cell), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split + shards -> benchmark
    man, anchor = build_split(cfg, split_seed)
    bench, variants = load_benchmark(cfg, man, anchor, a.data_dir)
    print(f"benchmark: {len(bench.samples)} samples, "
          f"{len(bench.train_idx)} train / {len(bench.test_compose_idx)} compose"
          f" / {len(bench.test_decompose_idx)} decompose; device={device}",
          flush=True)

    # vocab over the DETERMINISTIC universe (identical across cells)
    vocab = build_vocab([s.op for s in bench.samples])
    mcfg = cfg["model"]
    d_model, max_len = int(mcfg["d_model"]), int(mcfg["max_len"])
    n_in = int(mcfg.get("n_in_steps", 4))
    use_ar = bool(cfg["train"].get("ar_decoder", False))
    tol = float(cfg["train"]["match_capacity_tol"])
    in_rep, dec_rep = split_rep(rep, use_ar)
    vsizes = {r: max(len(vocab.get(split_rep(r, True)[0], {})), 4)
              for r in REPS}

    # capacity match (defense A1): resolve overrides, assert ALL arms at the
    # PRE-REGISTERED tolerance (a looser gate here would overstate the claim)
    base_reps = [r for r in REPS if "@" not in r]
    overrides, target, residuals = resolve_hidden_overrides(
        base_reps, bench.N, len(bench.t_eval), vsizes, d_model,
        len(MECHANISMS), max_len, tol=tol, n_in_steps=n_in,
        use_ar_decoder=use_ar)

    def build_any(r, seed=0, override="FROM_TABLE"):
        torch.manual_seed(seed)
        ir, dr = split_rep(r, use_ar)
        av = (max(len(vocab.get(dr, {})), 4) if (dr and "@" in r) else None)
        return OperatorLearner(
            bench.N, len(bench.t_eval), vsizes[r], d_model=d_model,
            symbol_kind=ir, fusion=mcfg["fusion"], n_mech=len(MECHANISMS),
            max_len=max_len, n_discovery_mech=len(ALL_MECHANISMS_DISCOVERY),
            data_hidden_override=(overrides[r] if override == "FROM_TABLE"
                                  else override),
            n_in_steps=n_in, use_ar_decoder=use_ar, ar_vocab_size=av)

    # decoupled '@' arms carry a decoder the generic resolver didn't price in;
    # binary-search their data-branch width against the same target
    from symcomp.model import count_params
    for r in [x for x in REPS if "@" in x]:
        lo, hi = 16.0, 1024.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if count_params(build_any(r, override=mid)) < target:
                lo = mid
            else:
                hi = mid
        overrides[r] = int(round(lo))

    def build(r, seed=0):
        return build_any(r, seed=seed)

    counts, report = assert_matched_capacity({r: build(r) for r in REPS},
                                             tol=tol)
    print("param counts (A1):\n" + report, flush=True)
    if residuals:
        print(f"capacity residuals > {tol:.0%}: "
              f"{ {k: f'{v:.1%}' for k, v in residuals.items()} }", flush=True)

    # register the run BEFORE training so a crashed task leaves a record
    run = registry.Run.create(
        {"config_path": a.config, "config": cfg, "cell": cell},
        cell_tag=f"cell{a.task_index:03d}", root=a.workdir, **cell,
        seeds={"split": split_seed, "init": init_seed},
        param_counts=counts,
        capacity_residuals={k: float(v) for k, v in residuals.items()},
        data_manifest_hash=shards.manifest_hash(a.data_dir),
        split_meta=man.meta, device=device)
    print(f"registered run {run.run_id} -> {run.dir}", flush=True)

    model = build(rep, seed=init_seed).to(device)
    epochs_run, final_loss = train_joint(model, bench, in_rep, vocab, cfg,
                                         init_seed, device,
                                         decode_rep=dec_rep)

    # evaluation -> rows
    base = {"stage": a.stage, "encoder": rep, "fusion": mcfg["fusion"],
            "backbone": mcfg["backbone"], "split_seed": split_seed,
            "init_seed": init_seed, "params": counts[rep]}
    rows = []
    for v in variants:
        ev = evaluate(model, bench, v["idxs"], in_rep, vocab, max_len,
                      device=device, n_in=n_in)
        rel = float(np.mean([r["rel_l2"] for r in ev]))
        metric = "rel_l2" if v["direction"] == "compose" else "rel_l2_decompose"
        if v["direction"] == "decompose" and cfg["split"].get("decompose_anchor"):
            # H5 asymmetry needs per-law rows (advection vs diffusion)
            metric = f"rel_l2_dec[{v['op'].names()[0]}]"
        rows.append({**base, "task": "prediction", "commutator": v["commutator"],
                     "metric_name": metric, "metric_value": rel})
        if (v["direction"] == "decompose" and model.ar_decoder is not None
                and cfg["split"].get("decompose_anchor")):
            em = ar_exact_match(model, bench, v["idxs"], dec_rep, vocab,
                                max_len, n_in, device)
            rows.append({**base, "task": "discovery", "commutator": 0.0,
                         "metric_name": f"exact_match_dec[{v['op'].names()[0]}]",
                         "metric_value": em})
        if v["direction"] == "compose":
            # E2 (channel masking): same eval with the symbol channel ablated.
            # leverage = rel_l2_masked - rel_l2 quantifies how much the model
            # actually RELIES on symbols (makes the H1 null non-vacuous).
            evm = evaluate(model, bench, v["idxs"], in_rep, vocab, max_len,
                           device=device, n_in=n_in, ablate_symbol=True)
            rows.append({**base, "task": "prediction",
                         "commutator": v["commutator"],
                         "metric_name": "rel_l2_masked",
                         "metric_value": float(np.mean([r["rel_l2"] for r in evm]))})
            f1, mae = discovery_metrics(model, bench, v["idxs"], device, n_in)
            rows.append({**base, "task": "discovery",
                         "commutator": v["commutator"],
                         "metric_name": "mech_f1", "metric_value": f1})
            rows.append({**base, "task": "discovery",
                         "commutator": v["commutator"],
                         "metric_name": "coef_mae", "metric_value": mae})
            if model.ar_decoder is not None:
                em = ar_exact_match(model, bench, v["idxs"], dec_rep, vocab,
                                    max_len, n_in, device)
                rows.append({**base, "task": "discovery",
                             "commutator": v["commutator"],
                             "metric_name": "exact_match",
                             "metric_value": em})
        print(f"  eval {v['label']:44s} [{v['stratum']}|{v['direction']}] "
              f"rel_l2={rel:.4f}", flush=True)

    # E3 (counterfactual symbol swap) on the anchor composite: observed frames
    # say advection+diffusion, the symbol claims PURE advection. Fraction of
    # the prediction's movement toward the pure solution = symbol causality
    # (1 = symbol fully steers the output, 0 = model follows the data only).
    if in_rep != "none":
        e3 = e3_symbol_swap(model, bench, variants, anchor, in_rep, vocab,
                            max_len, n_in, cfg, device)
        if e3 is not None:
            rows.append({**base, "task": "intervention", "commutator": 0.0,
                         "metric_name": "e3_symbol_causal", "metric_value": e3})
            print(f"  E3 symbol-causal fraction = {e3:.3f}", flush=True)
    run.append_rows(rows)

    # per-run extras: variant table + timing (not in the fixed schema)
    with open(os.path.join(run.dir, "variants.json"), "w") as f:
        json.dump([{k: v[k] for k in ("label", "stratum", "commutator",
                                      "direction")} for v in variants],
                  f, indent=2)
    with open(os.path.join(run.dir, "training.json"), "w") as f:
        json.dump({"epochs_run": epochs_run, "final_train_loss": final_loss,
                   "wall_seconds": time.time() - t_start}, f, indent=2)
    # checkpoint for post-hoc interventions (stays on work storage; the home
    # archive skips .pt by design)
    torch.save(model.state_dict(), os.path.join(run.dir, "model.pt"))
    run.archive_to_home()
    print(f"wrote {len(rows)} rows for run {run.run_id} "
          f"({time.time()-t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
