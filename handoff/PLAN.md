# PLAN

## Proposed plan (summary)
Stage a pre-registered, capacity-matched comparison of symbolic representations on
zero-shot held-out mechanism composition, with the commutator as the explanatory
variable, run as SLURM array jobs on Euler, with durable result storage. Stage A
is the minimum publishable core; B and C strengthen and extend.

## Ordered phases

### Phase 0 — Local validation (Alienware / MacBook Air, days 0–2)
- `pip install -r requirements.txt`.
- Run `tests/test_physics.py`: confirm the commuting split identity holds to
  machine zero and the variable-coefficient commutator is monotone in epsilon.
- Run `tests/smoke.py`: confirm the full pipeline (build benchmark, train 2 reps,
  run E1–E4) executes end-to-end at toy scale.
- Skim `EXPERIMENT_PLAN.md` (the pre-registration + referee-defense doc).

### Phase 1 — Euler environment + durable storage (days 1–3)
- Create a venv on the cluster; pin `requirements.txt`; match torch CUDA build to
  the loaded CUDA module.
- **Set up purge-proof result storage** (see "needs validation"): point all run
  outputs at `/cluster/work/<group>/symcomp/...`, NOT scratch. Raw data shards may
  go to scratch (regenerable).
- Configure wandb offline + a login-node sync step.
- Re-run the physics validation on the cluster.

### Phase 2 — Data generation (days 2–4)
- Implement `scripts/gen_data.py` per its docstring spec: enumerate the operator
  universe (union over split seeds + S2 variable-coeff variants + S3 nonlinear),
  partition across a SLURM array, solve and write per-operator trajectory shards
  with analytic `||[A,B]||` sidecars and a manifest.
- Validate leakage hygiene and sample a few trajectories for sanity (energy decay
  for diffusion, translation for advection, shock formation for Burgers).

### Phase 3 — Stage A pipeline wiring (days 4–7)
- Implement `scripts/run_task.py` per spec: map flat SLURM index ->
  (rep, split_seed, init_seed); load shards; resolve matched-capacity hidden size;
  train BOTH heads (prediction = MSE rollout; discovery = mechanism multilabel +
  coefficient regression, with the autoregressive symbolic decoder as the stronger
  discovery variant to add); evaluate on held-out compose/decompose sets across
  strata; append rows to the durable master CSV with the fixed schema.
- Dry-run 1 seed for all 6 reps; assert matched param counts; confirm logging.

### Phase 4 — Full Stage A (days 8–14) — MINIMUM PUBLISHABLE CORE / FREEZE POINT
- Launch the 90-task array (6 reps × 5 split-seeds × 3 init-seeds), both tasks.
- Aggregate with `scripts/aggregate.py`: H1 paired-CI table (prediction +
  discovery), H2 commutator regression, H4 real-vs-scrambled panel, the money plot.
- **Freeze here.** Everything after is upside; this is already a defensible result.

### Phase 5 — Stage B robustness (days 15–21)
- Sweep fusion (xattn vs FiLM), backbone (transformer vs VAE — VAE arm to be
  implemented), model scale (d_model 128/256/512), and noise levels.
- Evaluate `ViT_FM` as an optional numerical-lane backbone candidate after Stage A
  is green. Keep it out of the Stage A critical path: first run the controlled
  small SymComp datasets with the simple baseline encoder, then test whether a
  stronger ViT/CFD trajectory encoder improves prediction or discovery without
  erasing the symbolic-representation effect.
- Confirm H1 survives every robustness axis.

### Phase 6 — Stage C extensions (days 22–27)
- Data-budget / sample-efficiency curve; the decompose direction + nu->0 singular
  limit (H5); the S3 Burgers / cubic rung; finalize the commutator regression.

### Phase 7 — Write-up (days 28–30)
- Regenerate all figures from the durable CSV; write the 2-page results memo
  mapping each panel to H1–H5 and the decision taken; archive a frozen copy of
  results + configs to project storage.

## What has already been decided
(See DECISIONS.md for rationale.)
- Headline = grammar beats other symbolic reps on zero-shot composition.
- Commutator is the explanatory mechanism, not a co-headline.
- Scope = 1D broad, 6–8 mechanisms, many held-out combos.
- Both prediction AND discovery tasks.
- Reimplement our 4 reps + scrambled control + data-only floor; cite external SOTA
  rather than re-racing their codebases.
- Pre-registration + the scrambled-grammar control are the two load-bearing
  defensibility moves.
- Stage A at day 14 is the freeze point / minimum publishable core.

## What still needs validation
1. **Durable storage wiring on Euler.** Confirm the group's `/cluster/work/<group>`
   path and quota (`lquota /cluster/work/<group>`); repoint all SLURM outputs
   there; add a run-registry + nightly copy of small artifacts to home; verify a
   run can be fetched back by ID after >15 days. THIS IS THE TOP VALIDATION ITEM
   — without it, early runs are deleted before the project ends.
2. **Module stack + CUDA torch** match on the current Euler default (the versions
   in `cluster/*.sh` are placeholders).
3. **`gen_data.py` and `run_task.py`** are specs, not finished code — implement and
   dry-run.
4. **Autoregressive discovery decoder** + discovery metrics (exact-match,
   mechanism F1, coefficient MAE) — currently only the rep-agnostic
   multilabel+regression discovery baseline runs.
5. **VAE backbone arm** for the Stage B robustness check.
6. **Last 2–3% capacity residual** on coeff_vector / data_only arms — close it
   (dummy adapter or FFN-ratio tweak) or report exact per-arm param counts in the
   paper table.
7. **Runtime/GPU-hour estimate** confirmed against a real single-run timing on
   Euler hardware, to size the array and the timeline buffer.
8. **Solver accuracy on S2/S3** (variable-coeff and Burgers) — spot-check ETDRK4
   against a refined reference; the S1 exact solver is already validated.
