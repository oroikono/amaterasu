# CODEX_START

Paste the block below into your coding agent (Codex / Claude Code / etc.) as the
opening prompt. It assumes the agent has the SymComp repo and the four companion
docs (CONTEXT.md, PLAN.md, DECISIONS.md, TODO.md) in the working directory.

---

You are continuing an ML research engineering project called **SymComp**. Read
`CONTEXT.md`, `PLAN.md`, `DECISIONS.md`, `TODO.md`, and the repo's
`EXPERIMENT_PLAN.md` and `README.md` before acting. Do not re-derive decisions
already recorded; follow them.

**Project in one paragraph.** SymComp tests whether a grammar-structured symbolic
representation of a PDE operator, trained jointly with numerical solution data,
gives better zero-shot generalization to *held-out mechanism compositions* (train
pure advection + pure diffusion separately, predict/recover advection-diffusion)
than flat symbolic reps (Lample–Charton prefix, PROSE-style tree) or a
coefficient-vector control. The explanatory variable is the commutator `||[A,B]||`:
for commuting (constant-coefficient) operators the solution semigroup factorizes so
the additive generator the symbols encode is sufficient; for non-commuting ones,
BCH/commutator corrections the syntax can't encode appear, so zero-shot success
should degrade with `||[A,B]||`. Headline claim H1: grammar wins at matched
capacity. Killer control H4: a scrambled-grammar arm must NOT match real grammar
(proves it's compositional structure, not grammar-ness).

**Repo map.** `symcomp/`: `operators.py` (mechanisms + commutator), `solver.py`
(exact spectral for constant-coeff = machine-zero data; ETDRK4 for
variable-coeff/Burgers), `encoders.py` (grammar, prose_tree, lample_charton,
coeff_vector, + grammar_scrambled control), `splits.py` (combinatorial held-out
composition splits, both compose and decompose directions, with leakage
assertions), `dataset.py`, `model.py` (matched-capacity multimodal model;
prediction head + discovery head; swappable fusion xattn/FiLM),
`capacity.py` (param-count matching via searched data-branch hidden size),
`train.py`, `experiments.py` (E1 composition curve, E2 channel masking, E3
counterfactual swap, E4 embedding additivity). `scripts/`: `run_all.py` (local
staged sweep), `run_task.py` + `gen_data.py` (Euler SLURM array-task stubs WITH
detailed implementation specs in their docstrings), `aggregate.py` (paired
bootstrap CIs + commutator regression + money plot). `cluster/` SLURM templates.
`configs/default.yaml`. `tests/test_physics.py` (validates the commuting identity
to machine zero + commutator monotonicity), `tests/smoke.py` (full pipeline at
toy scale).

**Environments.** Local dev on an Alienware workstation (has GPU) and a MacBook
Air (CPU only — physics tests + editing, no real training). Production on the ETH
Euler cluster: SLURM array jobs, LMOD module stack, wandb offline + login-node
sync. CRITICAL storage rule: Euler personal scratch (`/cluster/scratch`) is purged
at 15 days and not backed up — results must go to group work/project storage
(`/cluster/work/<group>`) and home, never only scratch. Never hardcode secrets,
credentials, tokens, or private cluster paths; read paths from env vars / config.

**What I want you to validate / build next, in order:**
1. **Durable storage + run registry (top priority).** Repoint all run outputs from
   `$SCRATCH` to a configurable `WORK_DIR` (default `/cluster/work/<group>/symcomp`,
   taken from an env var, not hardcoded). Add a `registry.py` that writes each run
   to `runs/<run_id>/` (run_id = timestamp + git SHA + cell index) with resolved
   config, result CSV rows, and `manifest.json` (seeds, data-file hashes, param
   counts), plus a helper to fetch a run by ID and an append-only, file-locked
   master CSV. Add an end-of-job copy of small artifacts to home.
2. **Implement `scripts/gen_data.py`** to the spec in its docstring: enumerate the
   operator universe, partition across a SLURM array, solve and persist per-operator
   trajectory shards with analytic `||[A,B]||` sidecars + manifest. Reuse the
   validated solvers in `symcomp/solver.py`. Verify leakage hygiene.
3. **Implement `scripts/run_task.py`** to the spec: map flat SLURM index ->
   (rep, split_seed, init_seed); load shards; resolve matched-capacity hidden size
   via `symcomp.capacity`; train both heads; evaluate on held-out compose/decompose
   across strata; append rows to the durable master CSV with the fixed schema
   `stage,encoder,fusion,backbone,split_seed,init_seed,task,commutator,metric_name,metric_value,params`.
4. **Add the autoregressive discovery decoder** + discovery metrics (canonical
   exact-match, mechanism F1, coefficient MAE) for the symbolic arms.
5. Keep `tests/test_physics.py` and `tests/smoke.py` green after every change; add
   tests for the new storage/registry and data/run-task code.

**Working agreements.** Enforce matched capacity (assert param counts within
tolerance; the data-only and coeff-vector arms widen their data branch). Every run
must be reproducible (dump resolved config + git SHA + seed + data-manifest hash).
Stage A (6 reps × 5 split-seeds × 3 init-seeds × 2 tasks) is the minimum
publishable core and the freeze point — get it runnable and durable before
building Stages B/C. Prefer small, tested increments; run the physics + smoke
tests before declaring anything done. Ask me only if a decision conflicts with
DECISIONS.md or an open question there blocks you; otherwise proceed and note the
assumption.

Start by reading the docs, then propose a short ordered task list for item (1)
and wait for my go-ahead before large changes.

---
