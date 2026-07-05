# SESSION_LOG

Append-only, timestamped progress log. Purpose: if a session dies (usage cap,
crash) or another person/agent picks the project up, this file says where we
started, what happened since, and where things stand right now. Newest entries
at the bottom. Timestamps are local (with timezone).

Complements — does not replace — the handoff discipline in `AGENTS.md`
(update `TODO.md` / `DECISIONS.md` / `PLAN.md` at the end of meaningful work).
Log **during** work too, not only at the end.

Entry format:

```
## YYYY-MM-DD HH:MM TZ — <machine/agent> — <one-line summary>
- what was done / found / decided
- state right now (what's green, what's broken, what's mid-flight)
- next action if this session dies here
```

---

## 2026-07-01 (from git history) — baseline: repo created

Reconstructed from commits, for the record:
- `6fbad19` (00:59) SymComp AI handoff package (`handoff/` docs).
- `21954af` (01:07) Agent + Euler bootstrap docs (`AGENTS.md`, `docs/euler_pipeline.md`, `cluster/euler_gpu_smoke.sbatch`).
- `06eaf16` (06:10) SymComp experiment scaffold (`symcomp/`: ~1,600 lines — core
  modules, tests, configs, SLURM templates; `gen_data.py` / `run_task.py` are
  docstring-spec stubs).
- `e0c61c4`→`c066c59` (07:13) `external/ViT_FM` added **temporarily for
  inspection** (PR #1) — external ViT foundation-model codebase, not part of
  SymComp proper.

## 2026-07-04 02:20 CEST — Claude Code (Alienware/WSL2) — session start, handoff read

- Read all five `handoff/` docs + `AGENTS.md`. Working tree clean on `main`
  at `c066c59`. No work had been logged between 2026-07-01 and now.
- Project state per handoff: nothing implemented beyond the scaffold; top
  blocker is durable Euler storage (repoint `$SCRATCH` → `/cluster/work/<group>`,
  add `registry.py` run registry); then `gen_data.py`, then `run_task.py`;
  Stage A freeze target ≈ day 14 (~2026-07-14).
- Created this SESSION_LOG.md (user request: always keep a timestamped
  progress log for handoffs/crashes).
- Launched a multi-agent survey of the repo to verify actual implementation
  status vs. what the handoff claims, and to try running
  `tests/test_physics.py` + `tests/smoke.py` locally. Results to be appended
  below.
- Next action if session dies here: read the survey results (or re-survey),
  then start handoff TODO item #1 (durable storage + run registry).

## 2026-07-04 02:26 CEST — Claude Code (Alienware/WSL2) — repo survey done; tests PASS locally

**Verified state (6-agent survey of the whole repo + local test run):**

- **Tests are green on this machine** (miniconda base env: Python 3.13.5,
  numpy 2.3.2, torch 2.8.0+cu128; no project venv exists yet — none in repo).
  - `tests/test_physics.py`: PASS — commuting split identity max rel err
    **0.00e+00** (machine zero confirmed); commutator norm monotone in eps
    (0 → 331080 across eps 0→0.8). Numpy-only, runs in seconds.
  - `tests/smoke.py`: PASS — full pipeline (tiny S1S2 benchmark, grammar +
    lample_charton arms, E1–E4) end-to-end, exit 0, ~couple of minutes on CPU.
  - Phase 0 of PLAN.md (local validation) is therefore effectively **done**
    on Alienware as of this entry.
- **Core package `symcomp/symcomp/` (10 files, ~1,174 lines) is fully
  implemented** — no NotImplementedError anywhere; all 6 representation arms
  wired end-to-end (note: `data_only` is spelled `none` in code). Pure
  library: no file I/O, no env vars, no hardcoded paths. `registry.py` does
  NOT exist yet (as the handoff says).
- **Scripts:** `run_all.py` + `aggregate.py` are real working code (local toy
  pipeline). `gen_data.py` + `run_task.py` confirmed docstring-spec stubs —
  run_task resolves the 90-cell grid and writes meta JSON, but steps 3–8
  (load shards / train / eval / CSV rows) are a TODO block.
- **Cluster layer confirms blocker #1:** every output path is `$SCRATCH`
  (sbatch_data.sh:18, sbatch_stageA.sh:27, configs/default.yaml:59, even the
  venv). `SYMCOMP_WORK_DIR` is proposed in docs/euler_pipeline.md:89-97 but
  consumed by NOTHING (grep-verified). Module loads are placeholders.
- **`external/ViT_FM`** = Bogdan Raonic's ETH CAMLab ViT-foundation-model +
  GenCFD diffusion codebase, 24 MB / 236 files (13 MB is one notebook;
  GenCFD vendored twice). Useful as the reference for the real Euler module
  stack (`stack/2024-06 gcc/12.2.0 python_cuda eth_proxy`), sbatch
  generation, `/cluster/work` conventions, Poseidon-style dataloaders. No
  secrets found, but many hardcoded personal paths + a hardcoded wandb
  entity — must stay temporary.

**New issues found (added to TODO.md):**
1. [BUG] `encoders.py:127` `enc_grammar_scrambled` seeds its shuffle with
   Python `hash()` → salted per process → **H4 control arm is not
   reproducible across runs** unless PYTHONHASHSEED is pinned. Fix before
   any real run.
2. [BUG-RISK] `capacity.py` returns a `(value, "residual…")` tuple instead
   of an int when the capacity tolerance is exceeded — callers will crash.
3. sbatch scripts: no `set -euo pipefail` (failures can exit 0);
   `|| module load eth_proxy` fallback silently swallows a failed Python
   module load; eth_proxy never loaded in sbatch_data.sh; SLURM logs land in
   the submission cwd.
4. Test-coverage gaps: ETDRK4 solvers (S2/S3 data!) have no automated
   validation; smoke covers 2/6 arms, prediction head only; discovery
   metrics in aggregate.py have no producer yet.
5. `run_all.py` E3 wraps in bare try/except → silent NaN on failure.

**State right now:** working tree has this file + TODO.md edits only; no
code changed; nothing committed. Not yet on Euler (no venv, no storage
probe run).

**Next actions if this session dies here:** (1) commit+push the handoff
updates; (2) TODO #1 — implement `WORK_DIR`-based durable storage +
`registry.py` run registry (specs in CODEX_START item 1), fixing the
scrambled-grammar hash bug alongside; (3) then `gen_data.py` per its
docstring spec.

## 2026-07-04 02:56 CEST — Claude Code (Alienware/WSL2) — task list for TODO #1 proposed, waiting on user

- Proposed to the user the ordered implementation plan for TODO #1 (durable
  storage + run registry), all parameterized via `SYMCOMP_WORK_DIR` /
  `SYMCOMP_HOME_ARCHIVE` env vars so no Euler access is needed to write and
  test the code locally.
- Asked the user for: (a) the Euler storage-probe outputs (commands in
  `docs/euler_pipeline.md` "Durable Storage Probe": `my_share_info`,
  `lquota /cluster/work/<group>`, `lquota` home) to fill in the real group
  path + quota; (b) the current default Euler module stack (`module avail`)
  to replace the placeholder module loads; (c) go-ahead to implement; (d)
  whether to commit+push the pending handoff updates.
- Next action if this session dies here: same as previous entry — the plan
  is not blocked on Euler, only the final path value is.

## 2026-07-04 07:38 CEST — Claude Code (Alienware/WSL2) — TODO #1 implemented; mid-review checkpoint

- User gave go-ahead (env-var-parameterized, start before Euler probe) and
  approved commit+push; handoff updates pushed as `f3b664f`.
- **Implemented since then (working tree, uncommitted):**
  - `symcomp/symcomp/registry.py` (NEW): SYMCOMP_WORK_DIR/SYMCOMP_HOME_ARCHIVE
    resolution (refuses scratch-only cluster runs), `Run.create` →
    `runs/<run_id>/` with config.json + manifest.json, file-locked append-only
    master CSV with the fixed Stage-A schema, `get_run`/`list_runs`,
    `archive_to_home` (size-capped, skips checkpoints).
  - Fixed both survey bugs: `encoders.py` scrambled-grammar now seeds from
    sha256 (H4 control reproducible across processes); `capacity.py` both
    resolvers now return `(out, target, residuals)` — values always usable.
  - Wired: `run_task.py` registers runs via registry; `run_all.py --register`
    mirrors local runs into the registry; `configs/default.yaml` cluster
    section documents work_dir/home_archive; both sbatch scripts hardened
    (`set -euo pipefail`, unconditional eth_proxy, SYMCOMP_WORK_DIR guard,
    logs/ prefix + gitkeep).
  - `tests/test_registry.py` (NEW): 7 checks incl. 8-process concurrent
    append (200/200 rows intact) and cross-PYTHONHASHSEED scrambled
    determinism. **All green**; physics + smoke also re-run green.
- **Review state:** 4-lens adversarial review produced 22 raw findings, but
  the usage limit hit mid-verification (~03:15 CEST) — findings are
  UNVERIFIED. Session resumed 07:38 CEST; triaging them inline now. Key
  themes: fcntl.flock semantics on Lustre `/cluster/work` (unverifiable off-
  cluster → need per-run rows.csv as source of truth + master rebuild
  helper + on-cluster probe), no run_id column → requeue duplicates not
  dedupable, run_id 1-second collisions, non-atomic home-archive copy,
  venv itself lives on purgeable scratch, README/handoff staleness.
- Next action if this session dies here: apply the triaged fixes (run_id
  column + rebuild_master + collision suffix + atomic archive copy + venv
  relocation in sbatch + README/docs), rerun tests, update handoff docs,
  commit + push.

## 2026-07-04 07:44 CEST — Claude Code (Alienware/WSL2) — TODO #1 COMPLETE (code side); all tests green

- Applied all triaged review fixes:
  - registry.py: `run_id` provenance column auto-stamped on every CSV row
    (requeue dedup), `rebuild_master()` regenerates master.csv from per-run
    rows.csv (source of truth; flock-failure recovery path), same-second
    run-id collision suffix, torn-tail guard on locked appends, atomic
    (tmp+rename) home-archive copy, orphan-run-dir cleanup on failed create,
    local work-dir fallback anchored to repo root.
  - sbatch scripts: venv moved OFF scratch → `${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}`
    (purge would have killed it mid-project); WORK_DIR guard added to
    sbatch_data.sh too.
  - run_all.py `--register` now fails fast (resolves work dir before
    training); default.yaml comment no longer implies yaml env expansion;
    run_task.py TODO points at `registry.file_hashes` for shard provenance.
  - README cluster workflow rewritten (export SYMCOMP_WORK_DIR first, venv
    on work storage, aggregate from `$SYMCOMP_WORK_DIR/results/master.csv` —
    old scratch path was stale); docs/euler_pipeline.md gained a flock-probe
    section (`SYMCOMP_TEST_DIR=$SYMCOMP_WORK_DIR ... test_registry.py`).
- **Tests: registry 9/9, physics, smoke — ALL PASS** (WSL2, miniconda base).
- Handoff updated: TODO.md (survey bugs marked FIXED; storage item now
  "Euler validation only"), DECISIONS.md (+D11 storage design), PLAN.md
  (validation #1 rewritten).
- Committing + pushing this change set now.
- **Next actions for whoever picks this up:**
  1. On Euler: storage probe (`my_share_info`, `lquota`), export
     SYMCOMP_WORK_DIR/-HOME_ARCHIVE, venv on work storage, flock probe,
     re-run physics+registry tests on the cluster (TODO #1-3).
  2. Then implement `scripts/gen_data.py` per its docstring spec (TODO #4).
  3. Then `scripts/run_task.py` steps 3-8 (TODO #5).

## 2026-07-04 19:38 CEST — Claude Code (Alienware/WSL2) — night shift start; Euler SSH blocked, going local-first

- User on ETH VPN with an open Euler session; instructed: run the whole plan
  until usage limit, log lean + often, real large-scale runs only (no toy
  results passed off as real).
- SSH euler: publickey DENIED from WSL (both `ooikonomou` alias + `oroikon`);
  ControlMaster socket stale. Posted the WSL pubkey
  (~/.ssh/id_ed25519.pub, ed25519 ...VAzL oroikon@gmail.com) in chat for the
  user to append to Euler ~/.ssh/authorized_keys. Will retry periodically.
- Plan while blocked: implement gen_data.py + run_task.py (real trainer,
  both heads) locally, dry-run on the local GPU at reduced-but-real scale,
  review, push. If SSH opens: storage probe, work-dir venv, cluster tests,
  submit data + Stage A arrays with a background monitor.
- Next action if session dies: check `ssh euler hostname`; if OK follow
  TODO #1-3 cluster steps, else continue gen_data.py implementation.

## 2026-07-04 20:05 CEST — Claude (Alienware) — CRITICAL S2 bug found+fixed; solvers validated

- **Found a real pre-existing science bug** while validating solvers at
  production scale: S2 nu-profile used ABSOLUTE variation amplitude →
  nu(x)=0.5+eps·cos goes NEGATIVE for eps>0.5 → locally backward heat →
  ill-posed. Empirically: max|u| = 2.4e51 (eps=0.6), 1.1e204 (eps=0.8) at
  N=128. The toy benchmark's top S2 commutator bins (and smoke's
  relL2=1.000±0.000 top bin) were blow-up garbage.
- **Fix:** nu amplitude now RELATIVE (nu=0.5·(1+eps·cos), min>0 for eps<1)
  in dataset.py + tests. Commutator still strictly monotone (0→165697).
  Bounded solutions across the sweep (max|u|≈2.15). a-profile unchanged.
- Added batched solvers (solve_*_batch, exact match to singles) +
  `stable_n_sub` (explicit-RK4 stability at N=256 needs n_sub≈5349, not
  2000 — default would have blown up) + tests/test_solvers.py [6 checks,
  all green]. This closes the ETDRK4 test-gap TODO.
- physics + solvers green. Next: gen_data.py.

## 2026-07-04 20:55 CEST — Claude (Alienware) — gen_data + run_task IMPLEMENTED, dry-runs green, pushed 80a4589

- `symcomp/shards.py` (shard IO), `gen_data.py` (full impl: 29-entry
  universe = 25 S1 ops ∪ 4 S2 eps-variants; finite-check before write) and
  `run_task.py` (full Stage A cell: anchor forced held-out, IC-hygiene
  slices, capacity asserted across all 6 arms at 3.6M params ±1.6%, joint
  two-head training, per-variant eval rows → registry) all working.
- Dry-runs on cuda (grammar cell 0, coeff_vector cell 75, mini config):
  ~8s/cell, 35 rows each, S2 rel_l2 now meaningful (≈0.46-0.49, previously
  saturated at 1.000 due to the ill-posedness bug). Fixed duplicate
  decompose primitives (double-weighting).
- Euler SSH STILL DENIED (user hasn't added the key). Pushed 80a4589.
- Running now: production-config (N=256, 320 ICs) data gen locally for real
  timing + a full production cell for the GPU-hour estimate; background
  science-integrity review of the two new scripts.
- Next if session dies: check review findings + prod timings; retry
  `ssh euler`; on success → README cluster workflow steps 0-5.

## 2026-07-05 03:15 CEST — Claude (Alienware) — 6 review findings fixed (2 critical); tests green

- Science-integrity review returned 6 verified findings; all fixed:
  1. **[CRITICAL] Discovery head saw only the IC** (operator-independent →
     metrics structurally at chance, dead loss term). Fix: `n_in_steps: 4`
     observed trajectory frames now feed the data branch (model/train/
     capacity/run_task threaded; config documents it). Note: prediction task
     is now "4 observed frames + symbol → full trajectory".
  2. **[CRITICAL] Split premise violated in all 5 seeds** (decompose-held
     singletons overlapped compose-test primitives, incl. the anchor's).
     Fix: make_split picks ONE decompose primitive FIRST (never protected
     {advection,diffusion}), draws compose held-outs only from composites
     avoiding it; leakage assert #2 strengthened to singleton-in-train.
     Verified: premise holds seeds 0-4, decompose non-empty everywhere,
     universe unchanged (29 entries — existing shards stay valid).
  3. aggregate.py dedups requeued-task rows (latest run_id wins).
  4. run_task refuses partial/stale data dirs (manifest completeness).
  5. run_task asserts shard sidecar matches config (IC-hygiene guard).
  6. Capacity gate now exactly the pre-registered 2% (passes: max +1.6%).
- Concurrent edits appeared in model.py/train.py implementing the same
  n_in_steps fix (user or another agent) — merged cleanly, not reverted.
- **Perf**: 6.9h production cell was DataLoader overhead (GPU step = 43ms,
  total GPU compute ≈ 3 min!). Fixed: dataset precomputed into tensors.
- All tests green (physics, solvers, registry, smoke) + mini cell green.
- Production cell 0 re-running locally for the real timing.
- Next if dies: check prod_cell0.log timing; commit+push; ssh euler retry.
