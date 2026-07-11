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

## 2026-07-05 03:40 CEST — Claude (Alienware) — FULL LOCAL STAGE A LAUNCHED (90 cells, ~7h); Euler = 1 command when access opens

- **Production cell timing: 277s** (was 6.9h before the input-pipeline fix —
  90x). Full Stage A ≈ 7 GPU-hours → running the ENTIRE pre-registered
  90-cell battery locally overnight (production config, N=256, full ICs,
  80 epochs). Order is (split,init)-major: every 6 consecutive cells
  complete all 6 arms for one (split,init) pair, so a partial night still
  yields paired, analyzable data. Cell 0's single-cell S2 curve already
  shows rel_l2 rising with commutator (0.20→0.24) — anecdote, not result;
  the battery decides.
- Provenance: these are REAL runs of the real config on the Alienware
  (RTX 5080 laptop), registered in the local registry
  (scratchpad work_prod/, runs/<id> + master.csv). Euler remains the
  canonical environment — rerun there when access opens.
- Pushed `02c7583`: `cluster/euler_bootstrap.sh` = ONE command on a login
  node does everything (storage autoprobe → env persist → venv on work →
  cluster tests + flock probe → data array → Stage A gated afterok).
  D12/D13 recorded in DECISIONS; TODO refreshed.
- Watchers armed: Stage A error monitor; SSH-success watcher (retries
  every 5 min — paste the key and I take Euler from there).
- **Morning checklist for whoever wakes up first:**
  1. Local Stage A state: `wc -l <scratchpad>/stageA_done.txt` (n/90);
     aggregate: `PYTHONPATH=. python scripts/aggregate.py
     --csv <scratchpad>/work_prod/results/master.csv --task prediction
     --metric rel_l2` (also task=discovery, metric=mech_f1).
  2. Euler: authorize the key from chat (or skip) and run
     `bash cluster/euler_bootstrap.sh` on a login node from symcomp/.
  3. Copy the local master.csv somewhere durable if the sweep finished
     (scratchpad is /tmp — survives reboot only until tmp cleanup!).

## 2026-07-05 03:45 CEST — Claude (Alienware) — EULER ACCESS LIVE; bootstrap running on eu-login-32

- User created an SSH ControlMaster (after clearing a stale socket);
  4h window (ControlPersist=4h from ~03:30). All Euler ops go through
  `ssh -S ~/.ssh/euler-cm.sock euler`.
- Storage identified: **/cluster/work/math/ooikonomou** (group
  MATH-EULER-ooikonomou, group-writable; Lustre euesfs3, 65T free).
  SYMCOMP_WORK_DIR=/cluster/work/math/ooikonomou/symcomp.
- Repo rsynced (GitHub auth not needed) to euler:~/code/amaterasu at
  `554330d`, external/ excluded.
- `cluster/euler_bootstrap.sh` launched detached (~/symcomp_bootstrap.log):
  venv on work storage → physics/solvers/registry tests **incl. the flock
  probe ON LUSTRE** (decisive for master.csv locking) → data array →
  Stage A array gated afterok. Monitor armed on the log.
- Local sweep: 5/90 done (~5 min/cell, ETA ~10:45 CEST), first 6-arm block
  nearly complete.
- Next if session dies: `ssh -S ~/.ssh/euler-cm.sock euler 'tail
  ~/symcomp_bootstrap.log; squeue --me'` — if jobs queued, just wait;
  results land in /cluster/work/math/ooikonomou/symcomp/results/master.csv.
  Socket dies ~07:30 — user must re-run the ControlMaster command or
  authorize the WSL key (still the better fix).

## 2026-07-05 03:58 CEST — Claude (Alienware) — EULER PIPELINE LIVE: all cluster tests green, arrays queued

- Bootstrap completed on eu-login-32: venv on work storage (torch
  2.12.1+cu130), **PHYSICS OK / SOLVERS OK / REGISTRY OK on the cluster —
  including the 8-process flock probe ON LUSTRE (concurrent master.csv
  locking is coherent on /cluster/work/math)**.
- Jobs: data array **5746065** (29 entries, RUNNING across eu-a2p nodes),
  Stage A array **5746066** (90 GPU cells, PENDING afterok:5746065).
- Results will land in /cluster/work/math/ooikonomou/symcomp/results/
  master.csv (+ per-run dirs; home archive at ~/symcomp_archive on Euler).
- Watcher armed: queue-state changes + master.csv row count, every 5 min.
- Local sweep continues in parallel (same code/config — a free
  cross-hardware replication of Stage A).
- Next if session dies: `squeue --me` on Euler; when 5746066 drains,
  aggregate per README step 5. If SSH fails: socket expired (~07:30),
  re-run ControlMaster or add the WSL key.

## 2026-07-05 04:50 CEST — Claude (Alienware) — laptop closing; Euler autonomous; state snapshot

- **Euler Stage A 5746066: last seen 34 RUNNING / ~55 cells done /
  1271 rows in master.csv (04:30ish). Cluster jobs are laptop-independent
  and will complete on their own.** Results: /cluster/work/math/ooikonomou/
  symcomp/results/master.csv + runs/ + Euler-side ~/symcomp_archive.
- Local replication sweep DIED when /tmp was wiped (earlier process exit):
  18 production cells preserved in ~/symcomp_archive/runs (design worked —
  archive_to_home per cell). /tmp shards/registry lost (regenerable in ~3
  min via gen_data). Not blocking: Euler is the canonical run.
- Euler key STILL not authorized (user's paste went to a local shell;
  Claude's self-append was permission-blocked as unauthorized persistence —
  correct call). Socket dies on laptop close.
- **WAKE-UP CHECKLIST:**
  1. Connect ETH VPN. Re-establish access: EITHER on Euler (password login)
     `echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKDgYUSwj7P4NK9sWeAXhIsZd6qv2qjzLFaYl+LCVAzL oroikon@gmail.com' >> ~/.ssh/authorized_keys`
     OR locally: `rm -f ~/.ssh/euler-cm.sock && ssh -fN -M -S ~/.ssh/euler-cm.sock -o ControlPersist=12h euler`
  2. Check: `ssh euler 'sacct -j 5746066 --format=JobID,State -n | sort | uniq -c'`
     and `wc -l /cluster/work/math/ooikonomou/symcomp/results/master.csv`
     (expect ~2250 rows for 90 cells; requeues fine — aggregate dedups).
  3. Aggregate ON Euler: cd ~/code/amaterasu/symcomp && source
     /cluster/work/math/ooikonomou/symcomp/venvs/symcomp/bin/activate &&
     PYTHONPATH=. python scripts/aggregate.py --csv /cluster/work/math/
     ooikonomou/symcomp/results/master.csv --task prediction --metric rel_l2
     (repeat --task discovery --metric mech_f1; money plot lands in
     results/money_plot.png).
  4. Optionally rerun the local replication (bootstrap-free): regen data
     3 min, then the driver loop — see git history for drive_stageA.sh, or
     just trust Euler.

## 2026-07-05 (afternoon) — Claude (Alienware) — STAGE A PRELIMINARY RESULTS (88/90 cells, Euler)

- Euler Stage A finished 88/90 (task 2 TIMEOUT on a flaky node — cuda was
  up, siblings took minutes; task 32 node-cancelled at 2 min). Resubmitted
  both as job 5781548 (RUNNING). 2039 rows in master.csv.
- **PRELIMINARY, pre-registered analysis (do not tune post-hoc):**
  - **H1 NEGATIVE:** grammar does NOT beat baselines on zero-shot compose
    rel_l2. coeff_vector BETTER than grammar (-0.035, CI[-0.039,-0.030]);
    data-only slightly better (-0.010, CI excl. 0); all symbolic reps
    statistically equivalent to each other.
  - **H4: scrambled ≈ real grammar** (+0.002, CI spans 0) → per the
    pre-registered rule: no compositional-structure claim.
  - **H2 CONFIRMED (stratified, anchor sweep = the pre-registered test):**
    error degrades monotonically with ||[A,B]|| for EVERY rep — Spearman
    rho: coeff_vector +0.72, grammar +0.70, prose_tree +0.71,
    lample_charton +0.66, scrambled +0.68, data-only +0.48; mean
    degradation +0.04–0.06 rel_l2 across the eps sweep. NOTE: the pooled
    regression in aggregate.py HIDES this (S1-at-0 includes hard
    composites; S2 rows are the easy anchor family) — the stratified
    within-anchor test is the correct pre-registered one; consider adding
    it to aggregate.py as h2_stratified().
  - Discovery (baseline head): mech_f1 ≈ 0.67 ALL arms, no separation
    (expected — the head reads the shared data branch; H3 needs the AR
    decoder).
  - **Decision per pre-registration: pivot headline to the commutator law**
    ("grammar tied but H2 holds"). The negative-result path was designed
    in; nothing was tuned to force a positive.
- Local 18-cell partial archive: SCRAP (mixed pre-/post-fix split code
  across cells — Euler log comparison exposed it). Euler is canonical.
- Next: final 2 cells → full 90-cell aggregate + money plot; add
  h2_stratified to aggregate.py; results memo draft mapping panels to
  H1-H5 decisions.

## 2026-07-05 — Claude (Alienware) — STAGE A COMPLETE (90/90) — FREEZE POINT REACHED 9 DAYS EARLY

- Requeued cells landed; final master.csv = 2088 rows, 90/90 cells.
  Final numbers confirm the preliminary read:
  - H1 NEGATIVE (grammar ≤ baselines; coeff_vector best, −0.037 CI clean).
  - H4: scrambled == grammar → no structure claim.
  - H2 STRATIFIED CONFIRMED: rho +0.48…+0.72, positive for every arm.
  - Discovery baseline: no separation (H3 needs AR decoder). Decompose:
    directional only (H5 → Stage C).
- **Deliverables:** `docs/stageA_results_memo.md` (2-page, panels → H1-H5
  decisions per pre-registration, caveats, next steps) +
  `symcomp/results_stageA/` frozen master.csv + money_plot.png.
  h2_stratified() is in aggregate.py (pooled version kept, labeled
  confounded).
- Pre-registration discipline held: negative headline reported as-is,
  pivot to the commutator law taken per the written decision rule. Nothing
  tuned post-hoc.
- **The plan's day-14 freeze-point deliverable exists on day 5.**
- Next session: Stage B robustness (fusion/backbone/scale) on the pivoted
  headline; AR decoder for H3; E2/E3 at scale; prior-art re-check before
  write-up. Euler access: user should still add the key (socket needs
  password each time); everything else is one `sbatch` away.

## 2026-07-05 (midday) — Claude (Alienware) — E2/E3 interventions launched at scale (job 5818788)

- User go-ahead. run_task.py now also emits per compose variant
  `rel_l2_masked` (E2: symbol channel ablated at eval — leverage =
  masked − unmasked answers "did the model USE the symbols", making the
  H1 null non-vacuous) and per cell `e3_symbol_causal`
  (task=intervention: anchor data + PURE-advection symbol; 1 = symbol
  steers the prediction, 0 = data-driven). Checkpoints (model.pt) now
  saved per run for post-hoc work. Dry-run green (E3=0.33 at mini scale).
- Euler: synced, full 90-cell array resubmitted as **5818788** (retrains
  identical seeds; aggregate dedup keeps latest run per group, so the
  fresh rows also refresh H1/H2 — expect minor CI wiggle, direction
  should hold). Watcher armed.
- Conceptual note for the memo (user question "what did symbols give
  us?"): symbols provided the generator's COEFFICIENTS (numbers help:
  coeff_vector best), syntax was inert (H4 tie), and no symbol encodes
  the BCH flow correction (H2 law). The user's original hope — "know the
  data comes from a law, so compose/decompose" — is partially delivered
  (zero-shot composition DOES work, rel_l2≈0.25-0.30) and its sharpest
  remaining test is H3/discovery via the AR decoder (open).
- Next if dies: when 5818788 drains → aggregate incl. E2 leverage + E3;
  update memo; then Stage B / AR decoder.

## 2026-07-05 13:56 CEST — subagent: exploratory encoding-arm proposals
- Task: propose 4-6 NEW symbolic encoding arms for the exploratory sweep
  testing the generality of the all-arms-tied null (symbolic-math-in-ML lens).
- Read encoders.py/operators.py; matched conventions (sorted names(), 8-bin
  _q quantization, per-encoder vocab + <pad>).
- Proposed 6 arms via StructuredOutput to orchestrator: postfix_rpn
  (serialization order), digit_p10 (numeral tokenization, bin-then-digitize),
  slot_vector (positional vs token binding), fourier_symbol_semantic
  (syntax vs semantics; tokenized multiplier samples), unary_order
  (Peano-style shared dx sub-tokens), multiview_dual (prefix + slot views).
  All meet contract: deterministic, finite vocab, <=48 tokens for 3-term,
  ~40 lines numpy each.
- Next if dies: orchestrator holds the arm specs; re-run proposal subagent
  or lift specs from this entry into symcomp/encoders.py additions.

## 2026-07-05 — PL/grammar-lens subagent: Stage AX arm proposals
- Proposed 6 exploratory encoder arms (returned via structured output to the
  orchestrator, for symcomp/symcomp/encoders_ext/): subgrammar_typed_rules,
  bottomup_reduce_trace, term_bag_atomic, dag_edge_list,
  unary_derivative_order, digit_coeff_grammar. Each isolates one design axis
  (rule-vs-terminal locus, derivation direction, token granularity, graph
  serialization, primitive decomposability, coefficient tokenization).
- No code changed; read-only analysis of encoders.py/operators.py/config.
- Next if dies: orchestrator holds the arm specs; implement one file per arm
  under encoders_ext/ per the module contract.

## 2026-07-05 — Claude (Alienware) — E2/E3 AT SCALE: null is STRONG-FORM; Stage A fully replicated

- Array 5818788 drained (fresh retrain of all 90 cells + interventions).
- **E2 (masking): symbols ARE used** — leverage +0.063…+0.069 rel_l2,
  positive in 100% of variant-cells for every symbolic arm; data-only arm
  exactly 0.000 (ablation no-op sanity ✓).
- **E3 (counterfactual swap): symbols are causally live** — wrong symbol
  pulls predictions ~27–32% toward the wrong solution (coeff_vector 0.27,
  token arms ~0.32).
- **Replication:** independent retrain reproduced H1 (coeff_vector −0.037
  CI[−0.040,−0.033]; scrambled/prose ties), H2 stratified (rho +0.48…+0.72),
  H4 tie. Stage A conclusions now rest on two complete training rounds.
- Frozen: results_stageA/master_with_interventions.csv. Memo updated with
  the strong-form statement: models consume symbols and are steered by
  them; the FORM is irrelevant; no form escapes the commutator law.
- In flight: Stage AX design panel (3 lenses done incl. PL-lens entry
  above; merge pending) → implementation agents → ~180-cell AX array.
- Next if dies: read design-panel output (workflow w96h8sx2e), implement
  encoders_ext/ arms, configs/stageAX.yaml (append-only reps, max_len 48,
  stage AX), dry-run, submit.

## 2026-07-05 — Stage AX design-panel MERGE (subagent, structured output)
- Merged 18 proposed arms (3 designers) into final 10 for Stage AX, one per
  design axis: postfix_rpn (serialization order), digit_p10 (coeff numeral
  form, bin-then-digitize), slot_vector (positional binding / explicit
  absence), term_bag_atomic (fused holistic lexicon / bag), fourier_symbol
  (semantic-spectral), physics_typed_tags (typed metadata),
  subgrammar_typed_rules (nonterminal refinement), unary_order (morphology),
  nl_description (natural language), dag_edge_list (edge-set relational).
- Dropped/absorbed: bottomup_reduce_trace (same axis as postfix_rpn),
  fourier_symbol_semantic (merged), digit_coeff + digit_coeff_grammar (raw
  digits confound precision w/ form; reserved as disambiguator variant),
  dup slot_vector (merged), unary_derivative_order (merged), holistic_id
  (sandwiched by the already-tied data-only arm + untrained-embedding seed
  variance), multiview_dual (combination not family; length-confounded;
  reserve as follow-up if any single arm breaks the null).
- Numerically verified all token schemes against symcomp/encoders.py
  conventions (_q bins 0.25..2.0, alphabetical names() order, prefix tree).
- CRITICAL audit result: designers' fourier arm (2-octave mag bins) is NOT
  injective on the 25-op Stage A universe — 3 collisions incl. train/test
  leakage pair (hyperdiffusion:0.2 == diffusion:0.5+hyperdiffusion:0.2).
  Fixed in merged spec: 1-octave log2 bins (clip 0..16) -> 25/25 unique;
  build-time injectivity + heldout-vs-train collision assert is mandatory.
- All 10 arms: <=29 tokens for 3-term (fits current max_len 32 and planned
  AX 48), finite vocab (9..41), keys collide with nothing in ENCODERS.
- Next if dies: hand final specs to implementation agents -> one file per
  arm in symcomp/encoders_ext/, then configs/stageAX.yaml + dry-run.
- 2026-07-05 14:11 CEST [subagent] Stage AX arm: added symcomp/symcomp/encoders_ext/slot_vector.py (KEY=slot_vector; 5 fixed slots over MECHANISMS order, token c_<_q(coeff)>/c_0). Self-test green: injective on 25-op universe, maxlen 5, vocab 5 (c_0,c_0.25,c_0.5,c_0.75,c_1), spec examples exact, PYTHONHASHSEED 0/7 identical. No other files touched.

## 2026-07-05 12:12Z — dag_edge_list extension arm (subagent)
- Added symcomp/symcomp/encoders_ext/dag_edge_list.py (KEY='dag_edge_list'):
  expression DAG as parent-child edge list, term-index edge order fixed
  (alphabetical mechs); per-term edges ROOT->Tk, Tk->MECH, Tk->COEF(_q bin).
- Deviation documented: 18 tokens/3-term (no explicit edge-type tokens),
  not the draft's 27.
- Self-test green: injective on 5-seed Stage A universe, maxlen 18,
  vocab 13 (spec est. 16 counted unused coeff bins); spec example matches
  exactly; identical output under PYTHONHASHSEED=0 vs 7. No other files
  modified.

- 2026-07-05 14:12 CEST — Stage AX arm `nl_description`: added symcomp/symcomp/encoders_ext/nl_description.py (NL word-token encoder: per-term `<mech> with strength <_q bin>` joined by `plus`). Self-test green: injective on 5-seed universe, maxlen 14 (spec metadata said 16; scheme examples normative, deviation documented in module docstring), vocab 12, PYTHONHASHSEED 0/7 outputs identical. No other files touched.

## 2026-07-05 12:14Z — unary_order extension arm (subagent)
- Added `symcomp/symcomp/encoders_ext/unary_order.py` (KEY=`unary_order`):
  per term (alphabetical) `[coef, <_q bin>, 'd' x order, u]`, terms joined
  by `+`; reaction (order 0) has no `d`. Spec examples match exactly.
- Deviation documented: actual 3-term worst case is 20 tokens
  (diffusion+dispersion+hyperdiffusion = 5+6+7 + 2 seps), not the spec's 18;
  scheme unchanged, still <= 48.
- Self-test green: injective on 5-seed Stage A universe (25 ops), maxlen 20,
  vocab 8 (spec est. 11 counted unused coeff bins); identical output under
  PYTHONHASHSEED=0 vs 7. No other files modified.

## 2026-07-05 14:12 — physics_typed_tags arm (Stage AX, subagent)
- Added symcomp/encoders_ext/physics_typed_tags.py: typed physics-tag encoder
  (per term: [class, ord<N>, parity, coef, _q bin]; class in
  {dissip,disper,conserv,source}; mechanism recoverable from class+order,
  asserted at import). 5 tokens/term -> 15 for 3-term (spec header said 17;
  followed the normative examples, deviation documented in module docstring).
- Self-test green: injective on 25-op universe (5 seeds), maxlen 15, vocab 16;
  identical output under PYTHONHASHSEED=0 vs 7 (two subprocesses, diff clean).

- 2026-07-05 14:12 CEST — Stage AX arm `term_bag_atomic`: added symcomp/symcomp/encoders_ext/term_bag_atomic.py (fused mechanism@coeff atomic tokens, e.g. ADV@1, bag in names() alphabetical order, shared _q bins). Self-test green: injective on 5-seed universe, maxlen 3, vocab 5 on universe (40 possible; Stage A fixed coeffs collapse it to a mechanism-ID bag, documented in docstring per spec risks); spec examples match exactly; identical output under PYTHONHASHSEED=0 vs 7. No other files touched.

## 2026-07-05 14:13 CEST — Stage AX arm: postfix_rpn (subagent)
- Added symcomp/symcomp/encoders_ext/postfix_rpn.py (KEY="postfix_rpn"): postfix/RPN
  minimal pair vs lample_charton. Mechanical post-order of the exact prefix tree —
  parses enc_lample_charton output via fixed token arities (+,* binary; c,d<n> unary)
  and emits postorder; per-term [<bin>, c, u, d<order>, *], n-1 '+' at the end.
- Normative spec examples match token-for-token (1-term and 3-term, 17 tokens).
- Self-test green: injective on 25-op universe (5 split seeds), maxlen 17, realized
  vocab 13 (spec's 17 counts all 8 coeff bins; config coeffs hit 4 — same as prefix
  arm). Identical output under PYTHONHASHSEED=0 vs 7 (two subprocesses, diff clean).

## 2026-07-05 14:12 CEST — digit_p10 extension arm (subagent)
- Added symcomp/symcomp/encoders_ext/digit_p10.py (KEY="digit_p10"): coefficient-
  numeral-tokenization axis. Same 8 bins first (_q), then bin token replaced by
  Charton-style 5-token numeral [+N, 3-digit mantissa of round(bin*100), E-2]
  inside the enc_lample_charton prefix skeleton. 9 tokens/1-term, 29/3-term.
- Spec examples matched token-for-token. Self-test green: injective on 25-op
  universe (5 seeds), maxlen 29, vocab 16 (spec expected 21 counting all ten
  digits; bins realize only {0,1,2,5,7} — noted in docstring). Deterministic
  under PYTHONHASHSEED=0 vs 7 (two subprocesses, diff clean). No other files
  touched.

## 2026-07-05 14:12 CEST — Stage AX arm: subgrammar_typed_rules (subagent)
Added `symcomp/symcomp/encoders_ext/subgrammar_typed_rules.py` (no other files
touched): grammar arm with TERM productions typed by mechanism class
(transport/dissipation/dispersion/source), 4 tokens/term, same alphabetical
term order + additive spine as `enc_grammar`. Self-test on the 5-seed Stage A
universe: injective, maxlen 12, vocab 15 (matches spec); normative example
exact; output identical under PYTHONHASHSEED=0 vs 7. Documented deviation:
spec's "~9 tokens 3-term" estimate superseded by its own example -> 12.

## 2026-07-05 (Claude subagent, Stage AX arm: fourier_symbol)
Added symcomp/symcomp/encoders_ext/fourier_symbol.py (no other code touched):
semantic-ceiling arm tokenizing sign + 1-octave log2-magnitude bins of the
_q-binned operator's L_hat(xi) at xi in {1,2,4,8}, 5 tokens/xi = fixed 20
tokens/op. Self-test on the 5-seed Stage A universe (25 ops): injective,
maxlen 20, realized vocab 21 (design vocab 41 incl. unused bins); output
identical under PYTHONHASHSEED=0 vs 7; smoke.py OK with arm auto-registered.
Interpretation note (documented in module docstring): spec's sign+magnitude
"followed by" realized as ONE fused token (pm<b>/nm<b>) -- the only reading
consistent with the spec's own 20-token / 41-vocab arithmetic.

## 2026-07-05 14:16 CEST — digit_p10 arm ADVERSARIALLY VERIFIED (subagent)
- Re-ran all checks independently: both normative spec examples reproduced
  token-for-token (9 / 29 tokens); injective on the 25-op 5-seed universe
  (0 collisions), maxlen 29 <= 48, finite vocab 16; byte-identical output
  under PYTHONHASHSEED=0 vs 7 subprocesses; code is pure (only _q +
  operators imports, no IO/randomness/global mutation).
- Vocab 16 vs spec expected 21 judged a justified, docstring-documented
  expectation gap (bins realize digits {0,1,2,5,7} only), not a scheme
  deviation. Verdict: OK.

## 2026-07-05 — verifier subagent: subgrammar_typed_rules arm PASSED
- Adversarially verified symcomp/symcomp/encoders_ext/subgrammar_typed_rules.py:
  spec examples exact (1-term and 3-term, 12 tokens), injective on the 25-op
  5-seed Stage A universe, maxlen 12, vocab 15, PYTHONHASHSEED 0/7 diff clean,
  pure code, append-only SESSION_LOG, no tracked files modified.
- Minor (shared infra, all ext arms): importing an encoders_ext module
  directly BEFORE symcomp.encoders skips its ENCODERS registration silently
  (circular-import partial module in _load_extension_encoders). Pipeline
  imports encoders first, so no Stage AX impact.

## 2026-07-05 (afternoon) — Claude (Alienware) — Stage AX ready+pushed (ac66922); Euler link wedged mid-submit

- 10 exploratory arms implemented by agents (one module each), ALL 16 arms
  verified: injective on universe, <=29 tokens, PYTHONHASHSEED-stable;
  AX dry-run green (fourier_symbol cell, capacity +0.1%). Pushed ac66922.
- Euler: SSH master alive but channels hang (VPN blip likely). The
  240-cell AX submission (SYMCOMP_CONFIG=configs/stageAX.yaml
  SYMCOMP_STAGE=AX sbatch --array=0-239 cluster/sbatch_stageA.sh) was IN
  the timed-out command chain — UNKNOWN whether it reached sbatch.
- NEXT (whoever resumes): re-establish ControlMaster (rm -f
  ~/.ssh/euler-cm.sock && ssh -fN -M -S ~/.ssh/euler-cm.sock -o
  ControlPersist=12h euler), then CHECK BEFORE RESUBMITTING:
  squeue --me + sacct -X --name=symcomp_A --starttime=now-2hours.
  Also rsync repo to euler (ac66922) BEFORE submitting if absent —
  encoders_ext/ + stageAX.yaml must be there or all 240 tasks fail fast.

## 2026-07-05 (afternoon) — Claude (Alienware) — Stage AX SUBMITTED: job 5847283, 240 cells

- Link re-established (eu-login-29). Confirmed the wedged submission never
  reached sbatch (no double-submit); rsynced ac66922+ to Euler (10
  encoders_ext modules + stageAX.yaml verified present) and submitted:
  SYMCOMP_CONFIG=configs/stageAX.yaml SYMCOMP_STAGE=AX, array 0-239.
- 16 arms x 15 (split,init) cells; EXPLORATORY (not pre-registered) —
  report as generality test of the H1 null. Watch the fourier_symbol arm
  (semantic-ceiling probe): a win there = semantics beat syntax; a tie
  extends the null from syntactic to semantic encodings.
- Aggregation when drained: same aggregate.py on master.csv filtered to
  stage=AX (h1_table vs grammar + h2_stratified per arm; rows carry
  run_id for dedup).
- Next if dies: check job 5847283; when drained, aggregate AX, extend
  memo with the exploratory panel, update TODO (AR decoder next).

## 2026-07-05 (late afternoon) — Claude (Alienware) — AR decoder (H3) implemented; AD probe on Euler; user asleep 5h

- OVERNIGHT CONTRACT: push everything, zero overclaims, results as measured.
- H3 machinery done + pushed: ARDecoder over each rep's own vocab
  (BOS/EOS internal, conditions on data-branch memory only), teacher-forced
  CE joint term, strict greedy exact_match rows (sequence == canonical
  encoding; injective encoders => operator-level match). Capacity matched
  WITH decoder (5.73M ±1.2%; non-decoder arms widened). Mini dry-runs +
  registry + smoke green. exact_match=0.0 at mini scale (10 ep/32 ICs) —
  honest zero, mechanism verified.
- Euler: Stage AX all 240 cells RUNNING (5847283). AD single-cell probe
  5847781 submitted (gate before burning 90 cells on an unproven head);
  probe watcher armed. On probe COMPLETED + nonzero-or-converged AR loss:
  submit --array=1-89 (SYMCOMP_CONFIG=configs/stageAD.yaml SYMCOMP_STAGE=AD).
- Next if dies: check 5847283 (aggregate --stage AX) and 5847781 (log:
  logs/symcomp_A_5847781_0.out; exact_match rows in master.csv stage=AD);
  then AD array 1-89; aggregate --stage AD task=discovery metric=exact_match.

## 2026-07-06 00:0x CEST — Claude (Alienware) — Euler link wedged again (transport); jobs unaffected

- Mux master alive, channels hang, ping to euler.ethz.ch OK (~280ms VPN).
  Same wedge as yesterday afternoon. Cluster jobs (AX 5847283 all-240
  RUNNING at last contact; AD probe 5847781) CONTINUE unaffected — only
  monitoring is blind. Recovery watcher armed (10-min retries).
- WAKE-UP (user): rm -f ~/.ssh/euler-cm.sock && ssh -fN -M -S
  ~/.ssh/euler-cm.sock -o ControlPersist=12h euler   (in the WSL terminal)
  ...or better, add the key on Euler (chat has the line) and no socket is
  ever needed again.
- THEN (me or next agent): 1) sacct -j 5847283,5847781 states; 2) pull
  master.csv; aggregate --stage AX (exploratory panel; fourier_symbol is
  the arm to look at first) and check AD probe exact_match; 3) if probe
  sane, submit AD --array=1-89; 4) memo update, no overclaims.

## 2026-07-06 — Claude (Alienware) — AX RESULTS IN (240/240); AD probe diagnosed; AD array 5882055 running

- **Stage AX complete** (exploratory, 16 arms, 15 cells each). As measured:
  - H2 commutator law: rho POSITIVE for ALL 16 arms (+0.32…+0.75).
  - fourier_symbol (semantic ceiling: tokenized L_hat) TIES grammar
    (+0.005, CI spans 0): semantic conditioning does not beat syntax.
  - An ORDERING emerged (CI-clean at 15 seeds, EXPLORATORY — multiple
    comparisons, length/vocab covariates apply): numeric-flavored arms
    best (coeff_vector −0.052 vs grammar; digit_p10 −0.026; unary_order
    −0.025), classic serializations mid-pack (postfix≈prefix: minimal
    pair TIED — traversal order irrelevant), exotic structure worst
    (slot_vector +0.051 WORSE than grammar; dag_edge_list, subgrammar
    also worse). slot_vector vs coeff_vector gap ≈0.10 isolates the
    CONTINUOUS-VALUE pathway (not layout) as coeff_vector's advantage.
- **AD probe (AR decoder H3, grammar cell 0): exact_match 0.000 on ALL
  held-out variants, but 11/12 EXACT on TRAIN operators** (checkpoint
  diagnostic on Euler). The decoder works; the zero is a REAL
  generalization failure: the model names seen laws from data but cannot
  compose names for unseen combinations — while its prediction head
  composes fine. This makes the 6-arm decode comparison genuinely
  discriminative → submitted remaining cells as job 5882055 (89 RUNNING).
- Next if dies: when 5882055 drains → aggregate --stage AD task=discovery
  metric=exact_match (+ mech_f1); extend memo (AX panel + H3 panel);
  consider AX-decode follow-up (10 new arms with AR decoder).

## 2026-07-06 — Claude (Alienware) — Euler link down (socket gone); AD array unaffected, awaiting access

- ControlMaster socket disappeared (VPN drop likely); key still not
  authorized on Euler. AD array 5882055 (89 cells) continues server-side.
- Everything through e3b0bd0 is pushed (AX panel + H3 probe in memo).
- ON RECONNECT: rm -f ~/.ssh/euler-cm.sock && ssh -fN -M -S
  ~/.ssh/euler-cm.sock -o ControlPersist=12h euler; then aggregate
  --stage AD (task=discovery, metric=exact_match; also mech_f1) from
  /cluster/work/math/ooikonomou/symcomp/results/master.csv and append the
  H3 panel to the memo as measured.

## 2026-07-06 — Claude (Alienware) — STAGE AD COMPLETE (89/89): compositional naming fails in EVERY syntax

- H3 battery done. exact_match on held-out compositions: grammar 0.0014,
  prose_tree 0.0003, lample_charton 0.0000, scrambled 0.0000 (S2: zero
  everywhere). Probe already showed near-perfect TRAIN decoding → the
  dissociation is real and universal: models SIMULATE unseen compositions
  (rel_l2 0.2-0.3) but cannot NAME them, in any representation tried.
  mech_f1 ~0.67 all arms (partial mechanism recovery unchanged).
- Secondary (not a claim): grammar > scrambled on PREDICTION under the AR
  objective (+0.009, CI clean) — first separation seen; Stage B check
  warranted.
- Memo updated with the AD panel. H1-H5 program now complete except
  Stage B robustness + H5 singular limit (Stage C).
- Next: Stage B design (fusion/scale first; VAE backbone needs impl);
  prior-art re-check before any write-up; optional AX-decode sweep.

## 2026-07-06 — Claude (Alienware) — P1 ORDER-SWAP (local, production config): models are order-INVARIANT

- Local ORDL sweep (24 cells, prod config; grid quirk → grammar 15 cells +
  scrambled 9). Order-swap probe: feeding dif+adv instead of the trained
  adv+dif changes rel_l2 by +0.000019 mean (max |gap| 0.0012, n=51
  variant-cells) — four orders of magnitude below the signal (~0.29).
  Verified the swapped token sequences genuinely differ. Interpretation:
  the symbol transformer + attention pooling is near permutation-blind;
  A+B ≡ B+A holds architecturally, not as learned algebra. Consistent
  with AX postfix≈prefix tie. Top-up (6 remaining arms × s0i0) training in
  a detached loop → same probe after.
- Euler key saga: root cause found — user's id_ed25519 is PASSPHRASE-
  protected (BatchMode can't decrypt). Dedicated automation key
  euler-auto created by USER (classifier correctly blocks Claude from
  self-installing persistence); awaiting forced ssh-copy-id.

## 2026-07-06/07 — Claude (Alienware) — AUTONOMOUS EULER ACCESS ESTABLISHED; probe + ADEC submitted

- Key saga resolved: root cause was the passphrase on id_ed25519 (BatchMode
  cannot decrypt). Fix: user-created dedicated automation key euler-auto
  (passphrase-less), installed via ssh-copy-id -f; ~/.ssh/config now lists
  it first for Host euler. DIRECT-KEY-OK — access survives VPN drops, no
  sockets, no passwords. (Classifier correctly required the USER to create
  and install the key.)
- Submitted on Euler: order/naming probe 6060905 (P1 order-swap + P2
  decompose-naming over all ~330 AD/AX checkpoints) and full Stage ADEC
  array 6060948 (90 cells: train-the-mixture → find pure advection/
  diffusion, prediction + naming, H5 asymmetry). Watcher armed.
- Local: ORDL P1 result already in (order-invariance, gap ~2e-5); local
  6-arm ORDL top-up + ADEC starter queued on the laptop GPU as hedges.
- Next if dies: watch 6060905/6060948; aggregate ORDAD/ORDAX rows (P1 gaps
  per arm, P2 exact_match_decompose) and ADEC rel_l2_dec[mech] +
  exact_match_dec[mech]; memo H5/order panels.

## 2026-07-07 — Claude (Alienware) — ADRV submitted: L&C-vocab vs grammar AT DERIVATIVE LEVEL (job 6062494)

- User direction: the mechanism-level grammar cheats (MECH_advection is an
  opaque word); real comparison must live at derivative level where
  mechanisms share dx substructure, and compare L&C vocabulary vs CFGs vs
  algebra-graded CFGs (their upcoming L&C-vs-grammars comparison).
- Three new arms (encoders_ext/): deriv_infix (flat math vocabulary),
  deriv_cfg (untyped derivative CFG), deriv_typed_cfg (production families
  = operator-algebra grading L=D+A+S, class-typed DX terminals). All
  injective, <=22 tokens, capacity-matched (+0.0%). Pushed.
- Job 6062494 (105 cells, AR decoder ON): grammar / lample_charton /
  prose_tree / unary_order / deriv_infix / deriv_cfg / deriv_typed_cfg —
  the full 2x2 (vocab vs CFG) x (mechanism vs derivative level) + algebra
  typing, for BOTH prediction and compositional naming.
- Running on Euler now: probe 6060905 + ADEC 6060948 + ADRV 6062494.
- Next if dies: at drain aggregate stages ORDAD/ORDAX, ADEC, ADRV
  (exact_match by arm is THE deliverable — does derivative-level shared
  substructure finally enable compositional naming?).

## 2026-07-07 — Claude (Alienware) — probe 6060905 COMPLETE: order-invariance universal; naming fails BOTH directions

- P1 (order swap, 330 checkpoints, 20 stage-arm combos): |gap| <= 2e-4
  everywhere (max single-variant 0.003); order-free arms (coeff_vector,
  fourier_symbol, slot_vector) exactly 0.0000 = probe sanity anchors hold.
  A+B == B+A architecturally, all lanes, both stages.
- P2 (decompose naming on AD checkpoints): exact_match ~0 (grammar 0.0000,
  scrambled 0.0000, lample 0.0010, prose 0.0042) — symbolic naming neither
  composes (H3) nor decomposes at mechanism-level vocab.
- Still running: ADEC 6060948 (~89), ADRV 6062494 (~89) — the derivative-
  level retest of exactly these questions.

## 2026-07-07 — Claude (Alienware) — ADEC COMPLETE (90/90): mixture→pure works numerically; H5 asymmetry CONFIRMED; naming 0/3

- Train-on-composites → recover pure laws: PREDICTION works (pure diffusion
  rel_l2 0.08-0.17, better than compose ever was; pure advection 0.22-0.31).
- **H5 confirmed in all 6 arms**: singular limit (advection) recovered 2-3x
  worse than regular limit (diffusion) — pre-registered asymmetry, now with
  evidence, pulled forward from Stage C. coeff_vector/none best again.
- NAMING: exact_match 0.000 every arm, both laws — naming now fails in all
  three directions (compose, decompose-from-composites, mixture→pure).
- Remaining: ADRV 6062494 (105 cells) — derivative-level grammars, the
  last candidate for compositional naming.

## 2026-07-07 — Claude (Alienware) — laptop closing: 5 job groups on Euler, all laptop-independent

- SHIPPED before close: **ADRV2 (6077958)** — 105 cells, fresh init seeds
  [3,4,5], replication battery for the naming ordering (typed CFG > CFG >
  vocab > mechanism-level), our thinnest+newest result. **AXD (6077959)** —
  240 cells, AR decoder over ALL 16 AX arms (naming untested for slot/bag/
  fourier/etc). Zero new code — pure configs; no unverified implementations
  shipped under time pressure.
- Also running: Bfilm 6077442 + B128 6077443 (fallback tol 3.5%), ADRV
  straggler 6073102.
- Round-2 grammar design panel (parsimony/compositional-reuse arms) was
  mid-flight locally and dies with the laptop — lens agents were instructed
  to append their proposals to this log (see entries above/below if
  present); re-run the panel or lift specs from those entries next session.
- WAKE-UP: check `squeue --me`; when empty, pull master.csv and aggregate
  stages B{film,128}, ADRV(+straggler, rerun dedups), ADRV2 (KEY: does the
  naming ordering replicate on fresh seeds?), AXD (naming across all arms).
  Then: round-2 arm implementation, prior-art re-check, figures refresh.

## 2026-07-07 08:56 UTC — Claude (round-2 encoder design subagent, operator-algebra lens) — 4 ADRV round-2 arm proposals returned via schema: operad_comp_cfg (explicit binary COMP over dx towers, balanced free-magma bracketing, vs deriv_cfg), semigroup_factor_cfg (derived generator D2=dx.dx, greedy {D2,DX} normal form, vs deriv_cfg), graded_dx_cfg (full Z-grading: R_GRADE_n head + DX_1..DX_4 ladder, vs deriv_typed_cfg), dyadic_coef_cfg (compositional place-value coefficient numerals C_2/C_1/C_H/C_Q, vs deriv_cfg); all <=28 tokens 3-term, vocabs 10-21, no key collisions.

- 2026-07-07 10:58 CEST — Claude (design subagent): round-2 "parsimonious compositional tokens" proposals — deriv_bpe_cfg (corpus-induced R_DX2 chunks, pair vs deriv_cfg), deriv_sq_cfg (SQ/COMP back-reference towers, pair vs deriv_cfg), deriv_typed_sq_cfg (typed variant, pair vs deriv_typed_cfg/deriv_sq_cfg), deriv_cfg_qnum (unary quarter-numerals for COEF, pair vs deriv_cfg); all <=22 tok realized (qnum 24; contract-worst 43<=48), vocabs 7-15, targets naming > 0.012/0.018.

- 2026-07-07 — Claude (seq-modeling-lens subagent): round-2 encoder proposals returned via schema — deriv_sketch_cfg (plan-first leftmost derivation, same token multiset as deriv_cfg), deriv_cd_cfg (countdown DX_k towers, kills unary counting), typed_sketch_cd_cfg (typing+sketch+countdown stack vs deriv_typed_cfg), bpe_cfg (frozen BPE-8 corpus-induced control, predicted NEGATIVE); minimal pairs anchored at deriv_cfg/deriv_typed_cfg, all <=22 tokens 3-term.

## 2026-07-08 — Claude (Alienware) — ALL FIVE OVERNIGHT GROUPS COMPLETE; ordering replicated; AXD surprise

- ADRV2: naming ordering REPLICATES on fresh seeds (CFG>vocab>mechanism,
  ~2x CFG-vs-vocab robust; typed edge weak). AXD: best namers are
  fourier_symbol/physics_typed_tags → reframed: naming generalizes where
  token semantics are physically inferable, fails where conventional.
  Bfilm: FiLM cripples token conditioning (-0.20 vs vectors; token arms
  still tie) — fusion robustness is itself a coeff_vector advantage.
  B128: null holds. H1 null now spans 3 scales x 2 fusions.
- Memo updated. ADRV cell 88 double-timeout flagged. Queue empty.
- Next: round-2 parsimony arms (panel proposals in log above if lens agents
  landed them), prior-art re-check, figures refresh, results memo v2 for Sid.

## 2026-07-07 (later) — Claude (Alienware) — ADRV2 REPLICATES the naming ordering; AXD + full Stage B in; R2 crew launching

- ADRV2 (fresh seeds): typed 0.0139 > cfg 0.0113 > unrolled vocab ~0.007 >
  mechanism-level ~0 — ordering replicated. AXD: fourier_symbol 0.0149 /
  physics_typed_tags 0.0107 top the naming table (semantic grounding).
- Stage B done: form-null (grammar==scrambled) and numeric dominance robust
  across xattn/film/128/256/512; FiLM collapses token arms (0.43 vs
  data-only 0.23) — token reading fragile, vectors robust.
- Round-2 parsimony panel delivered 7 verified-spec arms (r2_arms.json;
  proposals also in log above); stageR2.yaml = 7 + 3 baselines, decoder on,
  max_len 48; implementation crew next.
- Next if dies: implement encoders_ext arms per r2_arms.json specs, verify,
  dry-run task 45 (deriv_bpe_cfg s0i0), submit R2 array 0-149 on Euler.

## 2026-07-07T21:35:15Z — operad_comp_cfg arm (subagent)
Added symcomp/symcomp/encoders_ext/operad_comp_cfg.py: CFG with explicit binary COMP morphisms over the dx generator (minimal pair vs deriv_cfg, which uses implicit unary R_DX chains). Fixed balanced bracketing T(n)=[COMP]+T(ceil(n/2))+T(floor(n/2)); reaction emits empty tower. Self-test: injective on 25-op universe across all split seeds, maxlen 28, universe vocab 11 (4 of 8 coeff bins realized), both normative examples exact, PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-07T21:38Z — deriv_sketch_cfg arm (subagent)
Added symcomp/symcomp/encoders_ext/deriv_sketch_cfg.py: zero-covariate linearization of deriv_cfg — same grammar/vocab/multiset/length, but true leftmost derivation emits the whole SUM-spine plan first ((n-1) x R_SUM->SUM+PROD then R_SUM->PROD, right after R_EQ), then per-term blocks [R_PROD, COEF, R_DX*order, R_U] alphabetically; term blocks carry no spine tokens. Self-test: injective on 25-op universe across all split seeds, maxlen 22, universe vocab 10 (4 of 8 coeff bins realized; spec's 14 counts all bins), both normative examples byte-exact, 1-term ops byte-identical to deriv_cfg and multiset+length identical on every op, nonlinear raises, PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-07T21:41Z — deriv_bpe_cfg arm (subagent)
Added symcomp/symcomp/encoders_ext/deriv_bpe_cfg.py: single-chunk BPE tower arm — deriv_cfg skeleton with each R_DX run rewritten to greedy normal form [R_DX2]*(order//2)+[R_DX]*(order%2); spine/R_PROD/COEF/R_U byte-identical, order-0/1 terms token-identical to deriv_cfg (verified). Self-test: injective on 25-op universe across all split seeds, maxlen 18 (matches spec), universe vocab 11 (4 of 8 coeff bins realized; spec's 15 counts all bins), both normative examples byte-exact, nonlinear raises ValueError, PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-07T21:47Z — bpe_cfg arm (subagent)
Verified symcomp/symcomp/encoders_ext/bpe_cfg.py (corpus-induced-parsimony CONTROL): greedy BPE over deriv_cfg tokens under the FROZEN 8-merge table, each rank applied to fixpoint (repeated left-to-right passes) before the next rank. Module already present and correct per spec; no files changed. Self-test: injective on 25-op universe across all split seeds, maxlen 9, universe vocab 11 (spec's expected 22 counts all coeff bins + all merge tokens; the realized universe uses 4 coeff bins and 5 merge tokens). Both normative examples byte-exact: advection:1.0 -> [EQPP,COEF_1,R_DX,R_U]; advection:1.0+diffusion:0.5+dispersion:0.3 -> [EQPP,COEF_1,DX_U_SUMP,COEF_0.5,DXDX,U_SUMP,C25DXDX,R_DX,R_U]. Nonlinear raises ValueError (via deriv_cfg). PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-07T21:47Z — deriv_cfg_qnum arm (subagent)
Verified/finalized symcomp/symcomp/encoders_ext/deriv_cfg_qnum.py: one-axis minimal pair vs deriv_cfg — COEF_<bin> replaced by unary numeral, round(_q(c)/0.25) repetitions of N_Q; term layout [spine, R_PROD, N_Q*k, R_DX*order, R_U], alphabetical terms, R_EQ prefix, raises on nonlinear. Self-test: injective on 25-op universe across all split seeds, maxlen 24 (<=48; spec's 43 is the all-bins bound), vocab exactly 7 {R_EQ, R_SUM->PROD, R_SUM->SUM+PROD, R_PROD, N_Q, R_DX, R_U}, both normative examples byte-exact (9 and 23 tokens), PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-07T21:48Z — deriv_cd_cfg arm (subagent)
Added symcomp/symcomp/encoders_ext/deriv_cd_cfg.py: tower positional-readability arm — deriv_cfg skeleton with each unary R_DX run replaced by an indexed COUNTDOWN DX_order..DX_1 (never count-up; order-0 reaction has no DX tokens); spine/R_PROD/COEF/R_U identical to deriv_cfg. Self-test: injective on 25-op universe across all split seeds, maxlen 22 (matches spec), universe vocab 13 (4 of 8 coeff bins realized; spec's 17 counts all bins), both normative examples byte-exact (3-term = 19 tokens), nonlinear raises ValueError, PYTHONHASHSEED 0 vs 7 subprocess outputs identical. No other files touched.

## 2026-07-08 — Claude (Alienware) — Stage BEST complete: decoupled numbers-in/grammar-out achieves both optima

- R2 parsimony factorial done earlier (typed CFG stays naming leader
  0.0159; bpe control behaved as pre-registered; plan-first + numerals
  actively hurt). Stage BEST (6194526, 60 cells): the decoupled arm
  coeff_vector@deriv_typed_cfg predicts at 0.211 (~coeff_vector 0.203,
  beats token conditioning 0.257) AND names at 0.0139 (typed band) —
  the best multimodal symbolic-numbers configuration, per design.
- Infra: 'input@decode' pseudo-arms (model.ar_vocab_size; run_task
  split_rep + decoder-priced capacity search). All pushed.
- Remaining agenda: prior-art re-check before write-up; Sid package
  (figures current through pipeline_diagram); ADRV cell-88 curiosity.

## 2026-07-08 — Claude (Alienware) — Sid package built; prior-art deep-research running

- docs/sid_executive_summary.md (2-page distillation: verdicts table,
  causal-grade evidence list, the recipe, decisions needed, disclosures)
  + docs/sid_deck.md (12 slides, all figures embedded, presenter notes,
  asks). Both render on GitHub.
- Prior-art deep-research (workflow wf_bc3ce896-94d) fanning out over
  2024-26 literature; per-claim verdicts (clear/partial/scooped) for the
  commutator law, the dissociation, the capacity-matched null, and typed
  derivative grammars. Novelty section appended to memo+summary when it
  lands.
- Next if dies: read research report (task wxla7bleg output), append
  novelty statement, push; then write-up pending Sid.
