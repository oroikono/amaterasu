# DECISIONS

## Key decisions and why

### D1 — Headline claim: grammar > other symbolic reps on zero-shot composition
**Why:** "another multimodal PDE model" is a crowded, hard-to-defend contribution;
"a controlled representation comparison showing grammar generalizes
compositionally where flat reps don't" is a specific, unoccupied claim. Prior-art
search confirmed nobody has run the controlled representation comparison on a
held-out mechanism-composition task. The commutator law is the *explanation* for
where the win holds, demoted from co-headline so the paper has one clear spine.

### D2 — Commutator (`||[A,B]||`) is the explanatory variable
**Why:** It converts a binary ("does composition transfer?") into a continuous,
pre-registered, model-independent predictor with real theory behind it (semigroup
factorization for commuting operators; BCH/Zassenhaus corrections otherwise). It
also makes a negative result publishable and directly instantiates the
"symbols carry the generator not the flow" framing. The nearest prior works
(Neural Operator Splitting, HyCOP) name non-commutativity but never measure it as
the predictor — that is the open gap.

### D3 — Scope: 1D, 6–8 mechanisms, many held-out combinations
**Why:** Defensibility scales with the number of held-out combinations and split
seeds (defeats the "cherry-picked split" attack). 1D keeps solvers exact/cheap so
the commuting stratum is machine-zero clean and the month-timeline is feasible.
2D is explicitly out of scope for month 1 (listed as a possible later extension).

### D4 — Both prediction and discovery tasks
**Why:** Referees weight them differently, and H3 predicts grammar's advantage is
*larger* on discovery (where structure matters most). Reporting both turns a
potential confound into a coherent directional story, and discovery is where the
identifiability/certification thesis connects.

### D5 — Reimplement our 4 reps + scrambled control + data-only floor; cite SOTA
**Why:** Racing heterogeneous external codebases (PROSE, PDEformer, Unisolver) at
different training budgets is unfair and time-expensive. Re-instantiating their
*conditioning ideas* as representation arms under one matched protocol isolates the
variable they confound — a cleaner, faster, more defensible comparison. External
SOTA is cited in related work.

### D6 — Pre-registration + scrambled-grammar control are the defensibility core
**Why:** Pre-registering hypotheses, metrics, and decision rules before running
removes "post-hoc" attacks. The single strongest referee objection to D1 is
"your grammar bakes the additive-composition answer into the representation."
The scrambled-grammar arm (identical machinery, permuted non-compositional
productions) is the control that answers it: if real grammar >> scrambled grammar,
the win is compositional *structure*, not grammar-ness.

### D7 — Matched capacity is enforced in code, not assumed
**Why:** "Your win is a capacity/tokenization artifact" is the first attack.
Param counts are asserted equal within tolerance; the data-only and coeff-vector
arms widen their data branch (via a searched hidden size) to compensate for a
smaller/absent symbol branch. A scale ablation (3 model sizes) shows the gap is
scale-robust.

### D8 — Stage A (day 14) is the freeze point / minimum publishable core
**Why:** Risk management. A month is tight for prediction + discovery + all
ablations + a second backbone + nonlinear physics. Defining a freeze point
guarantees a defensible deliverable for the supervisor even if later stages slip.

### D9 — Exact spectral solver for the commuting stratum
**Why:** Constant-coefficient linear operators are diagonal in Fourier space, so
`exp(t*L_hat)` is exact (no time-stepping error). This makes S1 ground-truth clean
and the commuting-composition claim effectively a theorem the experiment confirms,
rather than something contaminated by numerical error.

### D10 — Durable storage must not be scratch (added after storage review)
**Why:** Euler personal scratch is purged at 15 days with no backup; a one-month
project would lose its earliest runs. Results go to group work/project storage +
home + wandb; raw regenerable data may use scratch.

### D11 — Storage/registry design (implemented 2026-07-04)
**Why each choice:** (a) Paths come from env vars (`SYMCOMP_WORK_DIR`,
`SYMCOMP_HOME_ARCHIVE`, optional `SYMCOMP_VENV`) — never hardcoded group paths
(repo rule); `registry.work_dir()` REFUSES to run on a cluster node (detected
via `$SCRATCH`) without `SYMCOMP_WORK_DIR`, so a misconfigured job fails loudly
instead of writing to purgeable scratch. (b) Per-run `runs/<run_id>/rows.csv`
is the source of truth (zero cross-task contention); the file-locked
`results/master.csv` is a convenience union, and `rebuild_master()` can always
regenerate it — this de-risks unknown flock semantics on Lustre. (c) Every row
is stamped with a `run_id` column (timestamp + git SHA + cell + collision
suffix) so requeued/re-run SLURM tasks are dedupable; the 11 pre-registered
analysis columns are unchanged. (d) The venv lives on work storage, not
scratch, so the 15-day purge cannot kill the software mid-project. (e) Small
artifacts are copied to the home archive at end of job (atomic rename) as a
second copy on backed-up storage.

## Open questions
1. **Which group work/project path and quota** on Euler? (`/cluster/work/<group>`)
   — needed to finalize storage wiring. Confirm with `lquota`.
2. **Discovery decoder design:** autoregressive over each rep's own vocab vs a
   shared canonical target? The former tests the rep more directly; the latter is
   simpler to score. Leaning autoregressive-over-own-vocab for the symbolic arms
   with a canonical exact-match metric.
3. **How to fully close the 2–3% capacity residual** on coeff_vector/data_only —
   dummy matched-parameter adapter, or just transparently report per-arm counts?
4. **Triple compositions:** include 3-term held-out combos in Stage A or defer to
   Stage C? Currently sampled into the universe but lightly held out.
5. **Commutator normalization for plotting** — raw spectral operator norms are
   large; only the ordering is load-bearing. Settle on a normalization (per-pair
   max vs global) so the money-plot x-axis is comparable across strata.
6. **S2 coverage:** is the single advection×diffusion variable-coefficient family
   enough to trace the degradation curve, or extend the ε-sweep to other
   non-commuting linear pairs for a denser curve?
7. **Burgers commutator proxy (S3):** the nonlinear stratum uses a proxy for
   `||[A,B]||`; decide whether to formalize a better surrogate or treat S3
   qualitatively.
8. **Significance with 5 split-seeds:** the paired sign test floor with 5 seeds is
   p≈0.06; consider more split seeds if a sub-0.05 headline p-value is wanted.
