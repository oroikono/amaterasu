# SymComp — Ultra-Defensible Experiment Plan (Month-1, Euler)

**Headline claim (H1).** A *grammar-structured* symbolic representation yields
strictly better **zero-shot generalization to held-out mechanism compositions**
than flat symbolic representations (Lample–Charton prefix, PROSE-style tree) and
than a non-symbolic coefficient-vector control, under matched capacity, compute,
and information.

**Explanatory mechanism (H2, why/where).** Compositional zero-shot error is
predicted by the **commutator magnitude** `||[A,B]||` of the composed mechanisms:
near-perfect transfer in the commuting limit, monotone degradation as the
operators stop commuting. Grammar's advantage is largest where composition is a
single production (additive generator) and shrinks into the strongly
non-commuting / singular regime — because **symbols carry the generator, not the
non-abelian flow correction**.

**Scope.** 1D, 6–8 mechanisms, a *combinatorial battery* of held-out
compositions. Two tasks: **forward prediction** (rollout) and **discovery**
(recover the symbolic operator). Grammar's edge is predicted to be *larger on
discovery* (H3) — the task where structure matters most.

This document is the pre-registration + engineering spec. Nothing below is
chosen after seeing results; the decision rules are fixed in advance. That single
discipline is what converts "grammar looked better" into a referee-proof claim.

---

## 1. The defensibility argument (referee attacks → design)

The plan is built backwards from the attacks a NeurIPS/ICLR referee will make on
"our representation wins."

| # | Attack | Pre-built defense |
|---|--------|-------------------|
| A1 | "Grammar win is a capacity / tokenization artifact." | Matched `d_model`, depth, optimizer, schedule, step budget, early-stop rule; param counts asserted equal within 2%. Scale ablation (3 model sizes) shows the gap is scale-robust. |
| A2 | "You cherry-picked the held-out split." | Combinatorial split **battery**: many held-out combos × 5 split-seeds; report mean ± bootstrap CI over splits, not a single pair. |
| A3 | **"Grammar just bakes the answer in (OP→OP+TERM = additive composition)."** | **The killer control: a *scrambled-grammar* arm** — identical grammar machinery and token budget, but with non-compositional / permuted productions. If real-grammar ≫ scrambled-grammar, the win is the *compositional structure*, not "being a grammar." Also test compositions needing >1 production and product (nonlinear) terms the grammar does **not** trivially encode. |
| A4 | "Prediction and discovery are confounded." | Two separate heads, reported independently; H3 predicts a *larger* grammar gap on discovery — a coherent directional story, not a confound. |
| A5 | "Maybe it's your transformer." | Backbone robustness: rep comparison repeated on a 2nd backbone (latent-VAE encoder/decoder) and a 2nd fusion (xattn vs FiLM). Win must survive. |
| A6 | "Not statistically significant." | Seeds × split-seeds; paired bootstrap across splits; report effect size + 95% CI + paired sign test. |
| A7 | "Commutator story is post-hoc." | `||[A,B]||` computed analytically, independent of any model; H2 pre-registered; report error~commutator regression with R² and CI. |
| A8 | "No SOTA comparison." | External SOTA cited; our 4 reps are faithful re-instantiations of their *conditioning ideas* (PROSE tree, L–C prefix, Unisolver/equation-aware coeff-vector) under matched training — argued explicitly as a *cleaner* controlled comparison than cross-codebase races. |
| A9 | "Leakage." | Automated leakage check: every held-out combo's primitives appear in train; the combo (canonicalized) never does. Canonicalization collapses `A+B≡B+A`. |
| A10 | "Only easy physics." | Strata S1→S3 incl. variable-coeff and Burgers; decompose direction + ν→0 singular limit reported as a documented asymmetry. |

---

## 2. Pre-registered hypotheses, metrics, decision rules

**Hypotheses**
- **H1** grammar < {prose_tree, lample_charton, coeff_vector, data-only} in
  zero-shot composition rel-L2 (prediction) at matched capacity, averaged over
  the held-out battery, S1∪S2. *(lower error = better)*
- **H2** zero-shot rel-L2 increases monotonically with `||[A,B]||`; Spearman
  ρ(error, commutator) > 0 with 95% CI excluding 0, for every representation.
- **H3** grammar's *relative* advantage (vs best flat rep) is larger on discovery
  than on prediction.
- **H4** real-grammar ≪ scrambled-grammar (the A3 control): compositional
  structure, not grammar-ness, drives the win.
- **H5 (asymmetry)** decompose→pure transfer: clean for the regular limit
  (a→0, → diffusion), degraded for the singular limit (ν→0, → advection).

**Primary metrics**
- Prediction: relative L2 of the rollout on held-out compositions (per-combo,
  aggregated).
- Discovery: (i) canonical-form **exact-match** rate; (ii) mechanism-set F1
  (did it recover the right terms); (iii) coefficient MAE on recovered terms.
- Mechanism axis: analytic `||[A,B]||` per held-out combo (model-independent).

**Decision rules (fixed now)**
- H1 holds (grammar best, CI-separated from 2nd best on the battery) → **headline stands; write it.**
- Grammar ≈ flat but **H2 holds** → pivot headline to the commutator law
  (rep-agnostic), grammar demoted to a positive-but-secondary ablation.
- H4 fails (scrambled ≈ real grammar) → **do not claim grammar structure**; report
  honestly that "any structured side-channel" suffices — still a finding.
- Discovery shows grammar edge but prediction doesn't (or vice-versa) → report the
  split per H3; frame around the task where structure provably matters.
- Model recovers part of the Burgers commutator correction (E4 residual stays low
  into S3) → report as "architecture partially recovers non-abelian structure the
  symbol can't encode" — a finding, not a failure.

---

## 3. Mechanisms & data (1D, periodic)

**Linear constant-coefficient primitives** (all mutually commuting; S1 floor):
advection `a∂ₓ`, diffusion `ν∂ₓₓ`, dispersion `β∂ₓₓₓ`, reaction `r·u`,
hyperdiffusion `-γ∂ₓₓₓₓ`. **Nonlinear primitives** (break commutativity; S3):
Burgers advection `u∂ₓu`, cubic reaction `u−u³` (Fisher/Allen–Cahn-type).
→ 5 linear + 2 nonlinear = **7 mechanisms**, expandable to 8 (add `u²` source).

**Strata**
- **S1 commuting** — const-coeff linear, `||[A,B]||=0`. Exact spectral solution
  (machine-zero data). Sanity floor; the additive generator is exactly what
  symbols carry.
- **S2 weak-non-commuting** — variable-coeff linear, `c(x)=c₀+ε·g(x)`; `||[A,B]||`
  swept continuously by ε. The degradation-curve stratum (ETDRK4 / spectral RK4).
- **S3 strong/singular** — Burgers & cubic-reaction compositions; ν→0 vanishing
  viscosity (the asymmetry). ETDRK4.

**Trajectory budget (target):** per operator, ~256 train ICs / 64 test ICs,
T=32 timesteps, N=256 grid; 3 noise levels {0, 1e-3, 1e-2}. Smooth random
low-mode ICs. Store as sharded `.npy`/`.npz` or HDF5 per operator (see §9).

---

## 4. Split protocol (combinatorial, leakage-checked)

Enumerate compositions: all singletons, all pairs (and a sample of triples) over
the 7 mechanisms. For each **split-seed** (×5): randomly designate a set of
pair/triple compositions as **held-out**, subject to the constraint that **each
held-out combo's primitives are individually present in train** (so generalization
is genuinely compositional, not extrapolation to unseen mechanisms). Both
directions:
- **compose:** primitives in train → composite held out.
- **decompose:** composite in train → a pure piece held out (only where that
  piece isn't otherwise trained).

**Hygiene (automated, fails loudly):** canonicalize every operator (sorted terms,
binned coeffs) → assert no canonical held-out form appears in train; assert every
held-out primitive appears in train; log the full manifest per seed.

---

## 5. Models (matched everything)

**Representation arms (the controlled variable):**
1. `grammar` — production-rule sequence (composition = one `OP→OP+TERM`).
2. `prose_tree` — PROSE-style flattened tree tokens.
3. `lample_charton` — prefix/Polish expression tree.
4. `coeff_vector` — non-symbolic control (Unisolver/equation-aware style).
5. `data_only` — no symbol channel (floor; data branch widened to match params).
6. **`grammar_scrambled`** — the A3 control: same machinery, permuted
   non-compositional productions.

**Backbones:** (i) transformer encoder–decoder (primary); (ii) latent-VAE
encoder/decoder (robustness). **Fusion:** xattn (primary, = PROSE feature-fusion)
and FiLM (robustness). **Two heads:** forward-prediction head (→ trajectory) and
discovery head (→ symbolic operator, autoregressive over the *same* rep's vocab;
for coeff_vector/data_only, a regression+classification head over mechanisms).

**Matched-capacity contract (asserted at construction):** identical `d_model`,
depth, head count, FFN width, dropout, optimizer (AdamW), LR schedule (cosine +
warmup), total optimizer steps, batch size, early-stop on a held-*in* val set.
Param counts logged and asserted equal within 2%. Three sizes for the scale
ablation: d_model ∈ {128, 256, 512}.

---

## 6. The ablation matrix

Primary sweep (Stage A, the headline): **6 reps × 2 tasks × {S1,S2} × 5 split-seeds × 3 seeds**, backbone=transformer, fusion=xattn, d_model=256.

| Axis | Levels | Purpose | Stage |
|------|--------|---------|-------|
| Representation | grammar, prose_tree, lample_charton, coeff_vector, data_only, **grammar_scrambled** | **H1, H4** headline + killer control | A |
| Commutator stratum | S1, S2 (ε-sweep), S3 | **H2** mechanism law | A→C |
| Task | prediction, discovery | **H3** | A |
| Split-seed | 5 | **A2** robustness to split | A |
| Init-seed | 3 | **A6** significance | A |
| Fusion | xattn, FiLM | **A5** | B |
| Backbone | transformer, VAE | **A5** | B |
| Model scale | d_model 128/256/512 | **A1** scale-robustness | B |
| Noise | 0, 1e-3, 1e-2 | discovery robustness | B |
| Data budget | 64/128/256 ICs | sample-efficiency curve | C |
| Direction | compose, decompose | **H5** asymmetry | C |
| Singular limit | ν ∈ {0.1,0.05,0.02,0.01} | **H5** vanishing-viscosity | C |

Stage A is the **minimum publishable core**. B and C are strengthening.

---

## 7. SOTA positioning (cite, don't re-race)

Cited as related/SOTA, not reimplemented as full systems: PROSE / PROSE-PDE /
PROSE-FD (bi-modal symbolic+numeric), PDEformer-1/2 (graph-symbolic
conditioning), Unisolver (PDE-component conditioning), SymPlex/SymFormer
(grammar-constrained *expressivity*, not compositional generalization),
Equation-Aware Neural Operators (held-out Burgers via coeff vector), Neural
Operator Splitting and HyCOP (architectural composition; non-commutativity named
but never measured), ICON/ICON-LM (in-context). **The explicit argument for
referees:** rather than race heterogeneous codebases at different budgets, we
re-instantiate their *representation/conditioning ideas* as arms under one matched
training protocol — isolating the variable they confound. Our novelty vs each is
in §1/A8 and the prior-art memo: nobody crosses (held-out mechanism composition) ×
(controlled representation comparison) × (commutator law) × (discovery+prediction).

---

## 8. Statistics

Per metric: aggregate over init-seeds within a split-seed, then **paired bootstrap
(10k resamples) across split-seeds** for grammar-minus-baseline differences →
95% CI + paired sign test. Effect sizes reported. Commutator law: per-rep Spearman
ρ and an OLS `rel-L2 ~ ||[A,B]||_normalized` with R² and CI. All numbers emitted
to `results/aggregate.csv` + auto-generated figures (the money plot: error vs
commutator, one line per rep, CI bands).

---

## 9. Euler operations (ETH cluster)

- **Scheduler:** SLURM. Use array jobs over the (rep × split-seed × init-seed)
  grid; one GPU per task. Templates in `cluster/` (`sbatch_stageA.sh`,
  `sbatch_data.sh`).
- **Stack:** `module load stack/2024-06 gcc python_cuda/3.11` (adjust to current
  Euler default), then a venv: `python -m venv $SCRATCH/venvs/symcomp`. Pin
  `requirements.txt`. Confirm CUDA torch matches the loaded CUDA module.
- **Storage:** datasets on `$SCRATCH` (large, fast, purged) not `$HOME`. Data gen
  is a separate array job writing per-operator shards; training reads read-only.
  Checkpoints + logs to `$SCRATCH/symcomp/runs/<run_id>`.
- **Logging:** wandb offline mode on compute nodes (`WANDB_MODE=offline`), sync
  from the login node; or CSV logger fallback (built in).
- **Budget estimate (order-of-magnitude):** Stage A ≈ 6 reps × 5 split × 3 seed ×
  2 tasks = 180 runs; a 256-dim model on this data is minutes–low-hours per run
  on one modern GPU → comfortably a few GPU-days, i.e. days of wall-clock with
  modest parallelism. B/C add ~2–3×. Fits the one-month target with buffer.
- **Reproducibility:** every run dumps its resolved config + git SHA + seed +
  data manifest hash. `make reproduce RUN=<id>` re-runs identically.

---

## 10. Timeline & the freeze point

| Days | Milestone | Gate |
|------|-----------|------|
| 1–3 | Euler env + data-gen array job for S1∪S2 (7 mechanisms); leakage checks green; physics validation passes on cluster | data frozen |
| 4–7 | Stage-A pipeline runs end-to-end for all 6 reps, both heads, 1 seed; wandb/CSV logging; matched-param assertions pass | pipeline frozen |
| 8–14 | **Full Stage A**: 6 reps × 5 split × 3 seed × 2 tasks; aggregate + money plot + paired CIs | **MINIMUM PUBLISHABLE CORE — freeze here** |
| 15–21 | Stage B: fusion, backbone, scale, noise robustness | strengthening |
| 22–27 | Stage C: data-budget, decompose/asymmetry, S3 Burgers; commutator regression finalized | strengthening |
| 28–30 | Aggregate everything, regenerate all figures, write the results memo for Sid | **deliverable** |

If anything slips, **Stage A at day 14 is already a defensible result for Sid.**
Everything after is upside. Protect the freeze point.

---

## 11. Deliverables for Sid (end of month)

1. The money plot: zero-shot composition error vs commutator, one line per
   representation, CI bands, S1∪S2 (+ S3 if reached).
2. The H1 table: grammar vs each baseline, prediction + discovery, paired CIs.
3. The H4 control: real vs scrambled grammar (the "it's the structure" panel).
4. The commutator regression (H2) with R².
5. The decompose asymmetry panel (H5).
6. `results/aggregate.csv` + full run manifests + configs (reproducible).
7. A 2-page results memo mapping each panel to H1–H5 and the decision taken.

---

## 12. What's scaffolded vs to-implement on Euler

Built and validated locally (delivered skeleton + this upgrade): operator algebra
& exact/ETDRK4 solvers (physics verified to machine zero), the 4 reps + scrambled
control, combinatorial split generator with leakage checks, matched-capacity
model with prediction head, the four interventions, CSV/plot aggregation.

To implement on Euler (clear TODO markers in code): the discovery head's
autoregressive decoder + discovery metrics; the VAE backbone arm; SLURM array
templates wired to your module stack; wandb sync; the paired-bootstrap stats in
`scripts/aggregate.py`; the 8th-mechanism extension. These are spec'd precisely so
Claude-on-Euler implements them without re-deciding anything in this document.
