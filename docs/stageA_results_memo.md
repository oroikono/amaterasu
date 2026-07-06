# SymComp Stage A — Results Memo (2026-07-05)

**Provenance.** Full pre-registered Stage A battery on ETH Euler: 90 cells =
6 representation arms × 5 split-seeds × 3 init-seeds, production config
(N=256, T=32, 320 ICs/operator, 80 epochs, d_model 256, capacity matched to
the pre-registered 2% — worst arm +1.9%). Data: S1 exact spectral
(machine-zero) + S2 well-posed variable-coefficient sweep (D12). Jobs
5746065 (data), 5746066 + 5781548 (cells; 2 node-failure requeues, deduped
by run_id). Frozen artifacts: `symcomp/results_stageA/` (master.csv, 2088
rows; money_plot.png). Every run carries config + git SHA + seeds + data
manifest hash. Analysis code: `scripts/aggregate.py` (paired bootstrap,
stratified H2).

## Panel → hypothesis decisions (per pre-registered rules)

**H1 (headline: grammar > flat reps on zero-shot composition): NEGATIVE.**
Paired over split-seeds (positive = grammar better), rel_l2, compose:

| baseline | grammar − baseline | 95% CI | verdict |
|---|---|---|---|
| coeff_vector | −0.037 | [−0.040, −0.033] | coeff_vector BETTER |
| none (data-only) | −0.012 | [−0.020, −0.004] | data-only slightly better |
| lample_charton | −0.006 | [−0.010, −0.002] | slightly better |
| prose_tree | −0.002 | [−0.008, +0.006] | tie |
| grammar_scrambled | −0.000 | [−0.008, +0.006] | tie |

Grammar confers no zero-shot composition advantage at matched capacity; the
non-symbolic coefficient vector is marginally the best conditioning channel.
(Sign-test floor with 5 seeds is p≈0.06 as pre-registered; CIs are primary.)

**H4 (scrambled-grammar killer control): grammar == scrambled.** Per the
pre-registered rule, no compositional-structure claim is licensed. Combined
with H1, the conclusion is symmetric and clean: whatever the symbol channel
contributes, compositional syntax adds nothing beyond it at this scale.

**H2 (error grows with ‖[A,B]‖): CONFIRMED — the pivot headline.**
Within the pre-registered anchor sweep (operator identity held fixed,
epsilon sweeps the commutator):

| arm | Spearman ρ | Δ rel_l2 (max−min ‖[A,B]‖) |
|---|---|---|
| coeff_vector | +0.72 | +0.062 |
| grammar_scrambled | +0.68 | +0.048 |
| prose_tree | +0.66 | +0.053 |
| lample_charton | +0.66 | +0.053 |
| grammar | +0.64 | +0.049 |
| none (data-only) | +0.48 | +0.040 |

Zero-shot error degrades monotonically with the commutator for EVERY arm —
no representation escapes the BCH correction. This is the pre-registered
mechanism ("symbols carry the generator, not the flow") and it holds
regardless of representation choice. NOTE: the naive pooled regression
(also reported by aggregate.py) is stratum-confounded and near-zero — the
stratified test is the correct pre-registered one; keep both in the paper
for transparency.

**H3 (grammar's edge larger on discovery): NOT YET TESTABLE.** The
autoregressive decoder is unimplemented; the rep-agnostic baseline head
(trajectory-prefix input, D13) shows no separation as expected
(mech_f1 ≈ 0.67, coef_mae ≈ 0.31, all arms).

**H5 (decompose asymmetry): DIRECTIONAL DATA ONLY.** Decompose rel_l2 ≈
0.29–0.33 (vs compose ≈ 0.25–0.30), one held-out primitive per seed; the
nu→0 singular-limit test is Stage C. No decision taken.

## E2/E3 interventions (added 2026-07-05; independent full retrain)

The null is **strong-form, not vacuous** — the models demonstrably use the
symbol channel:

- **E2 (channel masking):** ablating symbols at eval costs +0.063–0.069
  rel_l2, positive in **100% of variant-cells** for every symbolic arm
  (data-only arm exactly 0.000 — ablation no-op sanity check). The symbol
  channel carries roughly as much error-reduction as the entire commutator
  degradation range.
- **E3 (counterfactual swap):** feeding the anchor's data with a PURE-
  advection symbol pulls predictions 27–32% of the way toward the wrong
  (pure) solution — the channel is causally steering, in every arm.
- **Replication:** this rerun independently retrained all 90 cells; H1, H2
  (stratified), and H4 reproduced within CI wiggle. Stage A conclusions
  rest on two complete training rounds.

One-sentence synthesis: *models consume the symbols and are steered by
them; the symbolic FORM is irrelevant (down to scrambled syntax); and no
form escapes the commutator law.* Symbols are, in effect, a lookup of the
generator's coefficients — which is why the minimal coefficient vector is
the best (and cheapest) conditioning channel.

## Stage AX — exploratory 16-arm generality sweep (2026-07-06; NOT pre-registered)

240 cells (16 representations × 15 seed-pairs), same protocol, max_len 48.
Multiple-comparison and covariate caveats apply (sequence length, realized
vocab differ by arm; slot/bag arms trigger capacity reallocation to the
data branch). As measured:

- **The commutator law is universal: Spearman ρ positive for all 16 arms**
  (+0.32 slot_vector … +0.75 physics_typed_tags).
- **Semantics does not beat syntax:** `fourier_symbol` (tokenized samples
  of the operator's own Fourier multiplier — the sufficient statistic in
  the commuting stratum) ties grammar (+0.005, CI spans 0).
- **Serialization order is irrelevant:** postfix vs prefix (exact
  minimal pair) statistically indistinguishable.
- **An ordering exists at 15-seed resolution** (grammar-minus-arm,
  positive = grammar better): numeric-flavored conditioning best
  (coeff_vector −0.052; digit_p10 −0.026; unary_order −0.025), classic
  tree serializations mid-pack (−0.004…−0.010), heavily-structured
  layouts worst (subgrammar_typed_rules +0.012; slot_vector +0.051).
  The slot_vector↔coeff_vector gap (≈0.10) isolates the CONTINUOUS-value
  input pathway, not the fixed-slot layout, as coeff_vector's advantage.

## H3 probe (AR discovery decoder; single grammar cell, 2026-07-06)

The autoregressive decoder (rep's own vocab, conditioned on observed
trajectory frames) decodes TRAINING operators nearly perfectly (11/12
strict exact-match) but scores **0.000 on every held-out composition** —
while the same model's prediction head handles those compositions at
rel_l2 ≈ 0.2–0.3. Naming a law and simulating it dissociate: the model
cannot compose symbolic NAMES for unseen combinations it can nonetheless
predict. The full 6-arm Stage AD battery (whether any representation's
decoder composes names better) is running; results will be appended as
measured.

## Caveats
- Single fusion (xattn) + backbone (transformer); Stage B robustness sweep
  pending — H1's negative should be confirmed across fusion/backbone/scale
  before it is called final.
- Stage AX and AD are exploratory (not pre-registered); report them as
  generality/mechanism probes, never pooled with Stage A statistics.
- Discovery baseline only (see H3). 5 split-seeds → sign-test floor 0.06.
- The 18-cell local (Alienware) partial replication was discarded (mixed
  split-code versions); Euler is the sole source of truth for Stage A.

## Recommended next steps (in order)
1. Stage B robustness on the pivoted headline (does the commutator law and
   the H1 null survive fusion/backbone/scale?).
2. AR discovery decoder → H3 becomes testable (the last place a
   representation effect could plausibly live).
3. E2/E3 interventions at Stage A scale; more split seeds if p<0.05 wanted.
4. Re-run the prior-art check before any write-up (scoop risk is on the
   commutator law, which is now the headline).
