# Prior-art / scoop-risk check (2026-07-12; PARTIAL — see status)

Deep-research sweep over 2024–26 literature; every claim below carries
adversarial verification votes (3 independent checkers per claim; quotes
fetched from sources). STATUS: Claims 1–2 verified; Claims 3–4 candidate
overlaps found but their verification pass died on a usage limit — MUST be
re-verified before submission (rerun: workflow wf_bc3ce896-94d resume).

## Claim 1 — commutator law: CLEAR, with one work to cite and position
- **HyCOP (arXiv 2605.00820)** [verified 3-0]: acknowledges commutators
  [F_i,F_j] govern splitting error, but ONLY as constants in classical
  O(h^p) bounds for their fixed modular schedules — never as an empirical,
  representation-independent predictor of neural compositional error.
  Differentiators: we MEASURE the monotone law across 16 representations
  and 4 architectures; theirs is a bound inside one architecture.
- No work found that measures error-vs-‖[A,B]‖ in neural operators.

## Claim 2 — naming/simulating dissociation: CLEAR for dynamics; cite analogue
- **PROSE-PDE (arXiv 2404.12355)** [3-0]: bi-modal predict+decode, but
  symbolic accuracy measured on SEEN operators only; unseen-operator tests
  report data error only. The dissociation is untested there.
- **PROSE (Neurocomputing 2024)** [3-0]: OOD = wider coefficient ranges
  only; never unseen equation forms; reports >97.9% expression validity —
  no decode-failure mode reported anywhere.
- **arXiv 2509.19849** [2-0]: fit/recover dissociation in STATIC symbolic
  regression (high R², ~0% recovery OOD) — closest analogue; different
  domain (no dynamics, no multimodal conditioning, no composition splits).

## Claims 3–4 — UNVERIFIED candidate overlaps (re-check before submission)
- **Unisolver**: scouts report equation-channel ablations and possibly a
  corrupted-syntax counterfactual — if true, partial overlap with our
  Claim-3 interventions (not with capacity matching or the scrambled arm).
- **PITT**: equation-token conditioning of transformer PDE surrogates —
  conditioning exists; no representation comparison claimed.
- **SymPlex**: grammar-constrained autoregressive decoding — potentially
  adjacent to Claim 4's grammar-decoding; task reportedly different.
- **GODE / ODEFormer / seq2seq-compositionality**: unimodal or
  non-compositional per scouts; verify.

## Bottom line (interim)
Headline (commutator law) and the dissociation appear CLEAR with proper
citations. The capacity-matched representation-null framing has no located
precedent but its intervention toolkit overlaps Unisolver's reported
ablations — verify and cite. Typed-grammar decoding gain: no located
precedent; verify SymPlex.
