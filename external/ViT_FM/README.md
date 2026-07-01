# ViT_FM — Foundation Models for PDE Operator Learning

A research codebase for training, finetuning, and running inference with
**ViT‑style foundation models** for PDE operator learning, together with
companion **diffusion / spectral‑resolver** models built on top of the
GenCFD denoiser stack.

The repository supports:

- **2D regression** training and finetuning (`Vit3_pl`, multi‑scale ViT variants).
- **3D regression** training and finetuning with FSDP/DDP, activation
  checkpointing, mixed precision, and patch‑size / resolution surgery for
  finetuning a 2D / coarse‑3D pretrained model on a different geometry.
- **Conditional diffusion** training (GenCFD‑style preconditioned denoiser,
  optionally with EDM skip parameterisation), in particular the
  **spectral resolver** that learns `p(y | x, x̂)` where `x̂` is a
  regression prediction.
- **Single‑shot and autoregressive inference**, with optional diffusion
  post‑correction (zero / non‑zero guidance) and ensemble sampling.
- A long list of in‑house and Poseidon‑style PDE datasets, all dispatched
  through a single [`get_loader`](utils/utils_data.py) factory.

This README is intended to be the **one document** that lets someone who has
never seen this repository before run any training or inference workflow that
the codebase supports. It is long on purpose.

---

## 1. Repository layout

```
ViT_FM/
├── configs/                          # All JSON configs grouped by purpose
│   ├── architectures_regression/     # ViT architecture sizes (tiny/small/medium/base)
│   ├── architectures_diffusion/      # U‑Net / U‑ViT architecture configs
│   ├── data_regression/              # Scratch training configs (regression)
│   ├── data_diffusion/               # Scratch training configs (diffusion / spectral)
│   ├── finetune/                     # Finetuning configs (2D and 3D)
│   ├── inference_regression/         # Pure‑regression inference configs (2D & 3D)
│   ├── inference_spectral/           # Regression + diffusion (spectral) inference
│   └── inference/                    # Misc generation configs (e.g. atm_msc_3d_moist)
│
├── train_regression_pl.py            # Train 2D regression from scratch
├── train_regression_pl3d.py          # Train 3D regression from scratch (FSDP / DDP + ckpt)
├── train_diffusion_pl.py             # Train diffusion / spectral resolver
│
├── finetune_regression_pl.py         # Finetune a 2D regression checkpoint
├── finetune_regression_pl3d.py       # Finetune a (2D or 3D) checkpoint on a 3D task
├── finetune_diffusion_pl.py          # Finetune a diffusion / spectral checkpoint
│
├── inference_regression.py           # 2D regression inference (single / AR)
├── inference_regression3d.py         # 3D regression inference (single / AR)
├── inference_spectral_resolver.py    # Regression + spectral diffusion correction
├── inference_spectral_resolver_MM.py # Same, with "micro / macro" sampling
├── inference_diffusion_pl.py         # Conditional diffusion sampling (GenCFD)
├── inference_param_difference3d.py   # Parameter‑perturbation study on 3D models
├── predict_vit_fm_gencfd.py          # Write GenCFD‑schema netCDFs (direct / AR)
├── eval_loop_spectral_resolver.py    # Statistical evaluation loop (spectra etc.)
├── eval_loop_spectral_resolver_mm.py # Same, micro/macro variant
│
├── submit_scratch_jobs.py            # SLURM submitter — train 2D from scratch
├── submit_scratch_jobs3d.py          # SLURM submitter — train 3D from scratch
├── submit_finetune_jobs.py           # SLURM submitter — finetune 2D
├── submit_finetune_jobs3d.py         # SLURM submitter — finetune 3D
├── scripts/sbatch4090.sh             # Example single‑job sbatch (rtx_4090)
│                                       (other sbatch_*.sh wrappers live under scripts/ too)
│
├── regression/                       # ViT models, lightning wrappers, losses, schedulers
├── diffusion/                        # Loss fns, variance schedules, samplers, U‑Net
├── GenCFD/, gencfd_copy/             # Vendored GenCFD code (denoiser, samplers, IO)
├── dataloader/                       # Custom datasets (CIFAR/MNIST/wave/...)
│   ├── dataloader.py                 # Simple in‑house datasets
│   └── dataloader_poseidon.py        # Poseidon‑style PDE time datasets
├── utils/
│   ├── utils_data.py                 # CLI parsers, `get_loader`, IO helpers
│   ├── utils_finetune.py             # 2D FT: replace encoder/decoder I/O modules
│   ├── utils_finetune_3d.py          # 3D FT: PatchEmbedding3D / Depatchify3D + weight transfer
│   └── utils_inference.py            # CSV bookkeeping, variable naming
└── visualization/                    # Plotting helpers + notebooks
```

Trained model artifacts (checkpoints + per‑run `param_regression_*.json`) are
written under
`/cluster/work/math/braonic/TrainedModels/OOD_Generalization/<which_data>/<tag>_regression/`
(or `..._diffusion_gencfd/`), and finetunes nest inside
`<base_run>/finetuned/<new_data>/<tag>_FT_<new_data>_(native|non_native)_<N>/`.

---

## 2. Environment

The cluster scripts all do the same three things first. On Euler this looks
like:

```bash
source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate
module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy
module load cuda/13.0.2          # only required for some 3D / FSDP jobs
```

Key Python dependencies (already pinned in the venv above): PyTorch ≥ 2.6,
PyTorch Lightning, `wandb`, `einops`, `xarray`/`netCDF4`, `albumentations`
(only for the brain dataset), `matplotlib`, `tqdm`.

Weights & Biases is initialised with `entity="bogdanraonic"`; override by
editing the `wandb.init(...)` block in the training script, or run
`wandb offline` / `WANDB_MODE=disabled` to skip cloud logging.

---

## 3. How configs work

Everything in this repo is driven by **JSON configs**. Three logical groups:

| Group | Purpose | Lives in | Used by |
| --- | --- | --- | --- |
| **Architecture** | Shape of the model only | `configs/architectures_*` | Loaded automatically from the training config |
| **Data / scratch training** | What dataset, batch size, LR schedule, masks, time horizon, … | `configs/data_*` | `train_regression_pl*.py`, `train_diffusion_pl.py` |
| **Finetune / inference** | Point at an *existing* run and override what changes | `configs/finetune/`, `configs/inference_*` | `finetune_*.py`, `inference_*.py` |

All scripts share the same calling convention: `--config=path/to/file.json`.
You may also pass every field on the command line — see
[`read_cli_regression`](utils/utils_data.py), [`read_cli_finetune`](utils/utils_data.py)
and [`read_cli_diffusion_gencfd`](utils/utils_data.py) for the full list of
flags and defaults. If both are given, the CLI flags are ignored when
`--config` is set (the file wins).

When a script runs it writes a copy of its merged config back to
`<workdir>/param_regression_<tag>.json`. **Never edit that file by hand —
it is the source of truth for downstream inference scripts.**

### 3.1 Architecture configs (regression — ViT)

[configs/architectures_regression/](configs/architectures_regression/) defines four sizes:

| File | `dims` | `depth` | `heads` | `dim_head` | `mlp_dim` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `config_basic_vit3_tiny.json` | 256 | 6 | 8 | 32 | 256 | smallest |
| `config_basic_vit3_small.json` | 512 | 10 | 8 | 64 | 1024 | |
| `config_basic_vit3_medium.json` | 512 | 8 | 8 | 64 | 1024 | shallower than small |
| `config_basic_vit3_base.json` | 1024 | 12 | 16 | 64 | 2048 | ViT‑B, ≈125M params |

Other fields:
- `patch_sizes`: spatial patch edge in pixels (default `8`).
- `latent_channels`: encoder/decoder bottleneck width (default `128`).
- `emb_channels`: channel count used for the (optional) Fourier time embedding.

### 3.2 Architecture configs (diffusion — U‑Net / U‑ViT)

[configs/architectures_diffusion/](configs/architectures_diffusion/) — used by the GenCFD
preconditioned denoiser. Example (`config_unet_small.json`):

```json
{
  "channels": [32, 64, 128, 256],
  "num_blocks": 4,
  "noise_embed_dim": 128,
  "proj_channels": 128,
  "num_heads": 8
}
```

### 3.3 Data / scratch training configs (regression)

A canonical example —
[configs/data_regression/config_regression_small_vit3_3d.json](configs/data_regression/config_regression_small_vit3_3d.json):

```json
{
  "device": "cuda",
  "which_model": "basic_vit3",
  "workdir": ".../OOD_Generalization/eul_riemann_ellipse3d/trial_scaling",
  "tag": "trial",
  "loss": 1,                          // 1 = L1, 2 = L2

  "epochs": 200,
  "warmup_epochs": 0,
  "batch_size": 5,
  "peak_lr": 0.00025,
  "end_lr":  0.00005,

  "which_data": "eul_riemann_ellipse3d",
  "in_dim": 5, "out_dim": 5,
  "N_train": 64,

  "is_time": true,
  "is_masked": false,
  "fix_input_to_time_step": null,
  "max_num_time_steps": 7,
  "time_step_size": 2,
  "allowed_transitions": [1,2,3,4,5,6,7],
  "s": 64,
  "is_fourier_emb": true,

  "config_arch": "configs/architectures_regression/config_basic_vit3_small.json",
  "wandb_run_name": "_1",
  "wandb_project_name": "foundation-model"
}
```

The full field reference is in §6.

### 3.4 Data / scratch training configs (diffusion)

[configs/data_diffusion/config_diffusion_gencfd_euler.json](configs/data_diffusion/config_diffusion_gencfd_euler.json)
shows the typical **spectral‑resolver** setup: it conditions on the prediction
of an existing regression model whose outputs were dumped to `spectral_file`
as a netCDF.

```json
{
  "tag": "spectral_10k",
  "which_data": "eul_riemann_curved",

  "is_spectral_resolver": true,
  "spectral_file": ".../predictions_eul_riemann_curved_generated_data/...pred_10000.nc",

  "which_type": "yx",            // see §3.6
  "in_dim": 4, "out_dim": 4,
  "sigma": 100.0,
  "is_exploding": true,          // VE schedule
  "ema_param": 0.999,
  "skip": true,                  // EDM skip parameterisation
  "is_log_uniform": false,
  "log_uniform_frac": 1.0,

  "is_time": true,
  "is_masked": true,
  "max_num_time_steps": 10,
  "time_step_size": 2,
  "allowed_transitions": [1,2,3,4,5,6,7,8,9,10],

  "s": 128,
  "epochs": 100, "batch_size": 36,
  "peak_lr": 2e-4, "end_lr": 1e-5,
  "config_arch": "configs/architectures_diffusion/config_unet_medium.json"
}
```

### 3.5 Finetune configs

The minimum a finetune config has to do is point at an existing training
run (`config_regression`) and say what changes. Example —
[configs/finetune/config_finetune_regression_3d_tg_n32t50.json](configs/finetune/config_finetune_regression_3d_tg_n32t50.json):

```json
{
  "config_regression": ".../OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000",

  "epochs": 100, "warmup_epochs": 1,
  "peak_lr": 2e-4, "end_lr": 2e-5,
  "batch_size": 6, "accumulate_grad": 2,
  "is_precision_16": false,
  "tag": "ft_tg3d_n32t50",

  "which_data": "tg3d_n32t50",
  "N_train": 9500,

  "is_post_trained": true,        // pretrained ckpt expected
  "reinit_ft": false,             // keep encoder/decoder — set true if in/out dim changes
  "init_new": false,              // see §7.4

  "in_dim": 5, "out_dim": 5,
  "err_group":      [1,3,1],
  "err_mask_group": [1,1,1],
  "is_masked": true,

  "max_num_time_steps": 8, "time_step_size": 1,
  "fix_input_to_time_step": null,
  "allowed_transitions": [1,2,3,4,5,6,7,8],
  "is_time": true,

  "loss_type": "rel",   // "rel" or "rel_g" (per‑group)
  "loss": 1,
  "s": 64,

  // optional patch‑size / resolution surgery:
  "s_new": 32,
  "patch_size_new": 2,
  "interpolate_patch_weights": true,

  "wandb_project_name": "foundation-model",
  "wandb_run_name": "ft_tg3d_n32t50_w8_p2"
}
```

### 3.6 Inference configs

Three flavours, depending on which inference script you call.

**Pure regression (2D / 3D)** —
[configs/inference_regression/config_inference_finetune.json](configs/inference_regression/config_inference_finetune.json):

```json
{
  "config_regression_folder": ".../mhd_orszag8_long/5steps_wall2all_32",
  "tags": ["32"],                    // substring filters over subfolders
  "exclude": [],                     // (3D script only) substring filters to exclude
  "error_file": "errors/errors_long_ar.csv",

  "which_data": "mhd_orszag8_long",
  "err_group":      [1,2,1,2],
  "err_mask_group": [1,1,1,1],
  "regression_scheme": [1],          // rollout schedule, see §10.1
  "is_masked": true,
  "is_ar": true,

  "dt": 0.01,
  "max_num_time_steps": 1,
  "time_step_size": 1,
  "allowed_transitions": [1],

  "N_samples": 256,
  "batch_size": 8,
  "tag_data": "0",
  "ood_share": 0.0,
  "device": "cuda",
  "save_data": false,
  "inference_tag": "ar",
  "is_time": true,
  "fix_input_to_time_step": null
}
```

**Regression + diffusion spectral correction** —
[configs/inference_spectral/config_inference_spectral.json](configs/inference_spectral/config_inference_spectral.json)
adds:

```json
{
  "config_diffusion": ".../eul_riemann_curved/spectral_10k_diffusion_gencfd",
  "ignore_diffusion": false,
  "sde_steps": 64,                       // Euler‑Maruyama / ODE steps
  "num_ensemble": 1,                     // independent samples per (x)
  "guidance_strength": 0.0,              // classifier‑free guidance scale
  "diffusion_mask_input":      [1,1,1,1,0,0,0,0,0],
  "diffusion_mask_prediction": [1,1,1,1,0,0,0,0,0],
  "renormalize": true,                   // re‑normalise after each AR step
  ...
}
```

The `_MM` (micro/macro) variant additionally needs `N_macro`, `N_micro`,
and a `macro_id`.

**Generation / probe** —
[configs/inference/config_generate_atm_msc_3d_moist.json](configs/inference/config_generate_atm_msc_3d_moist.json)
is a minimal config for dumping predictions on a probe dataset
(`which_type`, `channel_names`, `max_z`, …).

---

## 4. Training from scratch

### 4.1 2D regression — `train_regression_pl.py`

```bash
python3 train_regression_pl.py \
    --config configs/data_regression/config_regression_basic_vit3_eul_ns_mix.json
```

Behaviour:

- Trainer: `Trainer(devices=-1, strategy=DDPStrategy(find_unused_parameters=False))`.
  It uses **all visible GPUs**, so set `--gpus=N` in your sbatch / wrap with
  `CUDA_VISIBLE_DEVICES`.
- Workdir: `<config.workdir>/<tag>_<N_train>` if `workdir` is given,
  otherwise `.../OOD_Generalization/<which_data>/<tag>_regression`.
- Checkpoints: `<workdir>/model/`, top‑3 by `val_loss`. Logs under
  `<workdir>/logs/` (TensorBoard) and on W&B.
- Loss: `relative_lp_loss_fn(p=loss)` (L1 if `loss=1`, L2 if `loss=2`).
- `val_check_interval` is bumped down to `0.1` for `mix`, `merra`, `pdegym`
  datasets (10 mid‑epoch validations) and to `1.0` with
  `limit_val_batches=0.5` for `long` datasets.

### 4.2 3D regression — `train_regression_pl3d.py`

Same calling convention:

```bash
python3 train_regression_pl3d.py \
    --config configs/data_regression/config_regression_small_vit3_3d_turbo3d.json
```

Differences from the 2D script:

- Adds `accumulate_grad` (gradient accumulation), `bf16-mixed` precision.
- Wraps the model with `Vit3_pl` and then calls
  [`initialize_FT3d`](utils/utils_finetune_3d.py) to install a
  `PatchEmbedding3D`/`Depatchify3D` head with `patch_size = 4` (if `s == 64`)
  or `8` otherwise.
- Activation checkpointing is applied to every `Attention` and `FeedForward`
  block (`apply_activation_checkpointing` + `NO_REENTRANT` wrapper).
- DDP is configured with `static_graph=False`,
  `gradient_as_bucket_view=True`, `bucket_cap_mb=25`.
- `torch._dynamo.config.optimize_ddp = False` (workaround for a known DDP/dynamo
  bug) and `suppress_errors = True`.

### 4.3 Diffusion / spectral resolver — `train_diffusion_pl.py`

```bash
python3 train_diffusion_pl.py \
    --config configs/data_diffusion/config_diffusion_gencfd_euler.json
```

Notes:

- Two variance schedules: `is_exploding=true` ⇒ VE with `sigma_min=0.001`,
  `sigma_max=sigma`; otherwise VP‑style `marginal_prob_std_1`.
- `skip=true` swaps the loss to `loss_fn_denoised(weighting="edm", sigma_data=0.5,
  consistent_weight=0.1)` (i.e. the EDM skip parameterisation).
- `which_type` controls the conditioning topology of the denoiser:
  - `"yx"`: model `p(y | x)` — denoise `y` conditioned on `x`. Standard.
  - `"x"`: unconditional density `p(x)`.
  - `"x&y"`: joint `p(x, y)`.
- **Spectral resolver mode**: set `is_spectral_resolver=true` and
  `spectral_file=<path to .nc>`. The dataloader stacks the cached regression
  prediction onto `x`, so the in‑dim grows by `out_dim` and the model learns
  `p(y | x, x̂)`.
- EMA weights are tracked with `ema_param`.

---

## 5. Finetuning

### 5.1 2D finetune — `finetune_regression_pl.py`

```bash
python3 finetune_regression_pl.py \
    --config configs/finetune/config_finetune_regression_2d.json
```

The script:

1. Locates the latest checkpoint and parameter JSON in `config_regression/`.
2. Loads the architecture from the *original* run (so the backbone size is
   fixed by the pretrained model — never override `config_arch` here).
3. Builds a new `Vit3_pl` and calls [`initialize_FT`](utils/utils_finetune.py)
   to **replace the input lift and output projection** with new modules of
   shape `(new_in_dim → latent_channels)` and `(latent_channels → new_out_dim)`.
4. Writes outputs to
   `<config_regression>/finetuned/<which_data>/<tag>_FT_<which_data>_(native|non_native)_<N_train>/`.
   The string `native` ↔ `reinit_ft=false`, `non_native` ↔ `reinit_ft=true`.

Important flags:

- `reinit_ft`: `true` if input/output channels change. Re‑initialises the
  encoder/decoder heads.
- `init_new`: if `true`, start from a fresh model with the same architecture
  (no pretrained weights loaded — useful as a control).
- `ar_train`: enable autoregressive training (forces `allowed_transitions`
  effectively to a single horizon; the assertion is commented out but keep
  the list short).
- `rescale_time`: rescale relative time inputs into `[0, 1]` for the new task.
- `loss_type`: `"rel"` (relative Lp), `"rel_g"` (per err‑group relative Lp).
- `err_group` + `err_mask_group`: see §6.4.
- `ft_encoder_decoder_warmup_steps`: freeze the backbone for the first N
  optimiser steps so only the new I/O heads adapt, then unfreeze.

### 5.2 3D finetune — `finetune_regression_pl3d.py`

```bash
python3 finetune_regression_pl3d.py \
    --config configs/finetune/config_finetune_regression_3d_atm_msc_dry.json
```

Same general flow as 2D but:

- Uses [`initialize_FT3d`](utils/utils_finetune_3d.py): replaces the 3D patch‑embedding
  and the depatchify head, optionally with **spatial‑adaptive‑pool weight
  transfer** if you change `patch_size` (set `s_new` and `patch_size_new`
  with `interpolate_patch_weights=true`).
- Supports `is_post_trained` (load a previous full FT checkpoint) and
  `is_3d_scratch` (treat the source as a 3D scratch run).
- bf16 / activation checkpointing / DDP options identical to scratch 3D
  training. Use `accumulate_grad`, `attention_chunk_size`, `use_sdpa`
  (defaults `True`) to trade compute for memory.
- A checkpoint compat shim (`load_checkpoint_compat`,
  `load_state_dict_shape_compat`) handles PyTorch 2.6’s `weights_only=True`
  default and silently skips tensors whose shape no longer matches when
  doing patch‑size surgery.

### 5.3 Diffusion finetune — `finetune_diffusion_pl.py`

```bash
python3 finetune_diffusion_pl.py --config <ft_config.json>
```

Inherits `which_type`, `s`, `sigma`, `is_exploding`, `ema_param`, `skip`
from the base run; you typically only override `which_data`, `tag`,
`epochs`, `batch_size`, the LR schedule, `N_train`, and possibly
`is_time` / `is_masked` for the new task.

---

## 6. Config field reference

This section enumerates every field that appears in any config in the repo,
grouped by purpose. Defaults are taken from
[`read_cli_regression`](utils/utils_data.py:130),
[`read_cli_finetune`](utils/utils_data.py:181) and
[`read_cli_diffusion_gencfd`](utils/utils_data.py:313).

### 6.1 General

| Field | Default | Meaning |
| --- | --- | --- |
| `config` | `null` | Path to JSON. If set, **all other CLI flags are ignored**. |
| `device` | `"cuda"` | Device passed into `marginal_prob_std_fn` etc. |
| `tag` | `""` | Short string injected into the workdir name. Make it unique per run. |
| `workdir` | `null` | Override the auto workdir (regression scratch only). |
| `wandb_project_name` | `"foundation-model"` / `"diffusion-project"` | W&B project. |
| `wandb_run_name` | `""` | Suffix appended to the W&B run name. |

### 6.2 Optimisation

| Field | Default | Meaning |
| --- | --- | --- |
| `epochs` | 100 | Training epochs. |
| `warmup_epochs` | 0 | Linear warmup epochs (scheduler in `regression/lr_schedulers.py`). |
| `batch_size` | 20–32 | Per‑GPU batch size. |
| `peak_lr`, `end_lr` | 1e‑4, 1e‑6 | Cosine schedule peak and final LR. |
| `loss` | 1 | Lp exponent for the relative loss (`1`=L1, `2`=L2). |
| `loss_type` | `"rel"` | `"rel"` global relative Lp; `"rel_g"` splits by `err_group`. |
| `accumulate_grad` | 2 (3D) | Gradient accumulation steps. |
| `is_precision_16` | `false` | 3D only — enable fp16 training (else bf16‑mixed). |

### 6.3 Data — spatial / temporal layout

| Field | Default | Meaning |
| --- | --- | --- |
| `which_data` | `"wave"` | Selects a dataset in [`get_loader`](utils/utils_data.py:380). See §7. |
| `in_dim`, `out_dim` | task‑dependent | Channel counts of `x` and `y`. |
| `s` | 128 | Spatial side (resolution). 64 is common for 3D. |
| `N_train` | 128 | Number of trajectories used for training. |
| `ood_share` | 0.0 | Fraction of OOD samples mixed into the loader (legacy). |
| `is_time` | `true` | Time‑conditional model. |
| `is_masked` | `false` | Use channel masking — required for multi‑task / pdegym training. |
| `max_num_time_steps` | 7 | Max horizon (in dataset units) used during training. |
| `time_step_size` | 2 | Δt between sampled snapshots. |
| `fix_input_to_time_step` | `null` | Lock the input to a fixed `t_in`; usually `null`. |
| `allowed_transitions` | `[1,…,7]` | Allowed `Δsteps` values between input and target. |
| `is_fourier_emb` | `true` | Use Fourier time embedding. |
| `rescale_time` | `false` (finetune) | Normalise `t` to `[0, 1]`. |
| `ar_train` | `false` (finetune) | Train fully autoregressive (1‑step rollouts). |

### 6.4 Masking & per‑variable groups

`err_group` and `err_mask_group` are the workhorses that adapt a single
foundation backbone to many datasets with different physical channels.

- `err_group = [g_1, g_2, …, g_k]`: the *real* `out_dim` of the dataset
  partitioned into `k` groups of channel sizes `g_i`. They must sum to the
  number of physical output channels and to `in_dim/out_dim` (after
  whatever shared head expansion the dataset uses).
- `err_mask_group = [m_1, …, m_k]`: which groups are *active* for this
  dataset. `m_i = 1` means group `i` contributes to the loss and to
  reported errors; `m_i = 0` means it is padded / masked out.

So if pretraining used `in_dim = out_dim = 9` but the new dataset only has
4 physical channels, you might write:

```json
"err_group":      [1,2,1,1,1,0,0,0],
"err_mask_group": [1,1,1,1,0,0,0,0]
```

The same lists are used at inference time to compute per‑channel errors and
to mask diffusion conditioning (`diffusion_mask_input`,
`diffusion_mask_prediction`).

### 6.5 Architecture

| Field | Meaning |
| --- | --- |
| `config_arch` | Path to an architecture JSON. Scratch training only — finetune scripts read this from the parent run. |
| `which_model` | `"basic_vit3"` is the only fully supported model now (`MultiVit3_pl`, `MultiVit2_pl` exist but are not the default training path). |
| `s_new`, `patch_size_new` | Override resolution / patch size when finetuning. |
| `interpolate_patch_weights` | If `true`, transfer pretrained patch‑embed weights to the new patch size via spatial adaptive pool. |
| `use_sdpa` | Use PyTorch SDPA fused attention (3D FT default). |
| `attention_chunk_size` | Chunk size for low‑memory attention (`0` = no chunking). |

### 6.6 Diffusion‑specific

| Field | Meaning |
| --- | --- |
| `which_type` | `"yx"` (default), `"x"`, `"x&y"`, `"xy"`, `"y"` — conditioning topology, see §4.3. |
| `sigma` | `sigma_max` of the noise schedule. |
| `is_exploding` | Choose VE (`true`) or VP‑style (`false`) noise. |
| `is_log_uniform` | Log‑uniform sampling of `t` during training. |
| `log_uniform_frac` | Mixing fraction for log‑uniform vs uniform. |
| `ema_param` | EMA decay for parameter averaging (e.g. `0.999`). |
| `skip` | EDM skip parameterisation (`loss_fn_denoised`). |
| `is_spectral_resolver` | Activates `p(y | x, x̂)` mode. |
| `spectral_file` | netCDF with cached `x̂` predictions to condition on. |

### 6.7 Finetune‑only

| Field | Meaning |
| --- | --- |
| `config_regression` | Path to the pretrained run directory. Required. |
| `is_post_trained` | The source is a pretrained checkpoint (load weights). |
| `is_3d_scratch` | The source is a 3D *scratch* run (vs an FT). Affects which subfolder layout the loader expects. |
| `reinit_ft` | Replace input/output heads. Required if `in_dim`/`out_dim` changed. Controls `native` vs `non_native` tag. |
| `init_new` | Build the model from scratch instead of loading the parent checkpoint (control runs). |
| `ft_encoder_decoder_warmup_steps` | Freeze backbone for this many optimiser steps at the start. |
| `interpolate_patch_weights` + `s_new` + `patch_size_new` | See §6.5. |

### 6.8 Inference

| Field | Meaning |
| --- | --- |
| `config_regression_folder` | Either a single run folder or a parent folder of runs to sweep. |
| `tags`, `exclude` | Substring filters used to pick runs out of the parent folder. |
| `inference_tag` | Suffix appended to the saved predictions/error rows. |
| `error_file` | CSV where one row per (model, dataset, scheme) is appended. |
| `save_data` | Dump predictions to netCDF if `true`. |
| `regression_scheme` | Rollout schedule, see §10.1. |
| `is_ar` | Use autoregressive rollouts. |
| `dt` | Physical Δt used to compute lead time / time embeddings. |
| `N_samples`, `batch_size` | Test set size and batch size. |
| `tag_data` | OOD tag passed into the dataset (often `"0"` ⇒ ID test). |
| `config_diffusion` | Optional — path to a trained diffusion run, used by the spectral inference scripts. |
| `ignore_diffusion` | If `true`, skip the diffusion step and only score the regression model. |
| `sde_steps`, `num_ensemble`, `guidance_strength` | Sampler parameters. |
| `diffusion_mask_input`, `diffusion_mask_prediction` | Per‑channel masks for diffusion conditioning. |
| `renormalize` | Re‑apply dataset normalisation between AR steps. |
| `N_macro`, `N_micro`, `macro_id` | Micro/macro sampling for `*_MM.py` scripts. |
| `channel_names`, `max_z`, `overwrite`, `which_type` | Specific to the generation script. |

---

## 7. Datasets supported

All datasets are dispatched from
[`get_loader`](utils/utils_data.py:380). The supported `which_data` values are:

**Vision / debug**: `mnist`, `mnist_diff`, `mnist_class`, `cifar_diff`,
`cifar_class`, `brain`, `custom`.

**2D PDEs (Poseidon‑style time datasets)**:
`ns_shear`, `ns_shear_gencfd`, `ns_shear_gencfd_mm`, `ns_vortex`,
`ns_brownian`, `ns_sin`, `ns_sin_easy`, `ns_pwc`, `ns_gauss`,
`ns_mix1`, `eul_ns_mix1`,
`mhd_orszag8`, `mhd_orszag8_long`,
`eul_riemann_kh`, `eul_riemann_curved`, `eul_gauss`,
`wave`, `wave_ood`, `wave_seismic`,
`allen_cahn`, `helmholtz`, `poisson`, `rich_mesh`,
`shear_layer_rpb`, `shear_layer_rpb_ood`.

**3D PDEs**:
`eul_riemann_kh3d`, `eul_riemann_ellipse3d`, `eul_riemann3d`,
`eul_riemann_curved3d`, `ns_shear3d`, `ns_shear3d_mm`,
`tg3d`, `tg3d_n32t50`, `eul3d_mix1`, `eul_ns3d_mix1`, `ns3d_mix1`,
`atm_msc_3d_dry`, `atm_msc_3d_moist`, `conditional_atm_msc_3d_moist`.

**Real / large**: `merra2`, `pdegym_plus`, `pdegym_giga`, `pdegym_mini`.

Most of these accept the standard kwargs: `max_num_time_steps`,
`time_step_size`, `fix_input_to_time_step`, `allowed_transitions`,
`masked_input`, `is_time`, `rel_time`, `N_samples`, `in_dim`, `out_dim`.

---

## 8. Inference

### 8.1 Regression only (2D / 3D)

```bash
python3 inference_regression.py   --config configs/inference_regression/config_inference_finetune.json
python3 inference_regression3d.py --config configs/inference_regression/config_inference_finetune3d.json
```

What the script does:

1. Resolves `config_regression_folder` into one or more runs. If the path is
   itself a finetune/scratch run, just that run is used. Otherwise the
   script scans subfolders and keeps the ones whose name matches *all*
   strings in `tags` and none in `exclude`.
2. For each run, loads the per‑run `param_regression_*.json`, infers the
   architecture from it, and rebuilds the model (3D scripts call
   `initialize_FT3d`).
3. Loads the best checkpoint from `<run>/model/`.
4. Builds a test loader on `which_data` with the inference‑side
   `max_num_time_steps`, `time_step_size`, `allowed_transitions`,
   `is_masked`, etc.
5. Runs evaluation either *single‑shot* or *autoregressively*
   (`is_ar=true`) following `regression_scheme`:
   - `regression_scheme = [s_1, s_2, …]` means do `len(...)` rollout steps
     of sizes `s_i` each. For example `[2, 3, 2]` with `dt=0.1` evaluates
     errors at lead times 0.2, 0.5, 0.7.
6. Appends per‑channel relative errors to `error_file` (CSV) and,
   if `save_data=true`, writes predictions to
   `<run>/predictions_<which_data>_<inference_tag>/...`.

### 8.2 Regression + spectral diffusion correction

```bash
python3 inference_spectral_resolver.py    --config configs/inference_spectral/config_inference_spectral.json
python3 inference_spectral_resolver_MM.py --config configs/inference_spectral/config_inference_spectral_MM.json
```

Conceptually: run the regression model to get `x̂_t`, then sample
`y_t ~ p(y | x, x̂_t)` with the trained spectral denoiser. The MM variant
draws `N_micro` samples per macro mode and aggregates them.

Knobs that matter most:

- `ignore_diffusion`: short‑circuit to pure regression (use to sanity‑check
  the regression part before sampling).
- `sde_steps` and `num_ensemble`: more steps = better samples, more compute.
- `guidance_strength`: only meaningful when the denoiser was trained with
  classifier‑free guidance.
- `diffusion_mask_input` / `diffusion_mask_prediction`: which channels the
  denoiser conditions on / produces. Note these are *one element longer*
  than `err_group/err_mask_group` because they include the time channel.

### 8.3 Plain conditional diffusion (GenCFD)

```bash
python3 inference_diffusion_pl.py --config <cfg.json>
```

Used when you want to sample from a *pure* diffusion model (no regression
backbone in front). Supports both Euler‑Maruyama and the deterministic ODE
sampler. `which_ckpt` (epoch number) selects a non‑final checkpoint.

### 8.4 Predict‑and‑dump for GenCFD pipelines — `predict_vit_fm_gencfd.py`

This script writes per‑(trajectory, horizon) netCDF files in the schema
GenCFD’s certificate downstream expects:
`(x, y_pred, mask, lead_time, dataset_id [, y_true])`. Two modes:

- **direct** (default): one forward call per `(traj, horizon)`.
- **autoregressive** (`--ar`): unroll 10 steps of `lead_time = 0.1` each;
  the stored entry at horizon `(k+1)·0.1` keeps `x = IC_t=0` and
  `y_pred = AR rollout at step k`, so the joint `[x_at_0, y_pred_at_h]`
  is what the certificate consumes.

Typical invocation:

```bash
python3 predict_vit_fm_gencfd.py \
    --ckpt /cluster/work/.../epoch=10-step=336550.ckpt \
    --arch_config configs/architectures_regression/config_basic_vit3_base.json \
    --train_config configs/data_regression/config_regression_basic_vit3_pdegym_plus.json \
    --dataset ns_shear_pp \
    --num_trajectories 32 \
    --ar \
    --output_dir /cluster/work/.../Predictions/pdegym_plus_vit_ar/
```

### 8.5 Statistical evaluation — `eval_loop_spectral_resolver*.py`

Compute spectra, kinetic energy, structure functions, etc. from the
predictions written by either `inference_spectral_resolver*.py` or
`predict_vit_fm_gencfd.py`, and summarise the results into JSON.

### 8.6 Parameter‑perturbation study — `inference_param_difference3d.py`

Runs the same inference protocol as `inference_regression3d.py`, but loads
*two* models from the same parent folder and compares their predictions
channel‑by‑channel. Useful for scaling / ablation studies.

---

## 9. SLURM submitters and example sbatch

A minimal single‑GPU job is in [scripts/sbatch4090.sh](scripts/sbatch4090.sh):

```bash
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:81g
#SBATCH -A ls_math

source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate
module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy
module load cuda/13.0.2
python3 finetune_regression_pl3d.py --config=configs/finetune/config_finetune_regression_3d_atm_msc_moist_fromdry.json
```

The `submit_*.py` scripts generate one sbatch per `(dataset, N_train)`
combination from a hard‑coded `(n_train_list, gpus_list, time_hours)` table
at the top of the file, then call `sbatch` for you. Read the first ~60
lines of any of them; you typically only edit `which_types`, the `n_train_list`
arrays, and the `tag` / `regression_file` variables.

---

## 10. Recipes

### 10.1 Understanding `regression_scheme`

`regression_scheme` is a list of *step sizes* the inference rolls through.
Each entry consumes `time_step_size · step` lead time. Examples (with
`time_step_size=1`, `dt=0.1`):

| `regression_scheme` | What gets reported | Note |
| --- | --- | --- |
| `[1]` | 1 single‑shot step (lead time 0.1) | direct one‑shot |
| `[2,3,2]` | 3 AR steps of sizes 2, 3, 2 — lead times 0.2, 0.5, 0.7 | mixed schedule |
| `[1]*10` | 10 unit AR steps | full AR rollout |

`is_ar=true` is required for anything past a single‑shot direct evaluation.

### 10.2 Adapting a pretrained 2D backbone to 3D

1. Build the 3D finetune config off the 2D backbone (`config_regression`).
2. Set `is_post_trained=true`, `reinit_ft=false`, `init_new=false`.
3. If `in_dim/out_dim` change, switch to `reinit_ft=true`.
4. If the new resolution doesn’t match the pretrained patch grid:
   `s_new=<new>`, `patch_size_new=<new>`, `interpolate_patch_weights=true`.

The loader will spatially adaptive‑pool the pretrained patch‑embed and
depatchify linear weights into the new geometry, then load the rest of
the backbone normally.

### 10.3 Reproducing a published spectral resolver run

1. Train a regression model on the source dataset (e.g. `pdegym_plus`).
2. Dump predictions to a netCDF using `predict_vit_fm_gencfd.py` or one of
   the inference scripts with `save_data=true`. Note the path.
3. Train a spectral resolver: copy
   `configs/data_diffusion/config_diffusion_gencfd_euler.json`, point
   `spectral_file` at that netCDF, and set `which_data` to match.
4. Inference with `inference_spectral_resolver.py` pointing at both runs
   via `config_regression_folder` and `config_diffusion`.

### 10.4 Multi‑GPU caveats

- The 2D scripts use `DDPStrategy(find_unused_parameters=False)`. If you
  add new optional submodules, set `find_unused_parameters=True` *or*
  guarantee everything is used.
- The 3D scripts use bf16‑mixed + activation checkpointing. If you switch
  to fp16 (`is_precision_16=true`) expect NaNs unless you also reduce
  `peak_lr`.
- `torch._dynamo.config.optimize_ddp = False` is set on purpose; do not
  re‑enable it without testing — there is a known crash.
- `wandb.init` runs only on local rank 0; other ranks set
  `WANDB_MODE=disabled` automatically.

### 10.5 Where outputs go

```
<workdir>/
├── param_regression_<tag>.json         # exact config used (do not edit)
├── model/                              # PL checkpoints (top‑3 by val_loss)
├── logs/                               # TensorBoard
├── finetuned/<which_data>/<tag>_FT_..  # nested FT runs
└── predictions_<which_data>_<inference_tag>/   # inference dumps (if save_data=true)
```

`error_file` CSVs aggregate the *numerical* outcome of every inference run
in one place; they are written via
[`append_unique_dicts_to_csv`](utils/utils_inference.py) which deduplicates
rows by the first `check_columns` fields.

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `Weights only load failed` while loading a checkpoint | PyTorch 2.6 default. Already handled by `load_checkpoint_compat` in the 3D FT script; for other scripts call `torch.load(..., weights_only=False)`. |
| Inference can’t find the checkpoint | `config_regression_folder` is a *parent*, not a run. Add a `tags` filter that narrows it down to exactly one match. |
| DDP hangs at start | Stale W&B service. Confirm `WANDB__SERVICE_WAIT=300` is set (it is, at the top of each script) and that the wandb cache directory is writable. |
| Patch‑size finetune produces garbage | You forgot `interpolate_patch_weights=true`, or `s_new`/`patch_size_new` is not a divisor of the old patch geometry. |
| Spectral inference produces all‑zero predictions | `ignore_diffusion=true` *and* `regression_scheme` doesn’t roll the regression model. Use `regression_scheme=[1]` at minimum. |
| Per‑channel errors look wrong | `err_group` and `err_mask_group` don’t match the dataset’s actual channel layout. They must sum to `out_dim`. |

---

## 12. Where to look in the code

- `Vit3_pl`, `MultiVit3_pl`, `MultiVit2_pl`, transformer blocks:
  [regression/ViTModulev2.py](regression/ViTModulev2.py)
- Patch‑embed / depatchify (3D) and weight transfer logic:
  [utils/utils_finetune_3d.py](utils/utils_finetune_3d.py)
- 2D FT head replacement: [utils/utils_finetune.py](utils/utils_finetune.py)
- LR schedulers (cosine + warmup): [regression/lr_schedulers.py](regression/lr_schedulers.py),
  [diffusion/lr_schedulers.py](diffusion/lr_schedulers.py)
- Loss functions: [regression/loss_fn.py](regression/loss_fn.py),
  [diffusion/loss_fn.py](diffusion/loss_fn.py)
- Variance / SDE schedules: [diffusion/variance_fn.py](diffusion/variance_fn.py)
- Samplers: [diffusion/sampler.py](diffusion/sampler.py)
- Conditional denoiser (GenCFD):
  [GenCFD/model/lightning_wrap/pl_conditional_denoiser.py](GenCFD/model/lightning_wrap/pl_conditional_denoiser.py)
- Dataset registry and `get_loader`: [utils/utils_data.py](utils/utils_data.py)
- Poseidon‑style datasets: [dataloader/dataloader_poseidon.py](dataloader/dataloader_poseidon.py)
- Simple in‑house datasets: [dataloader/dataloader.py](dataloader/dataloader.py)
- Inference accounting / variable naming: [utils/utils_inference.py](utils/utils_inference.py)

---

## 13. Quick start cheatsheet

```bash
# Train ViT‑B regression from scratch on pdegym_plus (multi‑GPU via sbatch):
python3 train_regression_pl.py \
  --config configs/data_regression/config_regression_basic_vit3_pdegym_plus.json

# Train a 3D ViT‑B from scratch on the Euler/NS 3D mix:
python3 train_regression_pl3d.py \
  --config configs/data_regression/config_regression_small_vit3_3d_turbo3d.json

# Finetune a 2D backbone on mhd_orszag8_long:
python3 finetune_regression_pl.py \
  --config configs/finetune/config_finetune_regression_2d.json

# Finetune the same backbone on a new 3D dataset with patch‑size surgery:
python3 finetune_regression_pl3d.py \
  --config configs/finetune/config_finetune_regression_3d_tg_n32t50.json

# Train a spectral resolver conditioned on cached regression predictions:
python3 train_diffusion_pl.py \
  --config configs/data_diffusion/config_diffusion_gencfd_euler.json

# Evaluate a regression run with AR rollouts:
python3 inference_regression.py \
  --config configs/inference_regression/config_inference_finetune.json

# Evaluate a 3D regression run:
python3 inference_regression3d.py \
  --config configs/inference_regression/config_inference_finetune3d.json

# Evaluate regression + spectral diffusion correction:
python3 inference_spectral_resolver.py \
  --config configs/inference_spectral/config_inference_spectral.json

# Dump GenCFD‑schema predictions (AR mode):
python3 predict_vit_fm_gencfd.py \
  --ckpt /path/to/epoch=10-step=336550.ckpt \
  --arch_config configs/architectures_regression/config_basic_vit3_base.json \
  --train_config configs/data_regression/config_regression_basic_vit3_pdegym_plus.json \
  --dataset ns_shear_pp --num_trajectories 32 --ar \
  --output_dir /cluster/work/.../Predictions/pdegym_plus_vit_ar/
```
