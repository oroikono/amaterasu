"""Run a pretrained ViT-FM regression model on GenCFD probe datasets and save
predictions in the schema consumed by GenCFD's certification pipeline.

Pipeline:
    GenCFD dataloader (pdegym_plus-padded, 9-channel)
        -> ViT3 regression  (operates directly on the padded 9-channel layout)
            -> write_prediction_file netCDF
                (x, y_pred, mask, lead_time, dataset_id [, y_true])

The predicted tensor is kept in the dataset's normalized coordinates so the
downstream joint log-likelihood / certificate is computed in the same frame
the diffusion denoiser was trained in. Masked channels (mask=0) in the
prediction are zeroed so they do not inflate the certificate at padded slots.

Sample set
----------
The script restricts the test set to ``t_init = 0`` only. For each trajectory
this gives one sample per (t_init=0, t_final) horizon -- typically
``t_final in {2, 4, ..., 20}`` -> ``lead_time in {0.1, 0.2, ..., 1.0}``.

Two prediction modes (mutually exclusive)
-----------------------------------------
- **direct** (default): one ViT forward per (trajectory, horizon) pair. Each
  call uses the horizon's own ``lead_time`` as conditioning. Equivalent to the
  original script behaviour, just restricted to ``t_init = 0``.

- **autoregressive** (``--ar``): per trajectory, the ViT is unrolled in 10
  fixed steps of ``lead_time = 0.1`` each. Step ``k`` consumes step ``k-1``'s
  output (with the masked-channel pad tokens restored from the initial
  condition). The saved entry at horizon ``(k+1) * 0.1`` stores the
  **initial condition** as ``x`` and the **AR-rolled prediction at step k**
  as ``y_pred``, so the joint ``cat(x_at_t=0, y_pred_at_t=horizon)`` is what
  the certificate downstream consumes.

Example
-------
::

    python predict_vit_fm_gencfd.py \\
        --ckpt /cluster/work/.../epoch=10-step=336550.ckpt \\
        --arch_config configs/architectures_regression/config_basic_vit3_base.json \\
        --train_config configs/data_regression/config_regression_basic_vit3_pdegym_plus.json \\
        --dataset ns_shear_pp \\
        --num_trajectories 32 \\
        --ar \\
        --output_dir /cluster/work/.../Predictions/pdegym_plus_vit_ar/

Requires both this repo (for the ViT model classes) and the GenCFD repo (for
the probe dataloaders and the prediction-IO schema) to be importable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def _collate_optional(values: list[Any]) -> Any:
    """Collate fields that may be ``None`` for some or all samples."""

    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        return values

    first = values[0]
    if isinstance(first, Mapping):
        keys = set().union(*(v.keys() for v in values))
        return {k: _collate_optional([v.get(k) for v in values]) for k in sorted(keys)}

    if isinstance(first, (torch.Tensor, np.ndarray)):
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) > 1:
            return values

    try:
        return default_collate(values)
    except (TypeError, RuntimeError):
        return values


def optional_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dict collator that tolerates optional / ``None`` schema fields.

    Mirrors :func:`GenCFD.utils.dataloader_builder.optional_parameters_collate`
    so this script can avoid importing the GenCFD model code (and its
    torchvision dependency) just to get a collate function.
    """

    if not batch:
        return {}
    keys = set().union(*(item.keys() for item in batch))
    return {k: _collate_optional([item.get(k) for item in batch]) for k in sorted(keys)}


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViT-FM regression on a GenCFD probe and write a prediction netCDF.",
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the Lightning .ckpt file.")
    parser.add_argument("--arch_config", type=str, required=True,
                        help="JSON config describing the ViT3 architecture.")
    parser.add_argument("--train_config", type=str, required=True,
                        help="JSON config used at training time (provides s, in_dim, out_dim, is_fourier_emb, ...).")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=("ns_shear_pp", "eul_riemann_kh_pp", "eul_kh_pp", "ns_pwc_pp", "ns_gauss_pp", "ns_sin_pp", "ns_vortex_pp", "ns_brownian_pp"),
                        help="Name of the GenCFD probe dataset to predict on.")
    parser.add_argument("--num_trajectories", type=int, default=None,
                        help="Number of distinct trajectories to evaluate (preferred over --num_samples). "
                             "Each trajectory contributes one prediction per horizon (typically 10).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap the number of (trajectory, horizon) pairs. Less natural than --num_trajectories. "
                             "If set in --ar mode, must be a multiple of the number of horizons per trajectory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write <dataset>/<split>.nc into.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="In direct mode: samples per batch. "
                             "In --ar mode: trajectories per batch (each contributes ~10 horizon samples).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gencfd_root", type=str, default=None,
                        help="Optional: path to the GenCFD   checkout. Prepended to sys.path if set.")
    parser.add_argument("--save_y_true", action="store_true", default=True,
                        help="Also store ground-truth targets as y_true for downstream diagnostics.")
    parser.add_argument("--ar", "--autoregressive", dest="ar", action="store_true", default=False,
                        help="Roll the ViT autoregressively in single-step (lead_time=0.1) jumps from "
                             "t=0 to each horizon, reusing predictions across horizons. Default: direct "
                             "predictions (one ViT forward per horizon).")
    return parser.parse_args()


def _ensure_gencfd_importable(gencfd_root: str | None) -> None:
    if gencfd_root is not None:
        gencfd_root = str(Path(gencfd_root).resolve())
        if gencfd_root not in sys.path:
            sys.path.insert(0, gencfd_root)
    '''
    try:
        import GenCFD  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "GenCFD package not importable. Pass --gencfd_root <path-to-gencfd-checkout> "
            "or add it to PYTHONPATH so 'from GenCFD.dataloader...' resolves."
        ) from exc
    '''


def build_model(arch_config: dict, train_config: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Construct ViT3 from configs, load weights, return the inner nn.Module on device."""

    from regression.ViTModulev2 import Vit3_pl

    in_dim = int(train_config["in_dim"])
    out_dim = int(train_config["out_dim"])

    train_cfg_for_ctor = dict(train_config)
    train_cfg_for_ctor["workdir"] = train_cfg_for_ctor.get("workdir", None)

    wrapper = Vit3_pl(
        in_dim=in_dim,
        out_dim=out_dim,
        loss_fn=None,
        config_train=train_cfg_for_ctor,
        config_arch=arch_config,
    )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in state:
        wrapper.load_state_dict(state["state_dict"])
    else:
        wrapper.load_state_dict(state)

    model = wrapper.model.to(device).eval()
    return model


def _t_init_zero_indices(dataset) -> tuple[list[int], int, int]:
    """Return (index list into ``dataset``, n_steps_per_traj, n_trajectories_available).

    The PDEGym+ probe dataset enumerates samples as ``trajectory * multiplier + pair_index``;
    within a trajectory's ``multiplier`` pairs, the first ``n_steps`` correspond to
    ``t_init = 0`` (one per allowed horizon).
    """
    time_indices = dataset.time_indices
    pairs_per_traj = len(time_indices)
    t0_pair_positions = [k for k, (t0, _) in enumerate(time_indices) if t0 == 0]
    n_steps = len(t0_pair_positions)
    n_traj_available = len(dataset) // pairs_per_traj
    indices = [
        traj * pairs_per_traj + k
        for traj in range(n_traj_available)
        for k in t0_pair_positions
    ]
    return indices, n_steps, n_traj_available


def build_loader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    num_samples: int | None,
    num_trajectories: int | None,
    ar: bool,
) -> tuple[DataLoader, dict, int]:
    """Construct a dataloader over the t_init=0 subset of the requested GenCFD probe.

    Returns ``(loader, metadata, n_steps_per_traj)``. In ``--ar`` mode the loader's
    ``batch_size`` equals ``batch_size_trajectories * n_steps_per_traj`` so each batch
    holds an integer number of trajectories.
    """

    from GenCFD.dataloader.metadata import get_metadata
    from GenCFD.dataloader.pdegym_plus_ood import build_probe_dataset
    from GenCFD.dataloader.schema import CFDSampleMode

    metadata = get_metadata(name=dataset_name)
    full = build_probe_dataset(
        name=dataset_name,
        metadata=metadata,
        mode=CFDSampleMode.TRAIN,
        include_dataset_id=False,
    )

    indices, n_steps, n_traj_available = _t_init_zero_indices(full)

    if num_trajectories is not None:
        if num_trajectories > n_traj_available:
            print(
                f"[predict_vit_fm_gencfd] requested {num_trajectories} trajectories, "
                f"only {n_traj_available} available; capping."
            )
        keep_traj = min(num_trajectories, n_traj_available)
        indices = indices[: keep_traj * n_steps]
    elif num_samples is not None:
        if ar and num_samples % n_steps != 0:
            raise ValueError(
                f"--ar requires --num_samples to be a multiple of {n_steps} "
                f"(steps per trajectory); got {num_samples}."
            )
        indices = indices[:num_samples]

    dataset = Subset(full, indices)

    if ar:
        effective_bs = max(int(batch_size), 1) * n_steps
    else:
        effective_bs = max(int(batch_size), 1)

    loader = DataLoader(
        dataset=dataset,
        batch_size=effective_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False,
        collate_fn=optional_collate,
        drop_last=ar,  # AR needs whole trajectories per batch
    )
    return loader, metadata, n_steps


def _broadcast_mask(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Reshape an ``(N, C)`` channel mask to ``(N, C, 1, ..., 1)`` for broadcasting."""
    if mask.ndim < target_ndim:
        view = (*mask.shape, *([1] * (target_ndim - mask.ndim)))
        return mask.view(view)
    return mask


def run_direct(model, loader, device, dtype):
    """One ViT forward per (trajectory, horizon) sample."""

    xs, y_preds, y_trues, masks, lts, dids, tis, tfs = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict[direct]"):
            x = batch["x"].to(device, non_blocking=True).to(dtype)
            y = batch["y"].to(device, non_blocking=True).to(dtype)
            mask = batch["mask"].to(device, non_blocking=True).to(dtype)
            lead_time = batch["lead_time"].to(device, non_blocking=True).to(torch.float32)

            y_pred = model(x, lead_time)
            mask_bc = _broadcast_mask(mask, y_pred.ndim)
            y_pred = y_pred * mask_bc

            xs.append(x.detach().cpu())
            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y.detach().cpu())
            masks.append(mask.detach().cpu())
            lts.append(lead_time.detach().cpu())
            dids.append(batch["dataset_id"].detach().cpu())
            tis.append(batch["t_init"].detach().cpu())
            tfs.append(batch["t_final"].detach().cpu())

    return xs, y_preds, y_trues, masks, lts, dids, tis, tfs


def run_autoregressive(model, loader, device, dtype, n_steps: int):
    """Per-trajectory ``n_steps``-step AR rollout with lead_time = 1/n_steps each.

    Saved schema (per (trajectory, horizon) sample):
        x         : initial condition at t = 0
        y_pred    : AR prediction at step k (= horizon (k+1)/n_steps)
        y_true    : dataset target at the same horizon
        lead_time : full horizon from 0 (not the single-step 1/n_steps)
        t_init    : 0
        t_final   : dataset t_final at the horizon

    Masked channels in y_pred are zeroed for saving (consistent with direct
    mode). For the AR feedback they are restored from the initial condition
    so the model keeps seeing its training distribution on the pad slots.
    """

    xs, y_preds, y_trues, masks, lts, dids, tis, tfs = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict[ar]"):
            x = batch["x"].to(device, non_blocking=True).to(dtype)
            y = batch["y"].to(device, non_blocking=True).to(dtype)
            mask = batch["mask"].to(device, non_blocking=True).to(dtype)
            lead_time_full = batch["lead_time"].to(device, non_blocking=True).to(torch.float32)

            B = x.shape[0]
            if B % n_steps != 0:
                raise RuntimeError(
                    f"AR batch with {B} samples is not a multiple of {n_steps} steps; "
                    "check that --num_samples / --num_trajectories aligns with the loader."
                )
            T = B // n_steps
            spatial = x.shape[1:]

            # Group by trajectory: (T, n_steps, C, H, W) etc.
            x_g = x.view(T, n_steps, *spatial)
            y_g = y.view(T, n_steps, *spatial)
            mask_g = mask.view(T, n_steps, *mask.shape[1:])

            # Initial condition + per-trajectory mask are the same across horizons.
            x_init = x_g[:, 0]                       # (T, C, H, W)
            mask_init = mask_g[:, 0]                 # (T, C, ...)
            mask_bc = _broadcast_mask(mask_init, x_init.ndim).expand_as(x_init)

            single_step_lt = torch.full((T,), 1.0 / n_steps, device=device, dtype=torch.float32)

            ar_preds = []
            current = x_init
            for _ in range(n_steps):
                raw = model(current, single_step_lt)
                pred_save = raw * mask_bc                              # zero pad channels in saved pred
                current = pred_save + x_init * (1.0 - mask_bc)        # restore pad token for next step
                ar_preds.append(pred_save)

            ar_preds = torch.stack(ar_preds, dim=1)                   # (T, n_steps, C, H, W)

            # Repack into the same (B = T*n_steps) order as the loader.
            x_saved = x_init.unsqueeze(1).expand(T, n_steps, *spatial).reshape(B, *spatial)

            xs.append(x_saved.detach().cpu())
            y_preds.append(ar_preds.reshape(B, *spatial).detach().cpu())
            y_trues.append(y.detach().cpu())
            masks.append(mask.detach().cpu())
            lts.append(lead_time_full.detach().cpu())
            dids.append(batch["dataset_id"].detach().cpu())
            tis.append(batch["t_init"].detach().cpu())
            tfs.append(batch["t_final"].detach().cpu())

    return xs, y_preds, y_trues, masks, lts, dids, tis, tfs


def _stack_batched(items: list[torch.Tensor]) -> np.ndarray:
    return torch.cat(items, dim=0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    _ensure_gencfd_importable(args.gencfd_root)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32

    arch_config = _load_json(args.arch_config)
    train_config = _load_json(args.train_config)

    model = build_model(arch_config, train_config, args.ckpt, device)
    loader, metadata, n_steps = build_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        num_trajectories=args.num_trajectories,
        ar=args.ar,
    )

    mode_tag = "ar" if args.ar else "direct"
    print(
        f"[predict_vit_fm_gencfd] mode={mode_tag}  dataset={args.dataset}  "
        f"n_steps_per_traj={n_steps}  samples={len(loader.dataset)}  "
        f"batch_size_arg={args.batch_size}  loader_batch={loader.batch_size}  device={device}"
    )
    print(f"[predict_vit_fm_gencfd] metadata={metadata}")

    if args.ar:
        results = run_autoregressive(model, loader, device, dtype, n_steps)
    else:
        results = run_direct(model, loader, device, dtype)

    xs, y_preds, y_trues, masks, lts, dids, tis, tfs = results

    x_np = _stack_batched(xs)
    y_pred_np = _stack_batched(y_preds)
    y_true_np = _stack_batched(y_trues) if args.save_y_true else None
    mask_np = _stack_batched(masks)
    # Channel-wise mask: reshape (N, C) -> (N, C, 1, 1) so it broadcasts to
    # (N, C, H, W) -- needed by GenCFD's _is_mask_compatible check.
    if mask_np.ndim == 2 and mask_np.shape[1] == x_np.shape[1]:
        mask_np = mask_np.reshape(*mask_np.shape, *([1] * (x_np.ndim - 2)))
    lead_time_np = _stack_batched(lts)
    dataset_id_np = _stack_batched(dids)
    t_init_np = _stack_batched(tis)
    t_final_np = _stack_batched(tfs)

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    split = str(metadata.get("split", "test"))
    fname = f"{split}_ar.nc" if args.ar else f"{split}.nc"
    out_path = out_dir / fname

    from GenCFD.eval.prediction_io import write_prediction_file

    attrs = {
        "source_dataset": args.dataset,
        "source_split": split,
        "prediction_mode": mode_tag,
        "n_steps_per_traj": int(n_steps),
        "ckpt": os.path.abspath(args.ckpt),
        "arch_config": os.path.abspath(args.arch_config),
        "train_config": os.path.abspath(args.train_config),
    }

    write_prediction_file(
        path=out_path,
        x=x_np,
        y_pred=y_pred_np,
        mask=mask_np,
        lead_time=lead_time_np.astype(np.float32),
        t_init=t_init_np.astype(np.int64),
        t_final=t_final_np.astype(np.int64),
        dataset_id=dataset_id_np.astype(np.int64),
        y_true=y_true_np,
        attrs=attrs,
    )
    print(
        f"[predict_vit_fm_gencfd] wrote {out_path}  ({x_np.shape[0]} samples, "
        f"x.shape={x_np.shape[1:]}, y_pred.shape={y_pred_np.shape[1:]})"
    )


if __name__ == "__main__":
    main()
