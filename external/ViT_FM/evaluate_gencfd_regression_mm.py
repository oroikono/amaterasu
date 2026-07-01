"""
Evaluate 3D regression models in a micro-macro setting, using the gencfd
ConditionalShearLayer3D / ConditionalTaylorGreen3D datasets, with the same
distributional metrics that gencfd's diffusion eval reports.

The regression model is treated like a diffusion sampler: for each of the
N=num_macro x num_micro perturbed initial conditions u0 from a gencfd
conditional dataset, we autoregressively roll out to t_final and take the
final prediction as the "sample". We then accumulate per-pixel ensemble
mean/std and 1000 random spatial points into a StatsRecorder, and reuse
gencfd's summarize_metric_results + spectrum_from_physical_Linf_per_sample.

Resolution handling: gencfd Conditional* data is at 64^3, the regression
model expects 32^3. We downsample IC 64->32 (nearest), run the rollout at
32^3, then upsample prediction 32->64 (nearest) before metric computation.
If data_res == target_s, both resizes become no-ops.

Strides that don't divide total_steps_stride1 cleanly are skipped (we never
silently round). With t_final * REAL_DT_RATIO = 50 finetune-dt steps for
both datasets, strides 1, 2, 5, 10, ... are accepted; stride 4 is not.

Plan: /cluster/work/math/braonic/.claude/plans/now-i-want-to-enumerated-cupcake.md
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference_rollout3d import (
    load_model,
    get_dataset_constants,
    compute_const_dim,
    build_groups,
)
from utils.utils_data import load_data

sys.path.insert(0, "/cluster/home/braonic/gencfd")
from GenCFD.dataloader.fluid_flows_3d import (  # noqa: E402
    ConditionalShearLayer3D,
    ConditionalTaylorGreen3D,
)
from GenCFD.dataloader.metadata import (  # noqa: E402
    ConditionalShearLayer3D_Metadata,
    ConditionalTaylorGreen3D_Metadata,
)
from GenCFD.eval.metrics.stats_recorder import StatsRecorder  # noqa: E402
from GenCFD.utils.eval_utils import (  # noqa: E402
    summarize_metric_results,
    spectrum_from_physical_Linf_per_sample,
)


# (dataset_class, metadata, t_final, REAL_DT_RATIO)
DATASET_REGISTRY = {
    "ConditionalShearLayer3D":  (ConditionalShearLayer3D,  ConditionalShearLayer3D_Metadata,  4, 12.5),
    "ConditionalTaylorGreen3D": (ConditionalTaylorGreen3D, ConditionalTaylorGreen3D_Metadata, 5, 10.0),
}


def _resize3d_nearest(x, target):
    """Nearest-neighbor resize a 5D tensor (B, C, X, Y, Z) to (target,)*3.
    No-op if spatial dims already match."""
    if (x.shape[-1] == target and x.shape[-2] == target
            and x.shape[-3] == target):
        return x
    return F.interpolate(x, size=(target, target, target), mode="nearest")


def _to_chan_view(arr, none_default, length=3):
    """Turn a 1D channel-stat array (possibly None) into a (1, C, 1, 1, 1)
    float32 tensor. If `arr is None`, fill with `none_default`."""
    if arr is None:
        a = np.full(length, float(none_default), dtype=np.float32)
    else:
        a = np.asarray(arr, dtype=np.float32)
    return torch.tensor(a).view(1, -1, 1, 1, 1)


def build_dataset(name, num_macro, num_micro):
    cls, meta, _, _ = DATASET_REGISTRY[name]
    return cls(metadata=meta,
               macro_perturbations=int(num_macro),
               micro_perturbations=int(num_micro))


def _write_paraview_nc(out_path, vol_uvw):
    """Write a (3, Nx, Ny, Nz) velocity volume to a ParaView-friendly NetCDF4.
    Mirrors the schema used in visualization/rollout3d_analysis.ipynb so the
    same ParaView state files apply (U, V, W scalar volumes addressed by
    uniform x, y, z coordinates in [0, 1))."""
    vol_uvw = np.asarray(vol_uvw, dtype=np.float32)
    assert vol_uvw.ndim == 4 and vol_uvw.shape[0] == 3, (
        f"expected (3, Nx, Ny, Nz), got {vol_uvw.shape}"
    )
    _, Nx, Ny, Nz = vol_uvw.shape

    x = np.linspace(0.0, 1.0, Nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, Ny, endpoint=False, dtype=np.float32)
    z = np.linspace(0.0, 1.0, Nz, endpoint=False, dtype=np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w", format="NETCDF4") as ncf:
        ncf.createDimension("x", Nx)
        ncf.createDimension("y", Ny)
        ncf.createDimension("z", Nz)

        x_var = ncf.createVariable("x", "f4", ("x",))
        y_var = ncf.createVariable("y", "f4", ("y",))
        z_var = ncf.createVariable("z", "f4", ("z",))
        x_var[:] = x; y_var[:] = y; z_var[:] = z
        x_var.axis = "X"; y_var.axis = "Y"; z_var.axis = "Z"
        x_var.units = "m"; y_var.units = "m"; z_var.units = "m"

        U_var = ncf.createVariable("U", "f4", ("x", "y", "z"))
        V_var = ncf.createVariable("V", "f4", ("x", "y", "z"))
        W_var = ncf.createVariable("W", "f4", ("x", "y", "z"))
        U_var[:] = vol_uvw[0]
        V_var[:] = vol_uvw[1]
        W_var[:] = vol_uvw[2]
    return out_path


def _collate_skip_none(batch):
    """Stack tensor entries across a batch; pass None entries through as None.
    gencfd Conditional* datasets return dicts with parameters/spectral_data=None,
    which the default torch collate can't handle."""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if any(v is None for v in vals):
            out[k] = None
        else:
            out[k] = torch.stack(vals, dim=0)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Micro-macro eval of regression models using gencfd datasets")
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON config (see configs/inference_regression/)")
    p.add_argument("--num_macro", type=int, default=None,
                   help="Override config.num_macro (useful for smoke tests)")
    p.add_argument("--num_micro", type=int, default=None,
                   help="Override config.num_micro")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override config.batch_size")
    p.add_argument("--save_samples", type=int, default=0,
                   help=("If > 0, save the first N samples per stride as "
                         "ParaView NetCDF (pred + GT, with U/V/W scalar "
                         "volumes). Files land in save_dir/samples/."))
    p.add_argument("--noise_std", type=float, default=0.0,
                   help=("If > 0, inject per-voxel multiplicative Gaussian "
                         "noise at every AR rollout step: pred[:, active] *= "
                         "(1 + N(0, noise_std)). Helps prevent ensemble "
                         "collapse in TG3D. Outputs land in a sibling dir "
                         "with the _noise<std> suffix."))
    p.add_argument("--eval_at_model_res", action="store_true",
                   help=("If set, downsample both u0 and target to the "
                         "model's native resolution at __getitem__ time and "
                         "compute every metric at that resolution (no "
                         "upsampling back to 64^3). Effectively assumes the "
                         "GT lives at 32^3 from the start. Outputs land in "
                         "a sibling dir with the _at_model_res suffix."))
    p.add_argument("--diag", type=str, default="none",
                   choices=["none", "perfect_pred", "resample_only"],
                   help=("Diagnostic mode. "
                         "'perfect_pred' skips the model and passes "
                         "target_phys directly as the prediction (verifies "
                         "the metric/denorm pipeline, expected all zeros). "
                         "'resample_only' downsamples target_phys 64->32 "
                         "and upsamples back 32->64 using the same nearest "
                         "interpolation as the real pipeline, then uses "
                         "that as the prediction (isolates the resampling "
                         "error from the model error). Outputs go to a "
                         "sibling dir with the _<diag> suffix."))
    return p.parse_args()


@torch.no_grad()
def run_eval_one_stride(*, model, ds, cfg, stride, num_steps, time_emb,
                        mean_reg, std_reg, const_idx, active_idx,
                        gencfd_mean_in, gencfd_std_in,
                        gencfd_mean_out, gencfd_std_out,
                        data_res, target_s,
                        save_dir, device, dtype,
                        diag="none", eval_at_model_res=False,
                        save_samples=0, noise_std=0.0):
    batch_size = int(cfg["batch_size"])
    n_total_dataset = len(ds)
    n_total = (n_total_dataset // batch_size) * batch_size
    if n_total == 0:
        raise ValueError(
            f"batch_size={batch_size} > dataset size {n_total_dataset}")
    if n_total < n_total_dataset:
        warnings.warn(
            f"Dropping {n_total_dataset - n_total} trailing samples to keep "
            f"batches uniform (batch_size={batch_size}).")

    # Evaluation grid: model resolution when eval_at_model_res is set, else
    # the gencfd dataset's native resolution. Every metric is computed at
    # eval_res in physical units.
    eval_res = target_s if eval_at_model_res else data_res

    stats = StatsRecorder(
        batch_size=batch_size,
        ndim=3,
        channels=len(active_idx),
        data_shape=(len(active_idx), eval_res, eval_res, eval_res),
        monte_carlo_samples=n_total,
        num_samples=1000,
        device=device,
        world_size=1,
    )

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        drop_last=True,
        collate_fn=_collate_skip_none,
    )

    active_idx_t = torch.tensor(active_idx, dtype=torch.long, device=device)
    const_idx_t = (torch.tensor(const_idx, dtype=torch.long, device=device)
                   if len(const_idx) > 0 else None)

    save_dir.mkdir(parents=True, exist_ok=True)

    U_gen_chunks = []
    U_gt_chunks = []
    pushed = 0
    pbar = tqdm(total=n_total, desc=f"stride={stride}", leave=False)
    for batch in dl:
        if pushed >= n_total:
            break
        u0_norm = batch["initial_cond"].to(device, dtype)
        target_norm = batch["target_cond"].to(device, dtype)
        B = u0_norm.shape[0]

        # gencfd's ConditionalIncompressibleFlows3D.__getitem__ permute(3,2,1,0)
        # leaves the spatial axes in (z, y, x) order — the reverse of the
        # (x, y, z) order both regression-model training and inference_rollout3d
        # use. Undo it here so the model sees the same axis convention it was
        # trained on. For (B, C, z, y, x) -> (B, C, x, y, z) we permute axes
        # (0, 1, 4, 3, 2). This matters for anisotropic flows like shear; for
        # near-isotropic ones (Taylor-Green) the difference is mostly invisible.
        u0_norm = u0_norm.permute(0, 1, 4, 3, 2).contiguous()
        target_norm = target_norm.permute(0, 1, 4, 3, 2).contiguous()

        # 1) undo gencfd normalization -> physical units (3 channels, data_res)
        u0_phys = u0_norm * gencfd_std_in + gencfd_mean_in
        target_phys = target_norm * gencfd_std_out + gencfd_mean_out

        # If eval_at_model_res, immediately collapse the GT and the IC to the
        # evaluation grid (= target_s). The model then operates at its native
        # resolution and no upsampling step is needed. With this flag off,
        # eval_res == data_res, the next two calls are no-ops, and the rest
        # of the pipeline runs as before (64 -> 32 -> rollout -> 32 -> 64).
        u0_phys = _resize3d_nearest(u0_phys, eval_res)
        target_phys = _resize3d_nearest(target_phys, eval_res)

        if diag == "perfect_pred":
            # Sanity diagnostic: bypass the model entirely and treat the GT
            # itself as the prediction. If the metric/denorm pipeline is
            # bug-free, every metric should be ~ 0 (rel L2, abs L2, Wasserstein).
            pred_phys_uvw = target_phys.detach().clone()
        elif diag == "resample_only":
            # Isolates the resampling chain target_phys (eval_res) -> 32 -> eval_res.
            # When eval_at_model_res is on (eval_res == target_s) this is a no-op
            # and the resulting metrics are identical to perfect_pred.
            target_phys_lo = _resize3d_nearest(target_phys, target_s)
            pred_phys_uvw = _resize3d_nearest(target_phys_lo, eval_res)
        else:
            # 2) pad to 5 channels [rho=1, u, v, w, P=0] in physical units
            x_phys_5 = torch.zeros(
                (B, 5, eval_res, eval_res, eval_res), device=device, dtype=dtype)
            x_phys_5[:, 0] = 1.0
            x_phys_5[:, 1:4] = u0_phys
            # P (index 4) stays 0

            # 3) downsample to model resolution (no-op when eval_res == target_s)
            x_phys_5_lo = _resize3d_nearest(x_phys_5, target_s)

            # 4) apply regression-model normalization
            x_norm = (x_phys_5_lo - mean_reg) / std_reg
            x_init_norm = x_norm.clone()

            # 5) autoregressive rollout for num_steps
            t_batch = torch.full((B,), time_emb, device=device, dtype=dtype)
            for _ in range(num_steps):
                pred = model(x_norm, t_batch)
                if const_idx_t is not None:
                    pred[:, const_idx_t] = x_init_norm[:, const_idx_t]
                if noise_std > 0.0:
                    noise = torch.randn_like(pred[:, active_idx_t]) * noise_std
                    pred[:, active_idx_t] = pred[:, active_idx_t] * (1.0 + noise)
                x_norm = pred

            # 6) undo regression normalization -> physical units
            pred_phys_5_lo = x_norm * std_reg + mean_reg
            pred_phys_uvw_lo = pred_phys_5_lo.index_select(1, active_idx_t)

            # 7) upsample target_s -> eval_res (no-op when eval_at_model_res)
            pred_phys_uvw = _resize3d_nearest(pred_phys_uvw_lo, eval_res)

        # 8) optionally write the first save_samples (pred, gt) pairs to
        # ParaView NetCDF. `pushed` is the count of samples already pushed
        # before this batch.
        if save_samples > 0 and pushed < save_samples:
            samples_dir = save_dir / "samples"
            pred_cpu = pred_phys_uvw.detach().float().cpu().numpy()
            gt_cpu = target_phys.detach().float().cpu().numpy()
            for j in range(min(B, save_samples - pushed)):
                gi = pushed + j  # global sample index across batches
                _write_paraview_nc(
                    samples_dir / f"sample{gi:04d}_pred.nc", pred_cpu[j])
                _write_paraview_nc(
                    samples_dir / f"sample{gi:04d}_gt.nc", gt_cpu[j])

        # 9) feed to StatsRecorder (both in physical units, float32)
        stats.update_step(pred_phys_uvw.float(), target_phys.float())

        U_gen_chunks.append(pred_phys_uvw.detach().float().cpu().numpy())
        U_gt_chunks.append(target_phys.detach().float().cpu().numpy())

        pushed += B
        pbar.update(B)
    pbar.close()

    extra = {
        "dataset": cfg["dataset"],
        "stride": stride,
        "num_steps": num_steps,
        "time_emb": time_emb,
        "num_macro": int(cfg["num_macro"]),
        "num_micro": int(cfg["num_micro"]),
        "n_samples": int(pushed),
        "model_path": cfg["model_path"],
        "which_data": cfg["which_data"],
        "active_channel_indices": active_idx,
        "active_variables": ["u", "v", "w"],
        "diag": diag,
        "eval_at_model_res": bool(eval_at_model_res),
        "eval_res": int(eval_res),
        "data_res": int(data_res),
        "noise_std": float(noise_std),
    }

    summarize_metric_results(stats, str(save_dir), extra=extra)

    np.savez(
        save_dir / "physical_stats.npz",
        mean_gen=stats.mean_gen.detach().cpu().numpy(),
        mean_gt=stats.mean_gt.detach().cpu().numpy(),
        std_gen=stats.std_gen.detach().cpu().numpy(),
        std_gt=stats.std_gt.detach().cpu().numpy(),
        gen_samples=stats.gen_samples.detach().cpu().numpy(),
        gt_samples=stats.gt_samples.detach().cpu().numpy(),
    )

    U_gen = np.concatenate(U_gen_chunks, axis=0)
    U_gt = np.concatenate(U_gt_chunks, axis=0)
    _, spec_gen, counts = spectrum_from_physical_Linf_per_sample(U_gen)
    _, spec_gt, _ = spectrum_from_physical_Linf_per_sample(U_gt)
    np.savez(
        save_dir / "spectral_stats.npz",
        spectrum_gen=spec_gen,
        spectrum_gt=spec_gt,
        counts=counts,
    )
    print(f"[stride={stride}] wrote outputs to {save_dir}")


def main():
    args = parse_args()
    cfg = load_data(args.config)
    if args.num_macro is not None:
        cfg["num_macro"] = int(args.num_macro)
    if args.num_micro is not None:
        cfg["num_micro"] = int(args.num_micro)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    diag = args.diag
    eval_at_model_res = bool(args.eval_at_model_res)
    save_samples = int(args.save_samples)
    noise_std = float(args.noise_std)

    device = cfg.get("device", "cuda")
    model, config_reg = load_model(cfg["model_path"], device)
    dtype = next(model.parameters()).dtype

    in_dim = int(config_reg["in_dim"])
    consts = get_dataset_constants(cfg["which_data"], in_dim)
    mean_reg = consts["mean"].view(1, -1, 1, 1, 1).to(device, dtype)
    std_reg = consts["std"].view(1, -1, 1, 1, 1).to(device, dtype)
    real_dt_ratio = float(consts["real_dt_ratio"])

    err_group = cfg["err_group"]
    err_mask_group = cfg["err_mask_group"]
    const_dim = compute_const_dim(err_group, err_mask_group)
    _ = build_groups(err_group)  # not used here, kept for parity with rollout file
    const_idx = [i for i, c in enumerate(const_dim) if c == 0.0]
    active_idx = [i for i, c in enumerate(const_dim) if c != 0.0]

    ds_cls, ds_meta, t_final, ds_real_dt_ratio = DATASET_REGISTRY[cfg["dataset"]]
    ds = build_dataset(cfg["dataset"], cfg["num_macro"], cfg["num_micro"])
    data_res = int(ds.spatial_resolution[0])
    target_s = int(config_reg.get("s_new", config_reg["s"]))

    gencfd_mean_in = _to_chan_view(ds.mean_training_input, 0.0).to(device, dtype)
    gencfd_std_in = _to_chan_view(ds.std_training_input, 1.0).to(device, dtype)
    gencfd_mean_out = _to_chan_view(ds.mean_training_output, 0.0).to(device, dtype)
    gencfd_std_out = _to_chan_view(ds.std_training_output, 1.0).to(device, dtype)

    if abs(real_dt_ratio - ds_real_dt_ratio) > 1e-9:
        warnings.warn(
            f"REAL_DT_RATIO mismatch: regression={real_dt_ratio} vs "
            f"registry={ds_real_dt_ratio}. Using regression value.")

    total_steps_stride1 = int(round((t_final - 0) * real_dt_ratio))

    print("\n=== Normalization sanity ===")
    print(f"  regression mean (5ch): {consts['mean'].cpu().numpy().tolist()}")
    print(f"  regression std  (5ch): {consts['std'].cpu().numpy().tolist()}")
    print(f"  gencfd mean_in  (3ch): {gencfd_mean_in.view(-1).cpu().float().numpy().tolist()}")
    print(f"  gencfd std_in   (3ch): {gencfd_std_in.view(-1).cpu().float().numpy().tolist()}")
    print(f"  gencfd mean_out (3ch): {gencfd_mean_out.view(-1).cpu().float().numpy().tolist()}")
    print(f"  gencfd std_out  (3ch): {gencfd_std_out.view(-1).cpu().float().numpy().tolist()}")
    print("\n=== Geometry ===")
    print(f"  data_res (gencfd): {data_res}    target_s (model): {target_s}")
    print(f"  total_steps_stride1: {total_steps_stride1} "
          f"(t_final={t_final}, real_dt_ratio={real_dt_ratio})")
    print(f"  active_idx (uvw): {active_idx}    const_idx (rho,P): {const_idx}")
    print(f"  num_macro={cfg['num_macro']}, num_micro={cfg['num_micro']}, "
          f"len(ds)={len(ds)}")

    suffix_parts = []
    if eval_at_model_res:
        suffix_parts.append("at_model_res")
    if diag != "none":
        suffix_parts.append(diag)
    if noise_std > 0.0:
        # "p" for "." so the dir name stays filesystem-friendly.
        suffix_parts.append(f"noise{noise_std:g}".replace(".", "p"))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    base_out = Path(cfg["model_path"]) / (
        f"{cfg.get('output_subdir', 'eval_gencfd')}_"
        f"{cfg['dataset']}_M{cfg['num_macro']}xm{cfg['num_micro']}{suffix}"
    )
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== diag={diag}  eval_at_model_res={eval_at_model_res}  "
          f"Writing outputs under {base_out} ===\n")

    for stride in cfg["strides"]:
        stride = int(stride)
        if total_steps_stride1 % stride != 0:
            print(f"[WARN] stride={stride} skipped: total_steps_stride1="
                  f"{total_steps_stride1} not divisible.")
            continue
        num_steps = total_steps_stride1 // stride
        time_emb = float(stride) / (20.0 * real_dt_ratio)
        print(f"\n--- stride={stride} num_steps={num_steps} "
              f"time_emb={time_emb:.6f} ---")
        save_dir = base_out / f"stride{stride}"
        run_eval_one_stride(
            model=model, ds=ds, cfg=cfg,
            stride=stride, num_steps=num_steps, time_emb=time_emb,
            mean_reg=mean_reg, std_reg=std_reg,
            const_idx=const_idx, active_idx=active_idx,
            gencfd_mean_in=gencfd_mean_in, gencfd_std_in=gencfd_std_in,
            gencfd_mean_out=gencfd_mean_out, gencfd_std_out=gencfd_std_out,
            data_res=data_res, target_s=target_s,
            save_dir=save_dir, device=device, dtype=dtype,
            diag=diag, eval_at_model_res=eval_at_model_res,
            save_samples=save_samples, noise_std=noise_std,
        )

    snapshot = {
        "config": cfg,
        "diag": diag,
        "eval_at_model_res": eval_at_model_res,
        "noise_std": noise_std,
        "eval_res": target_s if eval_at_model_res else data_res,
        "n_samples_per_stride": (len(ds) // int(cfg["batch_size"])) * int(cfg["batch_size"]),
        "total_steps_stride1": total_steps_stride1,
        "data_res": data_res,
        "target_s": target_s,
        "real_dt_ratio": real_dt_ratio,
        "active_idx": active_idx,
        "const_idx": const_idx,
    }
    with open(base_out / "config.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
