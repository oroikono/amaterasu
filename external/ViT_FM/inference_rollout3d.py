"""
Autoregressive rollout for 3D regression models on shear3d_n32t50.

Outputs per-AR-step error curves for several strides so we can see how
error grows in time. See plan at
/cluster/work/math/braonic/.claude/plans/i-want-to-test-validated-starfish.md
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from regression.ViTModulev2 import Vit3_pl
from utils.utils_data import load_data, find_files_with_extension
from utils.utils_finetune_3d import initialize_FT3d
from utils.utils_inference import extract_meaning_variables
from dataloader.dataloader_poseidon import ShearLayer3dN32T50TimeDataset, TaylorGreenN32T50TimeDataset

# Reuse the 3D L-infty (cubic-shell) energy-spectrum routine from GenCFD.
sys.path.insert(0, "/cluster/home/braonic/gencfd")
from GenCFD.utils.eval_utils import spectrum_from_physical_Linf_per_sample  # noqa: E402


DATASET_CLASSES = {
    "shear3d_n32t50": ShearLayer3dN32T50TimeDataset,
    "tg3d_n32t50": TaylorGreenN32T50TimeDataset,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autoregressive rollout with per-step errors for 3D models"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to rollout JSON config")
    parser.add_argument("--N_samples", type=int, default=None,
                        help="Override N_samples from config (useful for smoke tests)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config")
    return parser.parse_args()


def load_model(model_path, device):
    regression_model_path = str(
        find_files_with_extension(model_path + "/model", "ckpt", [], is_pl=True)[0]
    )
    regression_config_path = str(
        find_files_with_extension(model_path, "json", ["param"])[0]
    )
    config_reg = load_data(regression_config_path)
    raw_arch = config_reg["config_arch"]
    if isinstance(raw_arch, str):
        config_arch = dict(load_data(raw_arch))
    else:
        config_arch = dict(raw_arch)

    init_new = config_reg.get("init_new", False)
    config_reg["workdir"] = None

    regression_model = Vit3_pl(
        in_dim=config_reg["in_dim"],
        out_dim=config_reg["out_dim"],
        loss_fn=None,
        config_train=config_reg,
        config_arch=config_arch,
    )

    # Mirror finetune_regression_pl3d.py: prefer s_new/patch_size_new when
    # present (the checkpoint was saved at the *finetune* geometry).
    target_s = config_reg.get("s_new", config_reg["s"])
    if config_reg.get("patch_size_new") is not None:
        patch_size = config_reg["patch_size_new"]
    elif isinstance(target_s, (list, tuple)):
        patch_size = [8, 8, 8]
    elif target_s == 64:
        patch_size = 4
    else:
        patch_size = 8
    regression_model = initialize_FT3d(
        model=regression_model,
        new_in_dim=config_reg["in_dim"],
        new_out_dim=config_reg["out_dim"],
        new_s=target_s,
        new_patch_size=patch_size,
        dims=config_arch["dims"],
        latent_channels=config_arch["latent_channels"],
        init_new=init_new,
    )

    checkpoint = torch.load(regression_model_path, map_location=device,
                            weights_only=False)
    state_dict = checkpoint["state_dict"]
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    if any("_orig_mod" in k for k in state_dict):
        state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    regression_model.load_state_dict(state_dict)

    inner = regression_model.model.to(device).eval()
    print(f"[load_model] ckpt: {regression_model_path}")
    return inner, config_reg


def get_dataset_constants(which_data, in_dim):
    """Instantiate the dataset class purely to grab mean, std, file path,
    REAL_DT_RATIO and the test split offset. We do not iterate the loader."""
    cls = DATASET_CLASSES[which_data]
    ds = cls(
        max_num_time_steps=8,
        time_step_size=1,
        fix_input_to_time_step=None,
        which="test",
        resolution=32,
        in_dist=True,
        in_dim=in_dim,
        out_dim=in_dim,
        num_trajectories=1,
        data_path="",
        time_input=True,
        masked_input=True,
        allowed_transitions=list(range(1, 9)),
        copy_to_local_scratch=False,
        perturb_p=False,
        time_window=8,
        train_multiplier=50,
    )
    n_total = ds.N_TIME_TOTAL
    # shear3d_n32t50 covers [0,1], TG covers [0,2] — both with 51 snapshots
    total_time = 1.0 if which_data == "shear3d_n32t50" else 2.0
    return {
        "mean": ds.mean.clone(),
        "std": ds.std.clone(),
        "file_path": ds.file_path,
        "test_start": ds.start,
        "real_dt_ratio": ds.REAL_DT_RATIO,
        "n_time_total": n_total,
        "phys_dt": total_time / (n_total - 1),
    }


def preload_trajectories(file_path, traj_start, N_samples, in_dim, resolution=32):
    """Read full (N, 51, in_dim, 32, 32, 32) trajectories with the same
    channel layout used during finetuning: [rho=1, u, v, w, P=0, zeros...]."""
    reader = nc.Dataset(file_path, "r")
    end = traj_start + N_samples
    u = reader["u"][traj_start:end, :, :, :, :]
    v = reader["v"][traj_start:end, :, :, :, :]
    w = reader["w"][traj_start:end, :, :, :, :]
    reader.close()

    N, T = u.shape[0], u.shape[1]
    traj = np.zeros((N, T, in_dim, resolution, resolution, resolution), dtype=np.float32)
    if in_dim >= 5:
        traj[:, :, 0] = 1.0
        traj[:, :, 1] = u
        traj[:, :, 2] = v
        traj[:, :, 3] = w
        traj[:, :, 4] = 0.0
    elif in_dim == 3:
        traj[:, :, 0] = u
        traj[:, :, 1] = v
        traj[:, :, 2] = w
    else:
        raise ValueError(f"Unsupported in_dim={in_dim}")
    return torch.from_numpy(traj)


def compute_const_dim(err_group, err_mask_group):
    const_dim = []
    for i, e in enumerate(err_group):
        const_dim = const_dim + e * [float(err_mask_group[i])]
    return const_dim


def build_groups(err_group):
    groups = [0]
    for g in err_group:
        groups.append(groups[-1] + g)
    return groups


def _spectrum_steps_from_fractions(num_steps, fractions):
    """Map fractions in (0,1] to AR-step indices in [1, num_steps]."""
    steps = []
    for f in fractions:
        s = int(round(float(f) * num_steps))
        s = max(1, min(num_steps, s))
        steps.append(s)
    # de-duplicate while keeping the input order
    seen = set()
    uniq = []
    for s in steps:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _resize3d(x, target):
    """Trilinear resize a 5D tensor (B, C, X, Y, Z) to (target, target, target).
    No-op if spatial dims already match."""
    if x.shape[-1] == target and x.shape[-2] == target and x.shape[-3] == target:
        return x
    return F.interpolate(x, size=(target, target, target),
                         mode="trilinear", align_corners=False)


@torch.no_grad()
def rollout_one_stride(model, traj_full, stride, time_emb_value,
                       mean_dev, std_dev, const_dim, groups, active_groups,
                       batch_size, device, dtype,
                       spectrum_fractions=None, save_samples=0,
                       model_resolution=None):
    """AR rollout at fixed stride.

    Returns a dict with:
      - errs_rel, errs_l1: (num_steps, N, num_active_groups) per-step errors
      - spectrum_steps: list of AR step indices where spectra were computed
      - spectrum_pred, spectrum_gt: (len(spectrum_steps), N, C_active, K) energy
      - kvec: (K,) shell indices
      - counts: (K,) modes per shell
      - samples_pred, samples_gt: (save_samples, num_steps+1, C_active, X, Y, Z)
        physical-space tensors for the first save_samples samples (None if 0).

    Spectra are computed channel-wise across the active channels (e.g. the 3
    velocity channels) in physical units (denormalized)."""
    N, T, _, X, Y, Z = traj_full.shape
    data_res = X
    if model_resolution is None or int(model_resolution) <= 0:
        model_res = data_res
    else:
        model_res = int(model_resolution)
    cross_res = (model_res != data_res)
    num_steps = (T - 1) // stride
    num_active = len(active_groups)
    errs_rel = np.zeros((num_steps, N, num_active), dtype=np.float32)
    errs_l1 = np.zeros((num_steps, N, num_active), dtype=np.float32)

    is_const = torch.tensor([c == 0.0 for c in const_dim], dtype=torch.bool)

    # Active-channel slice (assumes a contiguous block of active channels —
    # holds for our [rho | uvz | P] layout). Fallback to multiple slices works
    # the same way but the typical case is one block.
    active_channel_indices = []
    for d in active_groups:
        active_channel_indices.extend(range(groups[d], groups[d + 1]))
    C_active = len(active_channel_indices)
    active_idx_t = torch.tensor(active_channel_indices, dtype=torch.long)

    # Spectrum bookkeeping.
    spectrum_steps = _spectrum_steps_from_fractions(num_steps, spectrum_fractions or [])
    spec_step_to_idx = {s: i for i, s in enumerate(spectrum_steps)}
    K = max(X, Y, Z) // 2 + 1     # L-inf shells: floor(L/2)+1 shells
    spectrum_pred = (np.zeros((len(spectrum_steps), N, C_active, K), dtype=np.float64)
                     if spectrum_steps else None)
    spectrum_gt = (np.zeros((len(spectrum_steps), N, C_active, K), dtype=np.float64)
                   if spectrum_steps else None)
    kvec_out = None
    counts_out = None

    # Sample-saving bookkeeping (first save_samples global samples).
    save_samples = int(save_samples)
    save_samples = min(save_samples, N)
    if save_samples > 0:
        samples_pred = np.zeros(
            (save_samples, num_steps + 1, C_active, X, Y, Z), dtype=np.float32
        )
        samples_gt = np.zeros(
            (save_samples, num_steps + 1, C_active, X, Y, Z), dtype=np.float32
        )
        samples_gt[:, 0] = (
            traj_full[:save_samples, 0, active_idx_t].numpy().astype(np.float32)
        )
        samples_pred[:, 0] = samples_gt[:, 0]   # AR step 0 = ground truth IC
    else:
        samples_pred = None
        samples_gt = None

    for b0 in tqdm.tqdm(range(0, N, batch_size),
                       desc=f"stride={stride}", leave=False):
        b1 = min(b0 + batch_size, N)
        B = b1 - b0

        x_phys = traj_full[b0:b1, 0].to(device=device, dtype=dtype)
        x = (x_phys - mean_dev) / std_dev
        x_init_norm = x.clone()

        t_batch = torch.full((B,), time_emb_value, device=device, dtype=dtype)

        # which global indices in this batch fall under save_samples
        batch_save_local = [i for i in range(B) if (b0 + i) < save_samples]

        for n in range(1, num_steps + 1):
            if cross_res:
                x_in = _resize3d(x, model_res)
                pred_hi = model(x_in, t_batch)
                pred = _resize3d(pred_hi, data_res)
            else:
                pred = model(x, t_batch)
            if is_const.any():
                idx = torch.nonzero(is_const, as_tuple=False).flatten()
                pred[:, idx] = x_init_norm[:, idx]

            gt_phys = traj_full[b0:b1, n * stride].to(device=device, dtype=dtype)
            pred_phys = pred * std_dev + mean_dev

            # errors per active group
            for a, d in enumerate(active_groups):
                di, do = groups[d], groups[d + 1]
                diff = (pred_phys[:, di:do] - gt_phys[:, di:do]).abs()
                denom = gt_phys[:, di:do].abs().mean(dim=(1, 2, 3, 4)) + 1e-10
                rel = diff.mean(dim=(1, 2, 3, 4)) / denom
                l1 = diff.mean(dim=(1, 2, 3, 4))
                errs_rel[n - 1, b0:b1, a] = rel.detach().cpu().numpy()
                errs_l1[n - 1, b0:b1, a] = l1.detach().cpu().numpy()

            # spectra at requested AR steps (active channels, physical units)
            if n in spec_step_to_idx:
                s_idx = spec_step_to_idx[n]
                pred_active = pred_phys[:, active_idx_t].detach().cpu().numpy()
                gt_active = gt_phys[:, active_idx_t].detach().cpu().numpy()
                _kvec_p, E_p, c_p = spectrum_from_physical_Linf_per_sample(
                    pred_active, demean=True, window=None, dealias_23=False
                )
                _kvec_g, E_g, c_g = spectrum_from_physical_Linf_per_sample(
                    gt_active, demean=True, window=None, dealias_23=False
                )
                # E shape (B, K_local, C_active) → put into (1, N, C_active, K)
                K_local = E_p.shape[1]
                Kw = min(K, K_local)
                spectrum_pred[s_idx, b0:b1, :, :Kw] = (
                    E_p[:, :Kw, :].transpose(0, 2, 1)
                )
                spectrum_gt[s_idx, b0:b1, :, :Kw] = (
                    E_g[:, :Kw, :].transpose(0, 2, 1)
                )
                if kvec_out is None:
                    kvec_out = _kvec_g[:Kw].astype(np.int64)
                    counts_out = c_g[:Kw].astype(np.int64)

            # sample-trajectory capture (only for first save_samples globally)
            if batch_save_local:
                pred_act_cpu = (
                    pred_phys[:, active_idx_t].detach().cpu().numpy().astype(np.float32)
                )
                gt_act_cpu = (
                    gt_phys[:, active_idx_t].detach().cpu().numpy().astype(np.float32)
                )
                for i in batch_save_local:
                    samples_pred[b0 + i, n] = pred_act_cpu[i]
                    samples_gt[b0 + i, n] = gt_act_cpu[i]

            x = pred

    return {
        "errs_rel": errs_rel,
        "errs_l1": errs_l1,
        "spectrum_steps": spectrum_steps,
        "spectrum_pred": spectrum_pred,
        "spectrum_gt": spectrum_gt,
        "kvec": kvec_out,
        "counts": counts_out,
        "samples_pred": samples_pred,
        "samples_gt": samples_gt,
        "active_channel_indices": active_channel_indices,
    }


def git_head(cwd):
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    cfg = load_data(args.config)
    if args.N_samples is not None:
        cfg["N_samples"] = args.N_samples
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = cfg.get("device", "cuda")
    model_path = cfg["model_path"]
    which_data = cfg["which_data"]
    strides = cfg["strides"]
    N_samples = int(cfg["N_samples"])
    batch_size = int(cfg["batch_size"])
    err_group = cfg["err_group"]
    err_mask_group = cfg["err_mask_group"]
    output_subdir = cfg.get("output_subdir", "rollout_errors")
    spectrum_fractions = cfg.get("spectrum_fractions", [0.25, 0.5, 0.75, 1.0])
    save_samples = int(cfg.get("save_samples", 0))
    model_resolution = cfg.get("model_resolution", None)

    model, config_reg = load_model(model_path, device)
    dtype = next(model.parameters()).dtype

    in_dim = int(config_reg["in_dim"])
    consts = get_dataset_constants(which_data, in_dim)
    mean_dev = consts["mean"].unsqueeze(0).to(device=device, dtype=dtype)
    std_dev = consts["std"].unsqueeze(0).to(device=device, dtype=dtype)

    print(f"[main] Preloading {N_samples} test trajectories starting at "
          f"index {consts['test_start']} from {consts['file_path']}")
    traj_full = preload_trajectories(
        consts["file_path"], consts["test_start"], N_samples, in_dim
    )
    print(f"[main] Loaded trajectory tensor shape={tuple(traj_full.shape)} "
          f"(fp32 cpu, ~{traj_full.numel() * 4 / 1e9:.2f} GB)")

    const_dim = compute_const_dim(err_group, err_mask_group)
    groups = build_groups(err_group)
    active_groups = [d for d, m in enumerate(err_mask_group) if m > 0]
    print(f"[main] const_dim={const_dim} groups={groups} active_groups={active_groups}")

    variable_meaning = extract_meaning_variables(
        int(np.sum(err_group)), groups=err_group, which_data=which_data
    )
    if variable_meaning is None:
        variable_meaning = [f"g{i}" for i in range(len(err_group))]
    active_names = [variable_meaning[d] for d in active_groups]
    print(f"[main] variable_meaning={variable_meaning} active_names={active_names}")

    out_dir = Path(model_path) / f"{output_subdir}_{which_data}_{N_samples}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] Writing outputs to {out_dir}")

    summary_rows = []
    sanity_at_phys_time = {}
    phys_dt = float(consts["phys_dt"])

    for stride in strides:
        time_emb = float(stride) / (20.0 * float(consts["real_dt_ratio"]))
        print(f"\n=== stride={stride}  time_emb={time_emb:.6f}  "
              f"phys_dt_step={stride * phys_dt:.4f} ===")
        out = rollout_one_stride(
            model=model,
            traj_full=traj_full,
            stride=stride,
            time_emb_value=time_emb,
            mean_dev=mean_dev,
            std_dev=std_dev,
            const_dim=const_dim,
            groups=groups,
            active_groups=active_groups,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            spectrum_fractions=spectrum_fractions,
            save_samples=save_samples,
            model_resolution=model_resolution,
        )
        errs_rel = out["errs_rel"]
        errs_l1 = out["errs_l1"]

        np.save(out_dir / f"errors_rel_stride{stride}.npy", errs_rel)
        np.save(out_dir / f"errors_l1_stride{stride}.npy", errs_l1)

        if out["spectrum_pred"] is not None:
            np.save(out_dir / f"spectrum_pred_stride{stride}.npy",
                    out["spectrum_pred"])
            np.save(out_dir / f"spectrum_gt_stride{stride}.npy",
                    out["spectrum_gt"])
            np.save(out_dir / f"spectrum_kvec_stride{stride}.npy", out["kvec"])
            np.save(out_dir / f"spectrum_counts_stride{stride}.npy", out["counts"])
            # Save the AR-step indices and physical times for those spectra.
            spec_meta = {
                "stride": stride,
                "spectrum_steps": out["spectrum_steps"],
                "spectrum_times": [round(s * stride * phys_dt, 6)
                                   for s in out["spectrum_steps"]],
                "active_channel_indices": out["active_channel_indices"],
                "active_names": active_names,
            }
            with open(out_dir / f"spectrum_meta_stride{stride}.json", "w") as f:
                json.dump(spec_meta, f, indent=2)
            print(f"  spectra written at AR steps {out['spectrum_steps']} "
                  f"(times {spec_meta['spectrum_times']})")

        if out["samples_pred"] is not None:
            np.save(out_dir / f"samples_pred_stride{stride}.npy",
                    out["samples_pred"])
            np.save(out_dir / f"samples_gt_stride{stride}.npy",
                    out["samples_gt"])
            print(f"  saved {out['samples_pred'].shape[0]} sample "
                  f"trajectories at active channels {out['active_channel_indices']}")

        num_steps = errs_rel.shape[0]
        rel_med = np.median(errs_rel, axis=1)
        l1_med = np.median(errs_l1, axis=1)

        for n in range(num_steps):
            row = {
                "stride": stride,
                "ar_step": n + 1,
                "time": round((n + 1) * stride * phys_dt, 6),
            }
            for a, name in enumerate(active_names):
                row[f"{name}_l1_rel_med"] = float(rel_med[n, a])
                row[f"{name}_l1_med"] = float(l1_med[n, a])
            summary_rows.append(row)

        primary = active_names[0]
        print(f"  [stride={stride}] {primary} median rel-L1 by AR step:")
        for n in range(num_steps):
            print(f"    n={n+1:>3d}  t={(n+1)*stride*phys_dt:.4f}  "
                  f"rel-L1={rel_med[n, 0]:.4f}")

        for n in range(num_steps):
            t_phys = round((n + 1) * stride * phys_dt, 6)
            sanity_at_phys_time.setdefault(t_phys, {})[stride] = float(rel_med[n, 0])

    print(f"\n=== Cross-stride sanity (median rel-L1 on {active_names[0]} at matched real time) ===")
    common_times = sorted(t for t, d in sanity_at_phys_time.items()
                          if len(d) == len(strides))
    if common_times:
        print(f"  real_time | " + " | ".join(f"stride={s}" for s in strides))
        for t in common_times[:20]:
            cells = " | ".join(f"{sanity_at_phys_time[t][s]:.4f}" for s in strides)
            print(f"  {t:>9.4f} | {cells}")
    else:
        print("  (no real-times reachable by all strides)")

    import csv as _csv
    csv_path = out_dir / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[main] Wrote {csv_path}")

    snapshot = {
        "config": cfg,
        "git_head": git_head(os.path.dirname(os.path.abspath(__file__))),
        "n_time_total": int(consts["n_time_total"]),
        "real_dt_ratio": float(consts["real_dt_ratio"]),
        "phys_dt": phys_dt,
        "in_dim": in_dim,
        "test_start": int(consts["test_start"]),
        "variable_meaning": variable_meaning,
        "active_groups": active_groups,
        "active_names": active_names,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[main] Wrote {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
