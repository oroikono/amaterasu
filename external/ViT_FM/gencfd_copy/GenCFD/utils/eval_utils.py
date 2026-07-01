# Copyright 2024 The CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import json
import torch

from torch.utils.data import DataLoader
from typing import Dict, Any, Callable

from GenCFD. utils.model_utils import reshape_jax_torch

from GenCFD.eval.metrics.stats_recorder import StatsRecorder
from GenCFD.eval.metrics.probabilistic_forecast import relative_L2_norm, absolute_L2_norm
from GenCFD.eval.metrics.wasserstein import compute_average_wasserstein

Tensor = torch.Tensor


import numpy as np


def bc_spatial_to_spatial_c(U):
    """
    Convert [B, C, *spatial] → [B, *spatial, C]
    Works for 2D and 3D.
    """
    assert U.ndim in (4, 5), "Expected 2D or 3D tensor"

    B, C = U.shape[:2]
    spatial_dims = U.ndim - 2

    perm = (0,) + tuple(range(2, 2 + spatial_dims)) + (1,)
    return np.transpose(U, perm)

def spectrum_from_physical_Linf_per_sample(
    u,
    demean=True,
    window=None,          # None or 'hann'
    dealias_23=False      # apply 2/3-rule dealiasing in k-space
):
    """
    Per-sample L∞ (square/cubic-shell) energy spectrum from PHYSICAL-space fields.

    Works for 2D or 3D automatically.

    Args
    ----
    u : np.ndarray, shape [B, C, *spatial_dims]
        Real (or complex) physical-space fields.
        spatial_dims can be (Nx, Ny) or (Nx, Ny, Nz).
    demean : bool
        Subtract spatial mean per sample/channel before FFT.
    window : None | 'hann'
        Apply separable Hann window in each spatial dimension.
    dealias_23 : bool
        Apply 2/3-rule dealiasing in k-space.

    Returns
    -------
    kvec   : (K,) int
        Shell indices (L∞ norm).
    E_bkc  : (B, K, C) float
        Per-sample, per-shell, per-channel average energy per mode.
    counts : (K,) int
        Number of modes in each shell.
    """
    assert u.ndim >= 4, "u must be [B, *spatial_dims, C]"

    u = bc_spatial_to_spatial_c(u)

    B = u.shape[0]
    C = u.shape[-1]
    spatial_shape = u.shape[1:-1]
    D = len(spatial_shape)  # 2 or 3
    assert D in (2, 3), "Only 2D or 3D supported"

    x = u.astype(np.complex128, copy=False)

    # -------------------------
    # De-mean
    # -------------------------
    if demean:
        mean_axes = tuple(range(1, 1 + D))
        mean_bc = x.mean(axis=mean_axes, keepdims=True)
        x = x - mean_bc

    # -------------------------
    # Optional window
    # -------------------------
    if window is not None:
        if window.lower() != "hann":
            raise ValueError("window must be None or 'hann'")

        win = 1.0
        for n in spatial_shape:
            win = np.multiply.outer(win, np.hanning(n))
        win = win.reshape(spatial_shape)

        x = x * win[(None,) + (...,) + (None,)]

    # -------------------------
    # FFT
    # -------------------------
    fft_axes = tuple(range(1, 1 + D))
    u_hat = np.fft.fftn(x, axes=fft_axes)
    u_hat = np.fft.fftshift(u_hat, axes=fft_axes)

    # -------------------------
    # Optional 2/3 dealiasing
    # -------------------------
    if dealias_23:
        centers = [n // 2 for n in spatial_shape]
        grids = np.meshgrid(
            *[np.arange(n) for n in spatial_shape],
            indexing="ij"
        )
        k_abs = [np.abs(g - c) for g, c in zip(grids, centers)]
        limits = [int(np.floor(c * 2 / 3)) for c in centers]

        keep = np.ones(spatial_shape, dtype=bool)
        for ka, lim in zip(k_abs, limits):
            keep &= (ka <= lim)

        u_hat = np.where(keep[(None,) + (...,) + (None,)], u_hat, 0.0)

    # -------------------------
    # Energy (Parseval-consistent)
    # -------------------------
    norm = np.prod(spatial_shape)
    energy = (np.abs(u_hat) ** 2) / float(norm)

    # -------------------------
    # Build L∞ shells
    # -------------------------
    centers = [n // 2 for n in spatial_shape]
    grids = np.meshgrid(
        *[np.arange(n) for n in spatial_shape],
        indexing="ij"
    )
    kk = np.zeros(spatial_shape, dtype=int)
    for g, c in zip(grids, centers):
        kk = np.maximum(kk, np.abs(g - c))

    kk_flat = kk.ravel()
    K = kk_flat.max() + 1

    counts = np.bincount(kk_flat, minlength=K).astype(np.float64)

    # -------------------------
    # Bin energy
    # -------------------------
    E_bkc = np.zeros((B, K, C), dtype=np.float64)
    e_flat = energy.reshape(B, -1, C)

    for b in range(B):
        for c in range(C):
            E_bkc[b, :, c] = np.bincount(
                kk_flat,
                weights=e_flat[b, :, c],
                minlength=K
            )

    # Per-mode average
    E_bkc /= (counts[None, :, None] + 1e-16)

    kvec = np.arange(K)
    return kvec, E_bkc, counts

def summarize_metric_results(
    stats_recorder: StatsRecorder,
    save_dir: str,
    output_file: str = "metrics_results.json",
) -> dict:
    """
    Summarizes the evaluation metrics and stores them in a JSON file.

    Parameters:
    -----------
    stats_recorder : StatsRecorder
        Object that contains accumulated metrics for ground truth and generated data.
    save_dir : str
        Directory to store the JSON file.
    output_file : str
        Filename of the JSON file.

    Returns:
    --------
    metrics_dict : dict
        Dictionary containing the summarized metrics.
    """

    # Compute relative and absolute L2 norms
    rel_mean = relative_L2_norm(
        gen_tensor=stats_recorder.mean_gen,
        gt_tensor=stats_recorder.mean_gt,
        axis=stats_recorder.axis,
    ).tolist()

    rel_std = relative_L2_norm(
        gen_tensor=stats_recorder.std_gen,
        gt_tensor=stats_recorder.std_gt,
        axis=stats_recorder.axis,
    ).tolist()

    abs_mean = absolute_L2_norm(
        gen_tensor=stats_recorder.mean_gen,
        gt_tensor=stats_recorder.mean_gt,
        axis=stats_recorder.axis,
    ).tolist()

    abs_std = absolute_L2_norm(
        gen_tensor=stats_recorder.std_gen,
        gt_tensor=stats_recorder.std_gt,
        axis=stats_recorder.axis,
    ).tolist()

    # Monte Carlo sampled metrics
    gen_monte_carlo_samples = stats_recorder.gen_samples
    gt_monte_carlo_samples = stats_recorder.gt_samples

    # Compute average Wasserstein distances
    wasserstein_distance_torch = compute_average_wasserstein(
        num_particles=min(stats_recorder.num_samples, 1000),
        channels=stats_recorder.channels,
        gen_samples=gen_monte_carlo_samples,
        gt_samples=gt_monte_carlo_samples,
        p=1,
        method="custom",
    )

    # Construct the metrics dictionary
    metrics_dict = {
        "mean": {
            "relative": rel_mean,
            "absolute": abs_mean,
        },
        "std": {
            "relative": rel_std,
            "absolute": abs_std,
        },
        "wasserstein_distance": wasserstein_distance_torch,
    }

    # Print metrics to ensure they're logged regardless of save status
    print("Metric results:")
    print(json.dumps(metrics_dict, indent=4))

    # Save to JSON file
    save_file = os.path.join(save_dir, output_file)
    try:
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        with open(save_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics successfully saved to {save_file}")

    except Exception as e:
        print(f"Failed to save metrics to {save_file}: {e}")

    return metrics_dict
