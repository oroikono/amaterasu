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
# Get the directory of this script

# Add parent directory to Python import path

"""Run Inference loops to generate statistical metrics or visualize results."""
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.metrics.stats_recorder import StatsRecorder

#from GenCFD.dataloader.dataset import TrainingSetBase
#from GenCFD.utils.dataloader_builder import normalize, denormalize
#from GenCFD.utils.model_utils import reshape_jax_torch
from diffusion.metrics.eval_utils import summarize_metric_results
#from GenCFD.utils.visualization_utils import plot_2d_sample, gen_gt_plotter_3d
import os

"""def compute_spectrum(batch_u_hat_, spatial_ax = (1,2)):
        batch_size = batch_u_hat_.shape[0]
        channels = batch_u_hat_.shape[-1]
        nf = batch_u_hat_.shape[1]
        batch_u_hat_ = np.fft.fftshift(batch_u_hat_, axes=tuple(range(1, batch_u_hat_.ndim - 1)))
        u_hat_energy = np.abs(batch_u_hat_) ** 2

        # Handle both 2D and 3D cases
        spatial_dims = batch_u_hat_.shape[1:-1]
        center = [int(s / 2) for s in spatial_dims]

        if len(spatial_dims) == 2:
            box_sidex, box_sidey = spatial_dims
            px, py = np.meshgrid(np.arange(box_sidex), np.arange(box_sidey), indexing='ij')
            kk_matrix = np.maximum(np.abs(px - center[0]), np.abs(py - center[1]))
        elif len(spatial_dims) == 3:
            box_sidex, box_sidey, box_sidez = spatial_dims
            px, py, pz = np.meshgrid(np.arange(box_sidex), np.arange(box_sidey), np.arange(box_sidez), indexing='ij')
            kk_matrix = np.maximum(np.maximum(np.abs(px - center[0]), np.abs(py - center[1])), np.abs(pz - center[2]))
        else:
            raise ValueError("Data must be either 2D or 3D.")
        # Compute E_u for each unique kk value
        E_u = np.zeros((batch_size, nf // 2, channels)) + 1e-16

        for k in range(E_u.shape[1]):
            mask = (kk_matrix == k)
            expanded_mask = mask[None, ..., None]
            el = np.where(expanded_mask, u_hat_energy, 0.)
            el = el.sum(axis=spatial_ax)
            
            E_u[:, k, :] = el

        #kk_vec = np.arange(1, E_u.shape[1] + 1)

        return E_u"""
import numpy as np

import numpy as np

import numpy as np

import numpy as np

def spectrum_from_physical_Linf_per_sample(
    u,
    demean=True,
    window=None,          # None or 'hann'
    dealias_23=False      # apply 2/3-rule dealiasing in k-space
):
    """
    Per-sample L∞ (square-shell) energy spectrum from PHYSICAL-space fields.

    Args
    ----
    u : np.ndarray, shape [B, Nx, Ny, C]
        Real (or complex) physical-space fields.
    demean : bool
        Subtract spatial mean per sample/channel before FFT (removes DC bias).
    window : None | 'hann'
        Apply separable window in x,y before FFT to reduce leakage (changes energy).
    dealias_23 : bool
        Zero-out modes outside 2/3 box after FFT shift (common dealiasing).

    Returns
    -------
    kvec   : (K,) int
        Shell indices, 0..K-1 (for 128x128 → K=65).
    E_bkc  : (B, K, C) float
        Per-sample, per-shell, per-channel **average energy per mode**.
    counts : (K,) int
        Number of spatial modes in each shell.
    """
    assert u.ndim == 4, "u must be [B, Nx, Ny, C]"
    B, Nx, Ny, C = u.shape

    x = u.astype(np.complex128, copy=False)

    # De-mean to avoid a giant DC spike that can flatten tails
    if demean:
        mean_bc = x.mean(axis=(1,2), keepdims=True)  # [B,1,1,C]
        x = x - mean_bc

    # Optional window to reduce leakage (note: changes energy distribution)
    if window is not None:
        if window.lower() == 'hann':
            wx = np.hanning(Nx)[:, None]
            wy = np.hanning(Ny)[None, :]
            w2d = (wx * wy).astype(np.float64)        # [Nx, Ny]
        else:
            raise ValueError("window must be None or 'hann'")
        x = x * w2d[None, ..., None]                  # broadcast to [B,Nx,Ny,C]

    # FFT (NumPy convention: forward no scaling, inverse scaled by 1/N)
    u_hat = np.fft.fftn(x, axes=(1,2))               # [B,Nx,Ny,C]
    u_hat = np.fft.fftshift(u_hat, axes=(1,2))       # center DC

    # Optional 2/3-rule dealiasing (zero out high-k box)
    if dealias_23:
        cx, cy = Nx//2, Ny//2
        px, py = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
        kx = np.abs(px - cx)
        ky = np.abs(py - cy)
        kx_lim = int(np.floor(cx * 2/3))
        ky_lim = int(np.floor(cy * 2/3))
        keep = (kx <= kx_lim) & (ky <= ky_lim)
        u_hat = np.where(keep[None, ..., None], u_hat, 0.0)

    # Energy with Parseval-consistent scaling for NumPy's FFT convention:
    # sum |u|^2  = (1/(Nx*Ny)) * sum |û|^2  --> divide by Nx*Ny
    energy = (np.abs(u_hat) ** 2) / float(Nx * Ny)    # [B,Nx,Ny,C]

    # Build L∞ shell indices
    cx, cy = Nx//2, Ny//2
    px, py = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    kk = np.maximum(np.abs(px - cx), np.abs(py - cy)) # [Nx,Ny], ints 0..min(Nx,Ny)//2
    kk_flat = kk.ravel()
    K = int(kk_flat.max()) + 1                        # include Nyquist shell

    # Modes per shell
    counts = np.bincount(kk_flat, minlength=K).astype(np.float64)  # [K]

    # Bin per sample & channel using bincount
    E_bkc = np.zeros((B, K, C), dtype=np.float64)
    e_flat = energy.reshape(B, -1, C)                 # [B, Nx*Ny, C]
    for b in range(B):
        for c in range(C):
            E_bkc[b, :, c] = np.bincount(
                kk_flat, weights=e_flat[b, :, c], minlength=K
            )

    # Per-mode average in each shell
    E_bkc = E_bkc / (counts[None, :, None] + 1e-16)

    kvec = np.arange(K)
    return E_bkc, counts #kvec, E_bkc, counts

def compute_median_errors(U, U_gen, mask_group = None, p = 1, median = True):
    
    if mask_group is None:
        mask_group = np.ones(U_gen.shape[1])
    groups = [0]
    for i,g in enumerate(mask_group):
        groups.append(groups[-1]+g)
    num_groups = len(mask_group)


    errs_rel = np.zeros((U.shape[0],num_groups))
    errs = np.zeros((U.shape[0], num_groups))

    #print(mask_group,groups, U_gen.shape, U.shape)
    for d in range(num_groups):
        dim_in = groups[d]
        dim_out = groups[d+1]

        errs_rel[:,d] = np.mean(np.abs(U_gen[:,dim_in:dim_out] - U[:,dim_in:dim_out]), axis = (-3,-2,-1))/ np.mean(np.abs(U[:,dim_in:dim_out]) + 1e-10, axis = (-3,-2,-1))
        errs[:,d] = (np.mean(abs(U_gen[:,dim_in:dim_out] - U[:,dim_in:dim_out]) ** p,  axis = (-3,-2,-1))) ** (1 / p)
    #print(errs_rel)
    if median:
        errs = np.median(errs, axis = 0)
        errs_rel = np.median(errs_rel, axis = 0)
    return errs, errs_rel

def run(
    *,
    stats_recorder: StatsRecorder,
    monte_carlo_samples: int,
    folder_samples: str = None,
    samples_tag: str = None,
    compute_metrics: bool = True,
    pred_mask: list = None,
    return_raw_data: bool = True,
    mask_group: list = None,

) -> None:


    save_dir = folder_samples #os.path.join(cwd, "outputs" if save_dir is None else save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if compute_metrics:

        if folder_samples is not None:
            batch_size = 1
            n_iter = (monte_carlo_samples // batch_size)
            C,s1,s2 = np.load(f"{folder_samples}/sample_{0}_{samples_tag}_pred.npy").shape
            if pred_mask is not None:
                pred_mask = torch.tensor(pred_mask, dtype=torch.bool)
        # initialize the dataloader where the samples are drawn from a uniform discrete distribution
        else:
            dataloader = iter(dataloader)

        for i in range(n_iter):
            print(i, n_iter)
            if folder_samples is not None:
                u = torch.from_numpy(np.load(f"{folder_samples}/sample_{i}_{samples_tag}_out.npy")).type(torch.float32).reshape(1,-1,s1,s2)#
                #print(u.shape)
                u = u[:,pred_mask]
                gen_samples = torch.from_numpy(np.load(f"{folder_samples}/sample_{i}_{samples_tag}_pred.npy")).type(torch.float32).reshape(1,C,s1,s2)
                if gen_samples.shape[1] == pred_mask.shape[0]:
                    gen_samples = gen_samples[:,pred_mask]
            else:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                u = batch["target_cond"]
            
            if return_raw_data:
                if i == 0:
                    U = u
                    U_gen = gen_samples
                else:
                    U = np.concatenate((U, u))
                    U_gen = np.concatenate((U_gen, gen_samples))
            
            stats_recorder.update_step(gen_samples, u)

        if return_raw_data:
            errs, errs_rel = compute_median_errors(U, U_gen, mask_group = mask_group, p = 1, median = True)
            U = np.transpose(U, (0,2,3,1))
            U_gen = np.transpose(U_gen, (0,2,3,1))
            spectrum_gen, counts = spectrum_from_physical_Linf_per_sample(U_gen)
            spectrum_gt, counts_gt= spectrum_from_physical_Linf_per_sample(U)
            


        summarize_metric_results(stats_recorder, save_dir, err = errs, err_rel = errs_rel)
        np.savez(
            os.path.join(save_dir, "physical_stats.npz"),
            mean_gen = stats_recorder.mean_gen.cpu().numpy(),
            mean_gt = stats_recorder.mean_gt.cpu().numpy(),
            std_gen = stats_recorder.std_gen.cpu().numpy(),
            std_gt = stats_recorder.std_gt.cpu().numpy(),
            gen_samples = stats_recorder.gen_samples.cpu().numpy(),
            gt_samples = stats_recorder.gt_samples.cpu().numpy(),
            determinsitic_l1 = errs,
            determinsitic_relative_l1 = errs_rel
            )
        np.savez(
            os.path.join(save_dir,"spectral_stats.npz"), 
            spectrum_gen = spectrum_gen,
            spectrum_gt = spectrum_gt)


channels = 2
stats_recorder = StatsRecorder(batch_size =1,
                            ndim = 2,
                            channels = channels,
                            data_shape = (channels,128,128),
                            monte_carlo_samples = 240,
                            num_samples = 1000,
                            device = 'cpu',
                            world_size = 1)

'''
guidance = 0.0
folder_samples = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/pdegym_plus/PDEGYM_PLUS_10ep_ViTB_regression/predictions_spectral_resolver_64_steps_1_ensemble_True_renorm_{str(guidance)}_guidance_eul_riemann_curved_"
mask_group = [1,2,1]
pred_mask = [1,1,1,1,0,0,0,0,0]
monte_carlo_samples = 240
'''

guidance = 0.0
additional_tag = ""
#folder_samples = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/pdegym_plus/PDEGYM_PLUS_10ep_ViTB_regression/finetuned/ns_shear_gencfd/pdegym_plus_80k_FT_ns_shear_gencfd_non_native_80000/predictions_spectral_resolver_64_steps_1_ensemble_True_renorm_{str(guidance)}_guidance{additional_tag}_ns_shear_gencfd_"

folder_samples = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/pdegym_plus/PDEGYM_PLUS_10ep_ViTB_regression/finetuned/ns_shear_gencfd/pdegym_plus_80k_FT_ns_shear_gencfd_non_native_80000/predictions_spectral_resolver_64_steps_1_ensemble_True_renorm_0.0_guidance_MM_TRIAL_macro1_ns_shear_gencfd_mm_"
mask_group = [2]
pred_mask = [0,1,1,0,0,0,0,0,0]
pred_groups = [0,2,0,0,0,0,0,0]
monte_carlo_samples = 128

run(stats_recorder=stats_recorder,
    monte_carlo_samples=monte_carlo_samples,
    folder_samples = folder_samples,
    samples_tag = "steps_1_1.0",
    compute_metrics = True,
    pred_mask = pred_mask,
    return_raw_data = True,
    mask_group = mask_group)