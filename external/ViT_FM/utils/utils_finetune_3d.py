import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import time
import copy

def posemb_sincos_3d(d, h, w, dim, temperature=10000, dtype=torch.float32):
    """
    Generate a 3D sine-cosine positional embedding of shape (d*h*w, dim).
    If dim is not divisible by 6, it pads with zeros to match the target dim.
    """
    div6 = dim // 6
    usable_dim = 6 * div6

    z, y, x = torch.meshgrid(
        torch.arange(d, dtype=dtype),
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
        indexing="ij"
    )
    z = z.flatten()[:, None]  # (N, 1)
    y = y.flatten()[:, None]
    x = x.flatten()[:, None]

    omega = torch.arange(div6, dtype=dtype) / max(div6 - 1, 1)
    omega = 1.0 / (temperature ** omega)  # (div6,)

    out_x = x * omega  # (N, div6)
    if usable_dim < dim:
        omega2 = torch.arange(dim-usable_dim+div6, dtype=dtype) / max(dim-usable_dim+div6 - 1, 1)
        omega2 = 1.0 / (temperature ** omega2)  # (div6 + ddiv,)
        out_add = x * omega2  # (N, div6 + ddiv)
    else:
        out_add = out_x
    out_y = y * omega
    out_z = z * omega

    pe = torch.cat([
        out_add.sin(), out_x.cos(),
        out_y.sin(), out_y.cos(),
        out_z.sin(), out_z.cos()
    ], dim=1)  # shape: (N, dim)

    """if usable_dim < dim:
        pad = torch.zeros(pe.shape[0], dim - usable_dim, dtype=dtype, device=pe.device)
        pe = torch.cat([pe, pad], dim=1)"""

    return pe  # shape: (N, dim)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)
    

class PatchEmbedding3D(nn.Module):
    def __init__(self, 
                volume_size, 
                patch_size, 
                dim, 
                channels, 
                patch_embedding_old_linear,
                pos_embedding=True, 
                project_pe_if_needed=True,
                add_const = False):
        """
        Args:
            volume_size: tuple of (D, H, W)
            patch_size: tuple of (pd, ph, pw)
            dim: embedding dimension
            channels: input channels
            pos_embedding: whether to use positional embedding
            project_pe_if_needed: linearly project positional embeddings to dim if dim is not divisible by 6
        """
        super().__init__()
        d, h, w = volume_size
        pd, ph, pw = patch_size
        assert d % pd == 0 and h % ph == 0 and w % pw == 0, "Patch size must divide volume size"

        self.grid_d = d // pd
        self.grid_h = h // ph
        self.grid_w = w // pw
        self.patch_size = (pd, ph, pw)

        num_patches = self.grid_d * self.grid_h * self.grid_w
        patch_dim = channels * pd * ph * pw

        self.rearrange = Rearrange(
            "b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)",
            pd=pd, ph=ph, pw=pw
        )
        
        if pd == 4:
            self.linear = patch_embedding_old_linear
        else:
            self.linear = nn.Sequential(
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim))
        
        self.num_patches = num_patches
        self.is_pos_embedding = pos_embedding

        if pos_embedding:
            #self.pos_embedding = posemb_sincos_3d(self.grid_d, self.grid_h, self.grid_w, dim)
            self.register_buffer("pos_embedding", posemb_sincos_3d(self.grid_d, self.grid_h, self.grid_w, dim), persistent=False)  # not saved in checkpoint
            self.pos_embedding = self.pos_embedding.half()

            if dim % 6 == 0:
                self.pe_proj = None
            elif project_pe_if_needed:
                self.pe_proj = nn.Linear(dim, dim)
                nn.init.constant_(self.pe_proj.weight, 0.0)
                nn.init.constant_(self.pe_proj.bias, 0.0)
            
            self.pos_embedding =self.pos_embedding.to("cuda")
        self.dim = dim
        self.add_const = add_const

    def forward(self, x):
        d_in, h_in, w_in = x.shape[-3:]
        x = self.linear(self.rearrange(x))

        if self.is_pos_embedding:
            pe = self.pos_embedding

            # Robustness for finetuning on resolutions different from pretraining.
            # If token count changed, regenerate sin-cos PE on-the-fly from input grid.
            if pe.shape[0] != x.shape[1]:
                pd, ph, pw = self.patch_size
                gd, gh, gw = d_in // pd, h_in // ph, w_in // pw
                pe = posemb_sincos_3d(gd, gh, gw, self.dim, dtype=torch.float32).to(x.device)
                self.pos_embedding = pe.to(self.pos_embedding.device, dtype=self.pos_embedding.dtype)
            
            if self.pe_proj is not None:
                pe = pe.to(self.pe_proj.weight.dtype) 
                if self.add_const:
                    pe = pe + self.pe_proj(pe)
                else:
                    pe = self.pe_proj(pe)
            # Move to same device/dtype as x right before use
            pe = pe.to(x.device, dtype=x.dtype, non_blocking=True)
            x = x + pe

        
        return x

class Depatchify3D(nn.Module):
    def __init__(self, 
                volume_size, 
                patch_size, 
                in_dim, 
                out_channels,
                depatchify_old):
        """
        Args:
            volume_size: Tuple (D, H, W) of the full volume
            patch_size:  Tuple (pD, pH, pW) of the patch size
            in_dim:      Dimension of each patch embedding vector (i.e., ViT dim)
            out_channels: Number of channels in the reconstructed volume
        """
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_channels = out_channels

        pd, ph, pw = patch_size
        patch_dim = pd * ph * pw * out_channels

        # Linear projection from transformer dim to flattened 3D patch
        if pd == 4:
            self.project = depatchify_old
        else:
            self.project = nn.Linear(in_dim, patch_dim)
        
    #def forward(self, x):
    def forward(self, x, spatial_shape=None):
        """
        Args:
            x: Tensor of shape (B, N, D), where N = num_patches, D = in_dim
            spatial_shape: Optional (D, H, W) shape from current input batch.
        Returns:
            Reconstructed volume: (B, C, D, H, W)
        """
        B, N, D = x.shape
        #Dv, H, W = self.volume_size
        pd, ph, pw = self.patch_size

        if spatial_shape is None:
            Dv, H, W = self.volume_size
        else:
            Dv, H, W = (int(spatial_shape[-3]), int(spatial_shape[-2]), int(spatial_shape[-1]))

        gd, gh, gw = Dv // pd, H // ph, W // pw
        assert N == gd * gh * gw, "Number of patches does not match volume size"

        # Project back to flattened patch voxels

        x = rearrange(
            self.project(x), "b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)",
            d=gd, h=gh, w=gw, pd=pd, ph=ph, pw=pw, c=self.out_channels
        )

        return x

def _infer_old_voxel_count(patch_dim_old: int, channels: int) -> int:
    """Recover the per-axis voxel count of the pretrained patch (assumed cubic)."""
    assert patch_dim_old % channels == 0, (
        f"patch_dim_old={patch_dim_old} not divisible by channels={channels}; "
        "cannot infer pretrained patch grid."
    )
    n_voxels = patch_dim_old // channels
    side = round(n_voxels ** (1.0 / 3.0))
    assert side ** 3 == n_voxels, (
        f"pretrained patch ({n_voxels} voxels) is not a cubic grid; "
        "interpolate_patch_weights only supports cubic pretrained patches."
    )
    return side


def _spatial_pool_input_dim(weight: torch.Tensor,
                            channels: int,
                            old_side: int,
                            new_pd: int, new_ph: int, new_pw: int) -> torch.Tensor:
    """
    weight: (out_dim, patch_dim_old) with patch_dim_old = old_side**3 * channels,
            laid out as (pd, ph, pw, c) with c the fastest index (matches the
            einops Rearrange '(d pd) (h ph) (w pw) c -> (d h w) (pd ph pw c)').
    Returns a (out_dim, new_pd*new_ph*new_pw*channels) tensor obtained by
    adaptive_avg_pool3d over the spatial dims of the patch.
    """
    out_dim = weight.shape[0]
    W = weight.reshape(out_dim, old_side, old_side, old_side, channels)
    # Move spatial dims after channel for adaptive_avg_pool3d expecting (N, C, D, H, W).
    # Here N := out_dim, C := channels.
    W = W.permute(0, 4, 1, 2, 3).contiguous()                 # (out_dim, c, pd, ph, pw)
    W = F.adaptive_avg_pool3d(W, (new_pd, new_ph, new_pw))    # (out_dim, c, new_pd, new_ph, new_pw)
    W = W.permute(0, 2, 3, 4, 1).contiguous()                 # (out_dim, new_pd, new_ph, new_pw, c)
    return W.reshape(out_dim, new_pd * new_ph * new_pw * channels)


def _spatial_pool_output_dim(weight_or_bias: torch.Tensor,
                             channels: int,
                             old_side: int,
                             new_pd: int, new_ph: int, new_pw: int) -> torch.Tensor:
    """
    Pool the *output* axis of a depatchify Linear weight (patch_dim_old, in_dim)
    or its bias (patch_dim_old,). Same voxel layout (pd, ph, pw, c).
    """
    if weight_or_bias.dim() == 2:
        out_dim_old, in_dim = weight_or_bias.shape
        W = weight_or_bias.reshape(old_side, old_side, old_side, channels, in_dim)
        # (pd, ph, pw, c, in_dim) -> (in_dim, c, pd, ph, pw)
        W = W.permute(4, 3, 0, 1, 2).contiguous()
        W = F.adaptive_avg_pool3d(W, (new_pd, new_ph, new_pw))
        # back to (new_pd, new_ph, new_pw, c, in_dim)
        W = W.permute(2, 3, 4, 1, 0).contiguous()
        return W.reshape(new_pd * new_ph * new_pw * channels, in_dim)
    else:
        b = weight_or_bias.reshape(old_side, old_side, old_side, channels)
        b = b.permute(3, 0, 1, 2).unsqueeze(0).contiguous()           # (1, c, pd, ph, pw)
        b = F.adaptive_avg_pool3d(b, (new_pd, new_ph, new_pw)).squeeze(0)
        b = b.permute(1, 2, 3, 0).contiguous()                        # (new_pd, new_ph, new_pw, c)
        return b.reshape(new_pd * new_ph * new_pw * channels)


@torch.no_grad()
def _transfer_patch_weights(new_embedding: 'PatchEmbedding3D',
                            old_linear: nn.Module,
                            channels: int,
                            new_patch_size):
    """
    Initialize new_embedding.linear from old_linear (a Sequential of
    LayerNorm-Linear-LayerNorm) by spatially adaptive-avg-pooling each
    parameter from the pretrained (cubic) patch grid down to new_patch_size.

    Silently no-op if `new_embedding.linear is old_linear` (e.g. pd == 4
    matched, and PatchEmbedding3D already reuses the pretrained module).
    """
    if new_embedding.linear is old_linear:
        return

    # Locate the inner Linear in both Sequentials.
    def _seq_layers(seq):
        ln_pre = seq[0] if isinstance(seq[0], nn.LayerNorm) else None
        lin = next(m for m in seq if isinstance(m, nn.Linear))
        ln_post = seq[-1] if isinstance(seq[-1], nn.LayerNorm) else None
        return ln_pre, lin, ln_post

    old_ln_pre, old_lin, old_ln_post = _seq_layers(old_linear)
    new_ln_pre, new_lin, new_ln_post = _seq_layers(new_embedding.linear)

    patch_dim_old = old_lin.in_features
    old_side = _infer_old_voxel_count(patch_dim_old, channels)
    new_pd, new_ph, new_pw = new_patch_size

    # Inner Linear weight: pool input axis (patch dim).
    new_lin.weight.data.copy_(
        _spatial_pool_input_dim(old_lin.weight.data, channels, old_side, new_pd, new_ph, new_pw)
    )
    if old_lin.bias is not None and new_lin.bias is not None:
        new_lin.bias.data.copy_(old_lin.bias.data)

    # Pre-LayerNorm (over patch_dim): pool both weight and bias spatially.
    if old_ln_pre is not None and new_ln_pre is not None:
        new_ln_pre.weight.data.copy_(
            _spatial_pool_output_dim(old_ln_pre.weight.data, channels, old_side, new_pd, new_ph, new_pw)
        )
        new_ln_pre.bias.data.copy_(
            _spatial_pool_output_dim(old_ln_pre.bias.data, channels, old_side, new_pd, new_ph, new_pw)
        )

    # Post-LayerNorm (over dim): same shape, copy directly.
    if old_ln_post is not None and new_ln_post is not None:
        new_ln_post.weight.data.copy_(old_ln_post.weight.data)
        new_ln_post.bias.data.copy_(old_ln_post.bias.data)


@torch.no_grad()
def _transfer_depatchify_weights(new_patch_to_image: 'Depatchify3D',
                                 old_project: nn.Linear,
                                 channels: int,
                                 new_patch_size):
    """
    Initialize new_patch_to_image.project (a single Linear) from old_project
    by spatially adaptive-avg-pooling its output axis from the cubic
    pretrained patch grid down to new_patch_size.

    No-op if the new module already reuses the pretrained Linear.
    """
    if new_patch_to_image.project is old_project:
        return

    new_lin = new_patch_to_image.project
    if not isinstance(new_lin, nn.Linear):
        return

    patch_dim_old = old_project.out_features
    old_side = _infer_old_voxel_count(patch_dim_old, channels)
    new_pd, new_ph, new_pw = new_patch_size

    new_lin.weight.data.copy_(
        _spatial_pool_output_dim(old_project.weight.data, channels, old_side, new_pd, new_ph, new_pw)
    )
    if old_project.bias is not None and new_lin.bias is not None:
        new_lin.bias.data.copy_(
            _spatial_pool_output_dim(old_project.bias.data, channels, old_side, new_pd, new_ph, new_pw)
        )


def initialize_FT3d(model,
                    new_in_dim,
                    new_out_dim,
                    new_s,
                    new_patch_size,
                    dims,
                    latent_channels = 16,
                    init_new = False,
                    interpolate_patch_weights = False):

    if isinstance(new_s, (list, tuple)):
        volume_size = tuple(int(v) for v in new_s)
        if len(volume_size) != 3:
            raise ValueError("new_s must be an int or a 3-tuple/list for 3D finetuning")
    else:
        volume_size = (int(new_s), int(new_s), int(new_s))

    if isinstance(new_patch_size, (list, tuple)):
        patch_size = tuple(int(v) for v in new_patch_size)
        if len(patch_size) != 3:
            raise ValueError("new_patch_size must be an int or a 3-tuple/list")
    else:
        patch_size = (int(new_patch_size), int(new_patch_size), int(new_patch_size))

    if not init_new:
        del model.model.lift
        model.model.lift = nn.Conv3d(new_in_dim, latent_channels, kernel_size=1)
        del model.model.project
        model.model.project = nn.Conv3d(latent_channels, new_out_dim, kernel_size=1)
    else:
        with torch.no_grad():
            del model.model.lift

            model.model.lift = nn.Sequential(
            nn.Conv3d(new_in_dim, 2*latent_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(2*latent_channels, latent_channels, kernel_size=1)
            )

            del model.model.project
            model.model.project =  nn.Conv3d(latent_channels, new_out_dim, kernel_size=1)

    ###del model.model.project
    ###model.model.project = nn.Conv3d(latent_channels, new_out_dim, kernel_size=1)
    
    patch_embedding_old_linear = model.model.embedding.linear
    del model.model.embedding
    model.model.embedding = PatchEmbedding3D(volume_size, 
                                            patch_size, 
                                            dims, 
                                            latent_channels, 
                                            patch_embedding_old_linear,
                                            pos_embedding=True, 
                                            project_pe_if_needed=True,
                                            add_const = init_new)
    if interpolate_patch_weights:
        _transfer_patch_weights(model.model.embedding,
                                patch_embedding_old_linear,
                                latent_channels,
                                patch_size)
    del patch_embedding_old_linear

    depatchify_old = model.model.patch_to_image.project
    del model.model.patch_to_image
    model.model.patch_to_image = Depatchify3D(volume_size = volume_size,
                                            patch_size = patch_size,
                                            in_dim = dims,
                                            out_channels = latent_channels,
                                            depatchify_old = depatchify_old)
    if interpolate_patch_weights:
        _transfer_depatchify_weights(model.model.patch_to_image,
                                     depatchify_old,
                                     latent_channels,
                                     patch_size)
    del depatchify_old
    
    model.out_dim = new_out_dim
    model.in_dim = new_in_dim
    
    return model

