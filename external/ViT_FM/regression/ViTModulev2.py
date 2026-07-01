import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from regression.EmbeddingModule import FourierEmbedding
from regression.GeneralModule_pl import GeneralModel_pl
from regression.CNOEmbeddingModule import FILM

import einops
from typing import Any, Sequence, Callable
import torch.nn.functional as F
import time

Tensor = torch.Tensor

class AdaptiveScale(nn.Module):
    def __init__(
        self,
        emb_channels: int,
        input_channels: int,
        dim: int = 2,  # 2 = image, 1 = tokens
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        device: str = "cuda"
    ):
        super(AdaptiveScale, self).__init__()

        self.emb_channels = emb_channels
        self.input_channels = input_channels
        self.dim = dim
        self.act_fun = act_fun

        self.affine = nn.Linear(
            in_features=emb_channels,
            out_features=input_channels * 2,
            device=device
        )

        torch.nn.init.zeros_(self.affine.bias)
        torch.nn.init.zeros_(self.affine.weight)

    def forward(self, x, emb):
        """
        Args:
            x: (B, N, C) for tokens OR (B, C, H, W) for images
            emb: (B, emb_channels)
        Returns:
            Rescaled tensor with same shape as x
        """
        scale_params = self.affine(self.act_fun(emb))
        scale, bias = torch.chunk(scale_params, 2, dim=-1)

        if self.dim == 1:  # Token-based tensor (B, N, C)
            scale = scale.unsqueeze(1)  # (B, 1, C)
            bias = bias.unsqueeze(1)    # (B, 1, C)
        else:  # Spatial tensor (e.g., B, C, H, W)
            scale = scale.view(scale.size(0), -1, *[1] * self.dim)  # (B, C, 1, 1, ...)
            bias = bias.view(bias.size(0), -1, *[1] * self.dim)

        #print(scale.shape, bias.shape, x.shape)
        x =  x * (scale + 1) + bias
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)



# === Multiscale Patch Embedding ===

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels, pos_embedding = True):
        super().__init__()
        #print(image_size)
        #print(patch_size)
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, "Patch size must divide image size"
        num_patches = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw

        self.rearrange = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        self.linear = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.num_patches = num_patches

        self.is_pos_embedding = pos_embedding
        if pos_embedding:
            self.pos_embedding = posemb_sincos_2d(
            h = ih // ph,
            w = iw // pw,
            dim = dim,
            ) 

    def forward(self, x):
        x = self.rearrange(x)
        x = self.linear(x)
        if self.is_pos_embedding:
            x = x + self.pos_embedding.to(x.device, dtype=x.dtype)
        return x

class Depatchify(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_channels):
        """
        Args:
            image_size: Tuple (H, W) of the full image size
            patch_size: Tuple (pH, pW) of the patch size
            in_dim:     Dimension of each patch embedding vector (i.e., ViT dim)
            out_channels: Number of channels in the reconstructed image
        """
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.in_dim = in_dim
        self.out_channels = out_channels

        ph, pw = self.patch_size
        patch_dim = ph * pw * out_channels

        # Project back from transformer dim to patch pixels
        self.project = nn.Linear(in_dim, patch_dim)

    #def forward(self, x):
    def forward(self, x, spatial_shape=None):
        """
        Args:
            x: (B, N, D), where N = num_patches, D = in_dim
        Returns:
            Reconstructed image: (B, C, H, W)
        """
        B, N, D = x.shape
        ph, pw = self.patch_size

        if spatial_shape is None:
            H, W = self.image_size
        else:
            H, W = int(spatial_shape[-2]), int(spatial_shape[-1])

        gh, gw = H // ph, W // pw
        assert N == gh * gw, "Number of patches does not match image size"
            
        # Project patch embedding to flattened pixels
        x = self.project(x)  # (B, N, patch_dim)
        x = rearrange(x, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                      h=gh, w=gw, ph=ph, pw=pw, c=self.out_channels)
        return x

# === Cross-Attention Fusion ===

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_small, dim_large):
        super().__init__()
        self.query_proj = nn.Linear(dim_large, dim_small)
        self.key_proj = nn.Linear(dim_small, dim_small)
        self.value_proj = nn.Linear(dim_small, dim_small)
        self.out_proj = nn.Linear(dim_small, dim_large)
        self.scale = dim_small ** -0.5

    def forward(self, x_small, x_large):
        # x_small: (B, N1, C1), x_large: (B, N2, C2)
        q = self.query_proj(x_large)
        k = self.key_proj(x_small)
        v = self.value_proj(x_small)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        fused = torch.matmul(attn, v)
        fused = self.out_proj(fused)
        return fused

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, emb_channels=None, is_fourier_embedding = True, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        if is_fourier_embedding:
            self.adapt = AdaptiveScale(emb_channels, dim, dim=1, device=device) if emb_channels else None
        else:
            self.adapt = FILM(dim) if emb_channels else None

    def forward(self, x, emb=None):
        x = self.norm(x)
        x = self.net(x)
        if self.adapt and emb is not None:
            x = self.adapt(x, emb)
        return x

class Attention(nn.Module):
    #def __init__(self, dim, heads=8, dim_head=64, emb_channels=None, is_fourier_embedding = True, device="cuda"):
    def __init__(self, dim, heads=8, dim_head=64, emb_channels=None, is_fourier_embedding = True, device="cuda", use_sdpa: bool = True, attention_chunk_size: int = 0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.attention_chunk_size = attention_chunk_size
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = nn.Softmax(dim=-1)
        
        if is_fourier_embedding:
            self.adapt = AdaptiveScale(emb_channels, dim, dim = 1, device=device) if emb_channels else None
        else:
            self.adapt = FILM(dim) if emb_channels else None
    
    def forward(self, x, emb=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #attn = self.attend(dots)
        #del dots, q, k
        #out = torch.matmul(attn, v)
        #del attn, v
        if self.use_sdpa:
            if self.attention_chunk_size and q.shape[-2] > self.attention_chunk_size:
                out_chunks = []
                for q_chunk in q.split(self.attention_chunk_size, dim=-2):
                    out_chunks.append(F.scaled_dot_product_attention(q_chunk, k, v, dropout_p=0.0, scale=self.scale))
                out = torch.cat(out_chunks, dim=-2)
            else:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
            del q, k, v
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            del dots, q, k
            out = torch.matmul(attn, v)
            del attn, v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.adapt and emb is not None:
            out = self.adapt(out, emb)
        return out

class TransformerBlock(nn.Module):
    #def __init__(self, dim, depth, heads, dim_head, mlp_dim, emb_channels=None, is_fourier_embedding = True, device="cuda"):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, emb_channels=None, is_fourier_embedding = True, device="cuda", use_sdpa: bool = True, attention_chunk_size: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                #Attention(dim, heads, dim_head, emb_channels, is_fourier_embedding, device),
                Attention(dim, heads, dim_head, emb_channels, is_fourier_embedding, device, use_sdpa=use_sdpa, attention_chunk_size=attention_chunk_size),
                FeedForward(dim, mlp_dim, emb_channels, is_fourier_embedding, device)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, emb=None):
        for attn, ff in self.layers:
            x = attn(x, emb) + x
            x = ff(x, emb) + x
        x = self.norm(x)
        return x

class ViT3(nn.Module):
    def __init__(self, *, 
                image_size=(128, 128), 
                in_channels=4,
                out_channels=4,
                latent_channels=128, 
                patch_sizes=8,
                dims=512, 
                depth=10, 
                heads=8, 
                dim_head=128, 
                mlp_dim=1024,
                emb_channels=128,
                is_fourier_embedding = True,
                rescale_time = False,
                use_sdpa: bool = True,
                attention_chunk_size: int = 0):
        super().__init__()

        if is_fourier_embedding:
            self.fourier_embedding = FourierEmbedding(dims=emb_channels)
        else:
            self.fourier_embedding = nn.Identity()
        
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        )

        print(patch_sizes, dims)
        self.embedding =  PatchEmbedding(image_size, patch_sizes, dims, latent_channels, pos_embedding=True)
        #self.transformers = TransformerBlock(dims, depth, heads, dim_head, mlp_dim, emb_channels=emb_channels, is_fourier_embedding = is_fourier_embedding)
        self.transformers = TransformerBlock(dims, depth, heads, dim_head, mlp_dim, emb_channels=emb_channels, is_fourier_embedding = is_fourier_embedding, use_sdpa=use_sdpa, attention_chunk_size=attention_chunk_size)
        self.patch_to_image = Depatchify(image_size=image_size, patch_size=patch_sizes, in_dim=dims, out_channels=latent_channels)

        self.project = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
        )

        self.rescale_time = rescale_time

    def _rescale_time(self):
        if self.rescale_time:
            self.time_shift = nn.Parameter(torch.tensor(0.0))
            self.time_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t):
        x = self.lift(x)
        spatial_shape = x.shape[-3:] if x.ndim == 5 else x.shape[-2:]

        if self.rescale_time and t is not None:
            t = self.time_scale * t + self.time_shift
        emb = self.fourier_embedding(t)

        x = self.embedding(x)
        x = self.transformers(x, emb)
        del emb

        #x = self.patch_to_image(x)
        x = self.patch_to_image(x, spatial_shape=spatial_shape)
        x = self.project(x)
        return x

class MultiScaleViT3(nn.Module):
    def __init__(self, *, 
                image_size=(128, 128), 
                in_channels=4,
                out_channels=4,
                latent_channels=128, 
                patch_sizes=(4, 8, 16),
                dims=(192, 384, 768), 
                depth=10, 
                heads=8, 
                dim_head=128, 
                mlp_dim=1024,
                emb_channels=128):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(dims=emb_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        )
        print(patch_sizes, dims)
        
        self.embed_branches = nn.ModuleList([
            PatchEmbedding(image_size, ps, dim, latent_channels, pos_embedding=True)
            for ps, dim in zip(patch_sizes, dims)
        ])

        self.transformers = nn.ModuleList([
            TransformerBlock(dim, depth, heads, dim_head, mlp_dim, emb_channels=emb_channels)
            for dim in dims
        ])

        self.fuse_small = CrossAttentionFusion(dims[1], dims[0])
        self.fuse_large = CrossAttentionFusion(dims[2], dims[0])

        self.patch_to_image = Depatchify(image_size=image_size, patch_size=patch_sizes[0], in_dim=dims[0], out_channels=latent_channels)
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
        )
        self.project = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
)

    def forward(self, x, t):
        x = self.lift(x)
        emb = self.fourier_embedding(t)

        x_s = self.embed_branches[0](x)
        x_m = self.embed_branches[1](x)
        x_l = self.embed_branches[2](x)

        x_s = self.transformers[0](x_s, emb)
        x_m = self.transformers[1](x_m, emb)
        x_l = self.transformers[2](x_l, emb)

        x_s = x_s + self.fuse_small(x_m, x_s)
        x_s = x_s + self.fuse_large(x_l, x_s)

        x_s = self.patch_to_image(x_s)
        x_s = self.project(x_s)
        return x_s

class MultiScaleViT2(nn.Module):
    def __init__(self, *, 
                image_size=(128, 128), 
                in_channels=4,
                out_channels=4,
                latent_channels=128, 
                patch_sizes=(8, 16),
                dims=(384, 768), 
                depth=10, 
                heads=8, 
                dim_head=128, 
                mlp_dim=1024,
                emb_channels=128):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(dims=emb_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        )

        self.embed_branches = nn.ModuleList([
            PatchEmbedding(image_size, ps, dim, latent_channels, pos_embedding=True)
            for ps, dim in zip(patch_sizes, dims)
        ])

        self.transformers = nn.ModuleList([
            TransformerBlock(dim, depth, heads, dim_head, mlp_dim, emb_channels=emb_channels)
            for dim in dims
        ])

        self.fuse = CrossAttentionFusion(dims[1], dims[0])

        self.patch_to_image = Depatchify(image_size=image_size, patch_size=patch_sizes[0], in_dim=dims[0], out_channels=latent_channels)
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
        )
        self.project = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
)

    def forward(self, x, t):
        x = self.lift(x)
        emb = self.fourier_embedding(t)

        x_s = self.embed_branches[0](x)
        x_l = self.embed_branches[1](x)

        x_s = self.transformers[0](x_s, emb)
        x_l = self.transformers[1](x_l, emb)

        x_s = x_s + self.fuse(x_l, x_s)

        x_s = self.patch_to_image(x_s)
        x_s = self.project(x_s)
        return x_s

class MultiVit3_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()):
        super().__init__(in_dim, out_dim, config_train)
        self.loss_fn = loss_fn
        self.model = MultiScaleViT3(
            in_channels=in_dim,
            out_channels=out_dim,
            latent_channels=config_arch["latent_channels"],
            image_size=(config_train["s"], config_train["s"]),
            patch_sizes=config_arch["patch_sizes"],
            dims=config_arch["dims"],
            depth=config_arch["depth"],
            heads=config_arch["heads"],
            dim_head=config_arch["dim_head"],
            mlp_dim=config_arch["mlp_dim"],
            emb_channels= config_arch["emb_channels"]
        )

class MultiVit2_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()):
        super().__init__(in_dim, out_dim, config_train)
        self.loss_fn = loss_fn
        self.model = MultiScaleViT2(
            in_channels=in_dim,
            out_channels=out_dim,
            latent_channels=config_arch["latent_channels"],
            image_size=(config_train["s"], config_train["s"]),
            patch_sizes=config_arch["patch_sizes"],
            dims=config_arch["dims"],
            depth=config_arch["depth"],
            heads=config_arch["heads"],
            dim_head=config_arch["dim_head"],
            mlp_dim=config_arch["mlp_dim"],
            emb_channels= config_arch["emb_channels"]
        )

class Vit3_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()):
        super().__init__(in_dim, out_dim, config_train)
        self.loss_fn = loss_fn

        if "is_fourier_emb" in config_train:
            is_fourier_embedding = config_train["is_fourier_emb"]
        else:
            is_fourier_embedding = True

        if "rescale_time" in config_train:
            rescale_time = config_train["rescale_time"]
        else:
            rescale_time = False

        self.model = ViT3(
            in_channels=in_dim,
            out_channels=out_dim,
            latent_channels=config_arch["latent_channels"],
            image_size=(config_train["s"], config_train["s"]),
            patch_sizes=config_arch["patch_sizes"],
            dims=config_arch["dims"],
            depth=config_arch["depth"],
            heads=config_arch["heads"],
            dim_head=config_arch["dim_head"],
            mlp_dim=config_arch["mlp_dim"],
            emb_channels= config_arch["emb_channels"],
            is_fourier_embedding = is_fourier_embedding,
            rescale_time = rescale_time,
            use_sdpa = config_train.get("use_sdpa", True),
            attention_chunk_size = config_train.get("attention_chunk_size", 0)
        )