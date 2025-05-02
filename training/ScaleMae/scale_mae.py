import pytorch_lightning as pl
from collections import OrderedDict
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

# Normalization constants
_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])

class NormalizeInput(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, inputs: dict):
        x = inputs['input']
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return {'input': x}

_scale_mae_transforms = K.AugmentationSequential(
    NormalizeInput(mean=torch.tensor(0),    std=torch.tensor(255)),
    NormalizeInput(mean=_mean,              std=_std),
    data_keys=["input"],
)

def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, grid_size: int, res: Tensor, cls_token: bool = False
) -> Tensor:
    """Generate spatial resolution specific 2D positional embeddings.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid_size: Height (ph) and width (pw) of the image patches.
        res: Spatial resolution tensor of shape (N,) of the image.
        cls_token: Increase positional embedding size by 1 for class token.

    Returns:
        pos_embed: Spatial resolution aware positional embeddings (N, Ph*Pw(+1), D).
    """
    device, dtype = res.device, res.dtype
    # build grid coords scaled by per-sample resolution
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    # grid: (2, H, W) -> (2, N, H, W) after scaling by res
    grid = torch.einsum('chw,n->cnhw', grid, res)
    _, n, h, w = grid.shape
    # sin-cos positional embedding from the grid
    pos = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos = pos.reshape(n, h*w, embed_dim)
    if cls_token:
        # prepend zero embedding for class token per sample
        cls = torch.zeros(n, 1, embed_dim, dtype=dtype, device=device)
        pos = torch.cat([cls, pos], dim=1)
    return pos


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    angles = torch.einsum('m,d->md', pos, omega)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return torch.cat([sin, cos], dim=1)


class ScaleMAE(VisionTransformer):
    """Scale-aware MAE encoder with per-sample GSD positional embeddings."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # freeze any existing pos_embed
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def forward_features(self, x: Tensor, res: Tensor) -> Tensor:
        """
        x:   (B, C, H, W)
        res: (B,) per-sample resolution
        """
        # linear patch embedding
        x = self.patch_embed(x)           # (B, embed_dim, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B, P, D)

        # add per-sample pos embeddings
        x = self._pos_embed(x, res)       # (B, P+1, D)

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: Tensor, res: Tensor) -> Tensor:
        """
        x:   (B, N, D)
        res: (B,) resolutions

        returns: (B, N, D)
        """
        B, N, D = x.shape
        side = int(self.patch_embed.num_patches**0.5)

        # compute one pos_embed per sample
        # vectorize over unique res for efficiency
        res_unique, inv_idx = torch.unique(res, return_inverse=True)
        pe_list = []
        for r in res_unique:
            pe = get_2d_sincos_pos_embed_with_resolution(
                D, side, r.view(1), cls_token=bool(self.cls_token is not None)
            )  # shape (1, N, D)
            pe_list.append(pe)
        pe_stack = torch.cat(pe_list, dim=0)      # (U, N, D)
        batch_pe = pe_stack[inv_idx]              # (B, N, D)

        # prepend cls token if not included
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
        return self.pos_drop(x + batch_pe)


def interpolate_pos_embed(
    model: ScaleMAE, state_dict: OrderedDict[str, Tensor]
) -> OrderedDict[str, Tensor]:
    # identical to previous implementation for pretrained interpolation
    pos_embed_checkpoint = state_dict['pos_embed']
    emb_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    extra_tokens = 0
    if model.pos_embed is not None:
        extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - extra_tokens)**0.5)
    new_size = int(num_patches**0.5)
    if orig_size != new_size:
        print(f"Interpolating from {orig_size} to {new_size}")
        extras = pos_embed_checkpoint[:, :extra_tokens]
        tokens = pos_embed_checkpoint[:, extra_tokens:]
        tokens = tokens.reshape(-1, orig_size, orig_size, emb_size).permute(0,3,1,2)
        tokens = torch.nn.functional.interpolate(
            tokens, size=(new_size,new_size), mode='bicubic', align_corners=False
        )
        tokens = tokens.permute(0,2,3,1).flatten(1,2)
        state_dict['pos_embed'] = torch.cat([extras, tokens], dim=1)
    return state_dict


class CustomScaleMAE(pl.LightningModule):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        # instantiate encoder without fixed res
        self.encoder = ScaleMAE(
            img_size=120,
            patch_size=15,
            in_chans=12,
            embed_dim=768,  # specify your dims
            depth=12,
            num_heads=12,
        )
        self.to_logits = nn.Linear(self.encoder.embed_dim, num_classes)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        """
        x:   (B, C, H, W)
        res: (B,) per-sample resolutions
        """
        # … in forward …
        out = _scale_mae_transforms({"input": x})
        x   = out["input"]
        feats = self.encoder.forward_features(x, res)
        # typically take cls token
        cls_feat = feats[:, 0]
        logits = self.to_logits(cls_feat)
        return logits
