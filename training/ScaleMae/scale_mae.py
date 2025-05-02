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

# Preprocessing pipeline—no data_keys needed when only a single tensor is passed positionally
_scale_mae_transforms = K.AugmentationSequential(
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    K.Normalize(mean=_mean,           std=_std),
)


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, grid_size: int, res: Tensor, cls_token: bool = False
) -> Tensor:
    """Generate spatial resolution–specific 2D positional embeddings."""
    device, dtype = res.device, res.dtype
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    # scale the grid by each sample's resolution
    grid = torch.einsum('chw,n->cnhw', grid, res)
    _, n, h, w = grid.shape
    pos = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos = pos.reshape(n, h*w, embed_dim)
    if cls_token:
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
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def forward_features(self, x: Tensor, res: Tensor) -> Tensor:
        # x: (B, C, H, W), res: (B,) per-sample resolution
        x = self.patch_embed(x)            # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)   # (B, P, D)
        x = self._pos_embed(x, res)        # (B, P+1, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: Tensor, res: Tensor) -> Tensor:
        B, N, D = x.shape
        side = int(self.patch_embed.num_patches**0.5)

        res_unique, inv_idx = torch.unique(res, return_inverse=True)
        pe_list = []
        for r in res_unique:
            pe = get_2d_sincos_pos_embed_with_resolution(
                D, side, r.view(1), cls_token=bool(self.cls_token is not None)
            )
            pe_list.append(pe)
        pe_stack = torch.cat(pe_list, dim=0)  # (U, N, D)
        batch_pe = pe_stack[inv_idx]          # (B, N, D)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
        return self.pos_drop(x + batch_pe)


def interpolate_pos_embed(
    model: ScaleMAE, state_dict: OrderedDict[str, Tensor]
) -> OrderedDict[str, Tensor]:
    pos_embed_ckpt = state_dict['pos_embed']
    emb_size = pos_embed_ckpt.shape[-1]
    num_patches = model.patch_embed.num_patches
    extra = 0
    if model.pos_embed is not None:
        extra = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_ckpt.shape[-2] - extra)**0.5)
    new_size = int(num_patches**0.5)
    if orig_size != new_size:
        print(f"Interpolating from {orig_size} to {new_size}")
        extras = pos_embed_ckpt[:, :extra]
        tokens = pos_embed_ckpt[:, extra:]
        tokens = tokens.reshape(-1, orig_size, orig_size, emb_size).permute(0,3,1,2)
        tokens = nn.functional.interpolate(
            tokens, size=(new_size,new_size), mode='bicubic', align_corners=False
        )
        tokens = tokens.permute(0,2,3,1).flatten(1,2)
        state_dict['pos_embed'] = torch.cat([extras, tokens], dim=1)
    return state_dict


class CustomScaleMAE(pl.LightningModule):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.encoder = ScaleMAE(
            img_size=120, patch_size=15, in_chans=12,
            embed_dim=768, depth=12, num_heads=12
        )
        self.to_logits = nn.Linear(self.encoder.embed_dim, num_classes)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = _scale_mae_transforms(x)
        feats = self.encoder.forward_features(x, res)
        cls_feat = feats[:, 0]
        return self.to_logits(cls_feat)

# torchvision WeightEnum override
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]

class ScaleMAELarge16_Weights(WeightsEnum):
    FMOW_RGB = Weights(
        url='https://hf.co/torchgeo/vit_large_patch16_224_fmow_rgb_scalemae/...',
        transforms=_scale_mae_transforms,
        meta={ ... },
    )


def scalemae_large_patch16(
    weights: ScaleMAELarge16_Weights | None = None, *args: Any, **kwargs: Any
) -> ScaleMAE:
    model = ScaleMAE(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                     mlp_ratio=4, qkv_bias=True,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     *args, **kwargs)
    if weights:
        state_dict = weights.get_state_dict(progress=True)
        if 'img_size' in kwargs and weights.meta['img_size'] != kwargs['img_size']:
            state_dict = interpolate_pos_embed(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model
