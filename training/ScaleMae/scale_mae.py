import pytorch_lightning as pl
from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from torchvision.models._api import Weights, WeightsEnum



def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=-1)



def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    assert embed_dim % 2 == 0, "1D embed_dim must be even"
    # pos: (B, H, W)
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    pos_flat = pos.reshape(-1)
    angles = torch.outer(pos_flat, omega)
    sin, cos = torch.sin(angles), torch.cos(angles)
    B, H, W = pos.shape
    return torch.cat([sin, cos], dim=-1).view(B, H * W, embed_dim)



def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int,
    grid_size: int,
    res: Tensor,
    cls_token: bool = False
) -> Tensor:
    """Generate per-sample 2D sin-cos pos embeddings (B, grid_size**2(+cls), D)."""
    device, dtype = res.device, res.dtype
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    # scale by per-sample resolution: (2, grid_size, grid_size) -> (2, B, H, W)
    grid = torch.einsum('chw,b->c b h w', grid, res)
    pos = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  # (B, H*W, D)
    if cls_token:
        B = pos.shape[0]
        cls = torch.zeros(B, 1, embed_dim, dtype=dtype, device=device)
        pos = torch.cat([cls, pos], dim=1)  # (B,1+H*W,D)
    return pos


# ==================== ScaleMAE Model ====================

class ScaleMAE(VisionTransformer):
    """Vision Transformer encoder with GSD positional embeddings per sample."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # freeze original pos_embed if present
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def forward_features(self, x: Tensor, res: Tensor) -> Tensor:
        # x: (B, C, H, W), res: (B,)
        x = self.patch_embed(x)              # -> (B, P, D)
        x = self._pos_embed(x, res)          # -> (B, 1+P, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: Tensor, res: Tensor) -> Tensor:
        """
        Add per-sample positional embeddings and a cls token.
        x:   (B, P, D)
        res: (B,)
        returns: (B, 1+P, D)
        """
        B, P, D = x.shape
        # compute spatial pos embeddings only (no cls)
        side = int(P**0.5)  # Calculate grid size from number of patches
        spatial_pos = get_2d_sincos_pos_embed_with_resolution(
            D, side, res, cls_token=False
        ).to(x.dtype).to(x.device)  # (B, P, D)

        if self.cls_token is not None:
            # prepend cls token vector
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            x = torch.cat((cls_tokens, x), dim=1)           # (B,1+P,D)
            # create a zero positional embedding for cls
            cls_pos = torch.zeros_like(cls_tokens)
            pos = torch.cat((cls_pos, spatial_pos), dim=1)
        else:
            pos = spatial_pos  # (B,P,D)

        return self.pos_drop(x + pos)


def interpolate_pos_embed(
    model: ScaleMAE, state_dict: OrderedDict[str, Tensor]
) -> OrderedDict[str, Tensor]:
    pos_embed_ckpt = state_dict['pos_embed']
    _, L, D = pos_embed_ckpt.shape
    N = model.patch_embed.num_patches
    extra = L - N
    orig_size = int((L - extra)**0.5)
    new_size = int(N**0.5)
    if orig_size != new_size:
        # split cls and grid tokens
        cls_tokens = pos_embed_ckpt[:, :extra]
        grid_tokens = pos_embed_ckpt[:, extra:]
        grid_tokens = grid_tokens.reshape(-1, orig_size, orig_size, D).permute(0,3,1,2)
        grid_tokens = nn.functional.interpolate(
            grid_tokens, size=(new_size, new_size),
            mode='bicubic', align_corners=False
        )
        grid_tokens = grid_tokens.permute(0,2,3,1).reshape(-1, new_size*new_size, D)
        state_dict['pos_embed'] = torch.cat([cls_tokens, grid_tokens], dim=1)
    return state_dict


class CustomScaleMAE(pl.LightningModule):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.encoder = ScaleMAE(
            img_size=120, patch_size=15, in_chans=12,
            embed_dim=768, depth=12, num_heads=12,
            num_classes=num_classes  # Configure the built-in head
        )
        
    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        feats = self.encoder.forward_features(x, res)
        cls_feat = feats[:, 0]
        return self.encoder.head(cls_feat)
    




