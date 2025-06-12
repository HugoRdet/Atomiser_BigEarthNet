from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from .nn_comp import*
import einops as einops
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode_2d_from_coords(coords, max_freq=10.0, num_bands=4):
    """
    Args:
        coords: Tensor of shape (H, W, 2) with [x, y] coordinates
    Returns:
        Tensor of shape (H, W, 2 + 2*num_bands*2)
    """
    device, dtype = coords.device, coords.dtype
    freq_bands = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    
    # coords: (H, W, 2), unsqueeze → (H, W, 2, 1)
    scaled = coords.unsqueeze(-1) * freq_bands * pi  # (H, W, 2, num_bands)

    # Apply sin and cos → (H, W, 2, 2*num_bands), then flatten → (H, W, 4*num_bands)
    enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)

    return torch.cat([coords, enc], dim=-1)  # (H, W, 2 + 4*num_bands)

def fourier_encode_2D_batch_variable_res(images, res, max_freq, num_bands=4):
    """
    Fourier encode 2D positions with resolution-aware scaling per image.
    Args:
        images: (B, C, H, W)
        res: (B,) list or tensor of resolution factors (e.g., 1.0 → 120px)
    Returns:
        Tensor: (B, C + pos_dim, H, W)
    """
    B, H, W, C = images.shape
    device, dtype = images.device, images.dtype
    encoded = []

    for i in range(B):
        # Compute resolution-aware spatial grid
        scaling = 120 * res[i]
        x_coords = torch.linspace(-scaling/2, scaling/2, W, device=device, dtype=dtype)
        y_coords = torch.linspace(-scaling/2, scaling/2, H, device=device, dtype=dtype)
        mesh = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)  # (H, W, 2)

        # Encode coordinates
        pos_enc = fourier_encode_2d_from_coords(mesh, max_freq, num_bands)  # (H, W, pos_dim)
        
        
        encoded.append(pos_enc)

    return torch.stack(encoded)  # (B, C + pos_dim, H, W)

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

# main class

class Perceiver(pl.LightningModule):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            CrossAttention(
                query_dim   = latent_dim,
                context_dim = input_dim,
                heads       = cross_heads,
                dim_head    = cross_dim_head,
                dropout     = attn_dropout,
                use_flash   = True
            ),
            context_dim = input_dim
        ))

        get_cross_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ))

        get_latent_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            SelfAttention(
                dim        = latent_dim,
                heads      = latent_heads,
                dim_head   = latent_dim_head,
                dropout    = attn_dropout,
                use_flash  = True
            )
        ))

        get_latent_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        res=None,
        return_embeddings = False
    ):
        mask=einops.reduce(mask,"b c h w  -> b h w","min")
        mask = mask.to(torch.bool)
        mask=~mask
        data=rearrange(data,"b c h w -> b h w c")
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
       
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            if res!=None:
       
                enc_pos=fourier_encode_2D_batch_variable_res(data, res, self.max_freq, num_bands=self.num_freq_bands)
                
                data= torch.cat((data, enc_pos), dim = -1)
                
            else:

                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                
                data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        
        mask = rearrange(mask,"b h w -> b (h w)")
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)