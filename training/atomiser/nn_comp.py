import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pytorch_lightning as pl

# ---------------------------------
# Utilities
# ---------------------------------

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

# ---------------------------------
# PreNorm wrapper
# ---------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, context_dim: int = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and 'context' in kwargs:
            ctx = kwargs['context']
            kwargs['context'] = self.norm_context(ctx)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


@torch.compile
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult * 2)
        self.w2 = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, gate = self.w1(x).chunk(2, dim=-1)
        x = x1 * F.gelu(gate)
        return self.dropout(self.w2(x))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.,
        use_flash: bool = True
    ):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # one linear for Q,K,V
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (B, N, dim)
        B, N, _ = x.shape

        # project and split
        qkv = self.to_qkv(x)         # (B, N, 3·inner)
        q, k, v = qkv.chunk(3, dim=-1)

        # to (B·H, N, D)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.heads)

        if self.use_flash:
            # prepare mask for FlashAttention
            attn_mask = None
            if exists(mask):
                # mask: (B, N) -> (B, N, N)
                m = mask.unsqueeze(1).expand(-1, N, -1)            # (B, N, N)
                attn_mask = repeat(m, "b i j -> (b h) i j", h=self.heads)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.to_out[1].p,
                is_causal=False
            )
        else:
            # classic
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                m = mask.unsqueeze(1).expand(-1, N, -1)            # (B, N, N)
                m = repeat(m, "b i j -> (b h) i j", h=self.heads)
                sim = sim.masked_fill(~m, float("-inf"))
            attn = sim.softmax(dim=-1)
            attn = self.to_out[1](attn)
            out = einsum("b i j, b j d -> b i d", attn, v)

        # back to (B, N, inner)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out[0](out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_flash=True):
        super().__init__()
        context_dim = context_dim or query_dim
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, query_dim),
            nn.Dropout(dropout)
        )
        
        # Store dropout separately for manual attention
        self.dropout = nn.Dropout(dropout)
        
        # Will hold the last attention weights (manual path only)
        self.last_attn = None
        
    def forward(self, x, context, mask=None):
        B, Nq, _ = x.shape
        Nk = context.shape[1]
        
        # 1) Project Q, K, V
        q = self.to_q(x)  # (B, Nq, inner)
        kv = self.to_kv(context)  # (B, Nk, 2·inner)
        k, v = kv.chunk(2, dim=-1)  # each (B, Nk, inner)
        
        # 2) Split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        
        if self.use_flash:
            # FLASH path: no ability to grab attention weights
            attn_mask = None
            if mask is not None:
                # mask should be (B, Nq, Nk) - True for valid positions
                if mask.dim() == 2:
                    # If mask is (B, Nk), expand to (B, Nq, Nk)
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)
                attn_mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            self.last_attn = None
            
        else:
            # MANUAL path: compute attention, stash weights, then apply to values
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            
            
            
            if mask is not None:
                # mask should be (B, Nq, Nk) - True for valid positions
                if mask.dim() == 2:
                    # If mask is (B, Nk), expand to (B, Nq, Nk)
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)
                # Expand for heads: (B, 1, Nq, Nk) -> (B, heads, Nq, Nk)
                mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
                sim = sim.masked_fill(~mask, float("-inf"))
                

            
            
            
            attn = sim.softmax(dim=-1)  # (B, heads, Nq, Nk)
            
            # Check for NaN after softmax
            if torch.isnan(attn).any():
                print(f"ERROR: NaN detected after softmax!")
                print(f"Sim before softmax contains inf: {torch.isinf(sim).any()}")
                print(f"Sim before softmax min/max: {sim[~torch.isinf(sim)].min():.6f} / {sim[~torch.isinf(sim)].max():.6f}")
                # Replace NaN with uniform distribution over valid positions
                if mask is not None:
                    uniform_attn = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
                    attn = torch.where(torch.isnan(attn), uniform_attn, attn)
                else:
                    uniform_attn = torch.ones_like(attn) / attn.size(-1)
                    attn = torch.where(torch.isnan(attn), uniform_attn, attn)
            
            # Apply dropout to attention weights
            attn = self.dropout(attn)
            
            # IMPORTANT: Zero out attention scores for masked positions
            # This ensures masked tokens don't contribute to entropy calculations  
            if mask is not None:
                attn = attn * mask.float()
                # Renormalize to ensure rows sum to 1
                attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn = attn / attn_sum
            
            # Apply attention to values
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
            
            # Store attention weights for inspection/entropy calculation
            self.last_attn = attn.detach()  # Now contains zeros for masked positions
        
        # 3) Recombine heads and project
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class LatentAttentionPooling(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.cross  = CrossAttention(
            query_dim   = dim,
            context_dim = dim,
            heads       = heads,
            dim_head    = dim_head,
            dropout     = dropout
        )

    def forward(self, x):
        b = x.size(0)
        q = repeat(self.query, '1 1 d -> b 1 d', b=b)
        out = self.cross(q, context=x, mask=None)
        return out.squeeze(1)
    
class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=-2)
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)