from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
import seaborn as sns
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import wandb
import time



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


def pruning(tokens, attention_mask, percent):
    """
    Randomly drop `percent` of the *valid* tokens, i.e. those
    whose mask==True in *any* batch element.  Returns:
      - pruned tokens:     tokens[:, keep_idx, :]
      - pruned attention_mask: attention_mask[:, keep_idx]
      - keep_idx: indices of kept tokens in the original N
    """
    B, N, D = tokens.shape

    # find positions that are unmasked in *at least one* batch entry
    # (so we don't throw away tokens just because they happen
    #  to be masked in *some* images)
    valid = attention_mask.any(dim=0)           # shape (N,), bool
    valid_idx = torch.nonzero(valid, as_tuple=True)[0]  # (M,) positions

    M = valid_idx.numel()
    # how many of those M we want to drop
    n_drop = int(M * percent / 100)
    if n_drop <= 0:
        # nothing to drop
        return tokens, attention_mask, torch.arange(N, device=tokens.device)

    # shuffle only the valid positions
    perm = valid_idx[torch.randperm(M, device=tokens.device)]
    keep = perm[n_drop:]                       # keep the last M-n_drop

    # sort so we preserve original order
    keep_idx = torch.sort(keep)[0]

    # index into tokens & mask
    pruned_tokens = tokens[:, keep_idx, :]      # (B, M-n_drop, D)
    pruned_mask   = attention_mask[:, keep_idx] # (B, M-n_drop)
    
    return pruned_tokens, pruned_mask, keep_idx



class Atomiser(pl.LightningModule):
    def __init__(
        self,
        *,
        config,
        transform,
        depth: int,
        input_axis: int = 2,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        latent_attn_depth: int = 0,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        masking: float = 0.,
        wandb=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        self.input_axis = input_axis
        self.masking = masking
        self.config = config
        self.transform = transform

        # Compute input dim from encodings
        
        dx = self.get_shape_attributes_config("pos")
        dy = self.get_shape_attributes_config("pos")
        dw = self.get_shape_attributes_config("wavelength")
        db = self.get_shape_attributes_config("bandvalue")
        #ok
        input_dim = dx + dy + dw + db

        # Initialize spectral params
        #self.VV = nn.Parameter(torch.empty(dw))
        #self.VH = nn.Parameter(torch.empty(dw))
        #nn.init.trunc_normal_(self.VV, std=0.02, a=-2., b=2.)
        #nn.init.trunc_normal_(self.VH, std=0.02, a=-2., b=2.)

        # Latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)

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
        #d
        # Build cross/self-attn layers
        self.layers = nn.ModuleList()
        for i in range(depth):
            cache_args = {'_cache': (i>0 and weight_tie_layers)}
            # cross
            cross_attn = get_cross_attn(**cache_args)
            cross_ff   = get_cross_ff(**cache_args)
            # self
            self_attns = nn.ModuleList()
            
            for j in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = j),
                    get_latent_ff(**cache_args, key = j)
                ]))

            self.layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

  

        # Classifier
        if final_classifier_head:
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        else:
            self.to_logits = nn.Identity()
       


    def get_shape_attributes_config(self,attribute):
        if self.config["Atomiser"][attribute+"_encoding"]=="NOPE":
            return 0
        if self.config["Atomiser"][attribute+"_encoding"]=="NATURAL":
            return 1
        if self.config["Atomiser"][attribute+"_encoding"]=="FF":
            if self.config["Atomiser"][attribute+"_num_freq_bands"]==-1:
                return int(self.config["Atomiser"][attribute+"_max_freq"])*2+1
            else:
                return int(self.config["Atomiser"][attribute+"_num_freq_bands"])*2+1
        
        if self.config["Atomiser"][attribute+"_encoding"]=="GAUSSIANS":
            return int(len(self.config["wavelengths_encoding"].keys()))
        

    
                
    




    def forward(self, data, mask=None,resolution=None, training=True):
        # Preprocess tokens + mask
        
        if len(data.shape)==3:
            tokens=data
            tokens_mask=mask
        else:
            tokens, tokens_mask = self.transform.process_data(data, mask,resolution)
        



        b = tokens.shape[0]
        # initialize latents
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # apply mask to tokens
        tokens_mask = tokens_mask.to(torch.bool)
        tokens = tokens.masked_fill_(~tokens_mask.unsqueeze(-1), 0.)

        t, m = tokens, tokens_mask
        
        

        # cross & self layers
        for (cross_attn, cross_ff, self_attns) in self.layers:
            # optionally prune
            if self.masking > 0 and training:
                t, m, idx = pruning(tokens, tokens_mask, self.masking)
            # cross-attn
            x = cross_attn(x, context=t, mask=m) + x
            x = cross_ff(x) + x
            # restore tokens if pruned
            
            # self-attn blocks
            for (sa, ff) in self_attns:
                x = sa(x) + x
                x = ff(x) + x


        # classifier
        return self.to_logits(x)
    

    def visualize_attention(self, data, mask=None, resolution=None, step=None, batch_id=None, MAX_BB=128):
    

        N_LAYERS = len(self.layers)
        TOPK     = 100
        COLORS   = sns.color_palette("tab10", n_colors=N_LAYERS)

        # only process first MAX_BB batches
        if batch_id is None or batch_id < 0 or batch_id >= MAX_BB:
            return

        # initialize accumulators once
        if not hasattr(self, "_layer_concs"):
            self._layer_concs  = [[] for _ in range(N_LAYERS)]
            self._layer_ents   = [[] for _ in range(N_LAYERS)]
            self._overlaps     = []
            self._persistence  = []

        # turn on return_attention
        cross_mods, orig_ret, orig_flash = [], [], []
        for layer in self.layers:
            cross = layer[0].fn
            cross_mods.append(cross)
            orig_ret.append(cross.return_attention)
            orig_flash.append(cross.use_flash)
            cross.return_attention = True
            cross.use_flash         = False

        was_train = self.training
        self.eval()

        # forward and capture all layers' attentions
        all_attns = []
        try:
            with torch.no_grad():
                if data.ndim == 3:
                    t, m = data, mask
                else:
                    t, m = self.transform.process_data(data, mask, resolution)

                b = t.shape[0]
                x = repeat(self.latents, 'n d -> b n d', b=b)
                m = m.to(torch.bool)
                t = t.masked_fill_(~m.unsqueeze(-1), 0.)
                if self.masking > 0:
                    t, m, _ = pruning(t, m, self.masking)

                for cross, ff, selfs in self.layers:
                    out = cross(x, context=t, mask=m)
                    if isinstance(out, tuple):
                        x, attn = out
                        all_attns.append(attn.cpu())
                    else:
                        x = out
                    x = ff(x) + x
                    for sa, ff2 in selfs:
                        x = sa(x) + x
                        x = ff2(x) + x
        finally:
            for mod, ret, fl in zip(cross_mods, orig_ret, orig_flash):
                mod.return_attention = ret
                mod.use_flash         = fl
            if was_train:
                self.train()

        if not all_attns:
            return

        # per-layer conc & entropy
        topk_sets = []
        for l, attn in enumerate(all_attns):
            m = attn.mean(dim=0)                     # [Nq,Nk]
            sums = m.sum(dim=0)                      # [Nk]
            ent  = -(m * torch.log(m + 1e-8)).sum(dim=1)  # [Nq]

            conc = (sums.max() / sums.mean()).item()
            avg_ent = ent.mean().item()
            self._layer_concs[l].append(conc)
            self._layer_ents[l].append(avg_ent)

            topk = torch.topk(sums, TOPK).indices.tolist()
            topk_sets.append(set(topk))

        # overlap & persistence
        overlap = torch.zeros((N_LAYERS, N_LAYERS))
        for i in range(N_LAYERS):
            for j in range(N_LAYERS):
                inter = topk_sets[i] & topk_sets[j]
                overlap[i, j] = len(inter) / TOPK
        self._overlaps.append(overlap)
        common = set.intersection(*topk_sets)
        self._persistence.append(len(common))

        # on final batch, plot everything
        if batch_id == MAX_BB - 1:
            # 1) Concentration vs. Layer
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            for l in range(N_LAYERS):
                ax1.scatter(
                    [l]*len(self._layer_concs[l]),
                    self._layer_concs[l],
                    color=COLORS[l],
                    label=f"Layer {l}",
                    alpha=0.7
                )
            ax1.set_xticks(range(N_LAYERS))
            ax1.set(title="Concentration Ratio by Layer",
                    xlabel="Layer", ylabel="Concentration")
            ax1.legend(ncol=4, bbox_to_anchor=(1,1))
            wandb.log({"attention/conc_vs_layer": wandb.Image(fig1)})
            plt.close(fig1)

            # 2) Concentration vs. Entropy by Layer
            fig2, ax2 = plt.subplots(figsize=(6,6))
            for l in range(N_LAYERS):
                ax2.scatter(
                    self._layer_concs[l],
                    self._layer_ents[l],
                    color=COLORS[l],
                    label=f"Layer {l}",
                    alpha=0.7
                )
            ax2.set(title="Concentration vs. Entropy by Layer",
                    xlabel="Concentration", ylabel="Entropy")
            ax2.legend(ncol=2, bbox_to_anchor=(1,1))
            wandb.log({"attention/conc_vs_ent_by_layer": wandb.Image(fig2)})
            plt.close(fig2)

            # 3) Average Overlap Matrix
            mean_ov = torch.stack(self._overlaps).mean(dim=0)
            fig3, ax3 = plt.subplots(figsize=(6,6))
            sns.heatmap(mean_ov.numpy(), annot=True, cmap="viridis", ax=ax3)
            ax3.set(title="Avg Top-K Overlap Across Layers")
            wandb.log({"attention/avg_overlap_matrix": wandb.Image(fig3)})
            plt.close(fig3)

            # 4) Persistence Histogram
            fig4, ax4 = plt.subplots()
            sns.histplot(self._persistence,
                        bins=range(0, TOPK+2),
                        discrete=True,
                        ax=ax4)
            ax4.set(title="Persistence of Top-K Tokens",
                    xlabel="# tokens in all layers",
                    ylabel="Example Count")
            wandb.log({"attention/persistence_hist": wandb.Image(fig4)})
            plt.close(fig4)

            # 5a) Concentration Distribution by Layer
            fig5, ax5 = plt.subplots()
            all_concs = sum(self._layer_concs, [])
            bins_c   = np.linspace(min(all_concs), max(all_concs), MAX_BB)
            for l in range(N_LAYERS):
                sns.histplot(
                    self._layer_concs[l],
                    bins=bins_c,
                    ax=ax5,
                    color=COLORS[l],
                    label=f"Layer {l}",
                    alpha=0.6,
                    kde=False
                )
            ax5.set(title="Concentration Distribution by Layer",
                    xlabel="Concentration Ratio", ylabel="Count")
            ax5.legend(ncol=2, bbox_to_anchor=(1,1))
            wandb.log({"attention/dist_all_conc_by_layer": wandb.Image(fig5)})
            plt.close(fig5)

            # 5b) Entropy Distribution by Layer
            fig6, ax6 = plt.subplots()
            all_ents = sum(self._layer_ents, [])
            bins_e   = np.linspace(min(all_ents), max(all_ents), MAX_BB)
            for l in range(N_LAYERS):
                sns.histplot(
                    self._layer_ents[l],
                    bins=bins_e,
                    ax=ax6,
                    color=COLORS[l],
                    label=f"Layer {l}",
                    alpha=0.6,
                    kde=False
                )
            ax6.set(title="Entropy Distribution by Layer",
                    xlabel="Mean Attention Entropy", ylabel="Count")
            ax6.legend(ncol=2, bbox_to_anchor=(1,1))
            wandb.log({"attention/dist_all_ent_by_layer": wandb.Image(fig6)})
            plt.close(fig6)

        # restore cross-attn flags
        for layer in self.layers:
            cross = layer[0].fn
            cross.return_attention = False
            cross.use_flash = True

        
