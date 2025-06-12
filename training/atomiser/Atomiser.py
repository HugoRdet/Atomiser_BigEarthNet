from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt


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
    N = tokens.size(1)
    n_mask = int(N * percent/100.)
    perm = torch.randperm(N, device=tokens.device)
    keep_idx = perm[n_mask:]              
    return tokens[:, keep_idx], attention_mask[:, keep_idx], keep_idx



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
        
        if self.masking > 0 and training:
            t, m, idx = pruning(t, m, self.masking)

        # cross & self layers
        for (cross_attn, cross_ff, self_attns) in self.layers:
            # optionally prune
            
            # cross-attn
            x = cross_attn(x, context=t, mask=m) + x
            x = cross_ff(x) + x
            # restore tokens if pruned
            #if self.masking > 0 and training:
            #    tokens[:, idx] = t
            #    tokens_mask[:, idx] = m
            # self-attn blocks
            for (sa, ff) in self_attns:
                x = sa(x) + x
                x = ff(x) + x


        # classifier
        return self.to_logits(x)
    
    def visualize_attention(self, data, mask=None, resolution=None, step=None, batch_id=None):
        """
        Extracts and plots cross-attention WEIGHTS for attention sink analysis.
        Logs to wandb instead of showing plots.
        
        Args:
            data: Input data
            mask: Attention mask
            resolution: Resolution parameter
            step: Training step for logging
            batch_id: Batch ID for tracking individual batches (0-9 for first 10 batches)
        """
        
        # Only visualize for the first 10 batches
        if batch_id is not None and batch_id >= 10:
            return
        
        # Temporarily enable attention return for cross-attention layers
        cross_attn_modules = []
        original_flags = []
        original_flash_flags = []
        
        for layer in self.layers:  # self.layers exists in Atomiser
            cross_attn = layer[0].fn  # CrossAttention inside PreNorm
            cross_attn_modules.append(cross_attn)
            original_flags.append(getattr(cross_attn, 'return_attention', False))
            original_flash_flags.append(cross_attn.use_flash)
            cross_attn.return_attention = True
            cross_attn.use_flash = False  # Disable flash attention to get weights
        
        # Store original training mode and set to eval
        was_training = self.training
        self.eval()
        attention_scores = []
        
        try:
            with torch.no_grad():
                if len(data.shape) == 3:
                    processed_tokens = data
                    tokens_mask = mask
                else:
                    processed_tokens, tokens_mask = self.transform.process_data(data, mask, resolution)

                b = processed_tokens.shape[0]
                x = repeat(self.latents, 'n d -> b n d', b=b)
                tokens_mask = tokens_mask.to(torch.bool)
                processed_tokens = processed_tokens.masked_fill_(~tokens_mask.unsqueeze(-1), 0.)

                t, m = processed_tokens, tokens_mask
                
                if self.masking > 0:
                    t, m, idx = pruning(t, m, self.masking)

                # Process layers and capture attention from first cross-attention
                for i, (cross_attn, cross_ff, self_attns) in enumerate(self.layers):
                    # Cross-attention with attention capture
                    if hasattr(cross_attn.fn, 'return_attention') and cross_attn.fn.return_attention:
                        x_out, attn = cross_attn(x, context=t, mask=m)
                        if i == 0:  # Only capture first layer
                            attention_scores.append(attn.detach().cpu())
                        x = x_out + x
                    else:
                        x = cross_attn(x, context=t, mask=m) + x
                    
                    x = cross_ff(x) + x
                    
                    # Self-attention blocks
                    for (sa, ff) in self_attns:
                        x = sa(x) + x
                        x = ff(x) + x

        finally:
            # Restore original settings
            for cross_attn, original_flag, original_flash in zip(cross_attn_modules, original_flags, original_flash_flags):
                cross_attn.return_attention = original_flag
                cross_attn.use_flash = original_flash
            
            # Restore training mode
            if was_training:
                self.train()

        if attention_scores:
            # Process attention weights
            attn = attention_scores[0]  # [B*heads, Nq, Nk]
            
            # Since batch_size=1, we have heads attention matrices
            # attn shape: [heads, Nq, Nk]
            attn_mean = attn.mean(dim=0)  # [Nq, Nk] averaged over heads
            
            # Create batch-specific identifier for logging
            batch_suffix = f"_batch_{batch_id}" if batch_id is not None else ""
            step_suffix = f"_step_{step}" if step is not None else ""
            log_suffix = f"{batch_suffix}{step_suffix}"
            
            # Create figures for wandb
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Full attention heatmap
            sns.heatmap(attn_mean.numpy(), cmap="viridis", ax=axes[0,0])
            axes[0,0].set_title(f"Cross-Attention Weights (Batch {batch_id})")
            axes[0,0].set_xlabel("Input Tokens (context)")
            axes[0,0].set_ylabel("Latent Tokens (query)")
            
            # 2. Attention weight distribution per input token (potential sinks)
            token_attention_sum = attn_mean.sum(dim=0)  # Sum over all queries
            axes[0,1].bar(range(len(token_attention_sum)), token_attention_sum.numpy())
            axes[0,1].set_title(f"Total Attention per Input Token (Batch {batch_id})")
            axes[0,1].set_xlabel("Input Token Index")
            axes[0,1].set_ylabel("Total Attention Weight")
            
            # 3. Max attention weight per input token
            max_attention_per_token = attn_mean.max(dim=0)[0]
            axes[1,0].bar(range(len(max_attention_per_token)), max_attention_per_token.numpy())
            axes[1,0].set_title(f"Max Attention Weight per Input Token (Batch {batch_id})")
            axes[1,0].set_xlabel("Input Token Index") 
            axes[1,0].set_ylabel("Max Attention Weight")
            
            # 4. Attention entropy per query (diversity measure)
            attn_entropy = -(attn_mean * torch.log(attn_mean + 1e-8)).sum(dim=1)
            axes[1,1].plot(attn_entropy.numpy())
            axes[1,1].set_title(f"Attention Entropy per Query (Batch {batch_id})")
            axes[1,1].set_xlabel("Query Index")
            axes[1,1].set_ylabel("Entropy")
            
            plt.tight_layout()
            
            # Log to wandb with batch-specific naming
            wandb.log({
                f"attention/cross_attention_analysis{log_suffix}": wandb.Image(fig),
            })
            
            # Also log individual attention heads for this batch
            if batch_id is not None and batch_id < 5:  # Only for first 5 batches to avoid too many plots
                fig_heads, axes_heads = plt.subplots(2, 4, figsize=(20, 10))
                axes_heads = axes_heads.flatten()
                
                num_heads = min(8, attn.shape[0])  # Show up to 8 heads
                for head_idx in range(num_heads):
                    if head_idx < len(axes_heads):
                        sns.heatmap(attn[head_idx].numpy(), cmap="viridis", ax=axes_heads[head_idx])
                        axes_heads[head_idx].set_title(f"Head {head_idx}")
                        axes_heads[head_idx].set_xlabel("Input Tokens")
                        axes_heads[head_idx].set_ylabel("Latent Tokens")
                
                # Hide unused subplots
                for head_idx in range(num_heads, len(axes_heads)):
                    axes_heads[head_idx].set_visible(False)
                
                plt.suptitle(f"Individual Attention Heads - Batch {batch_id}")
                plt.tight_layout()
                
                wandb.log({
                    f"attention/individual_heads{log_suffix}": wandb.Image(fig_heads),
                })
                plt.close(fig_heads)
            
            # Close the main figure to free memory
            plt.close(fig)
            
            # Log attention statistics as scalars with batch-specific naming
            top_5_sinks = torch.topk(token_attention_sum, min(5, len(token_attention_sum))).indices.tolist()
            attention_concentration = token_attention_sum.max().item() / token_attention_sum.mean().item()
            mean_entropy = attn_entropy.mean().item()
            
            wandb.log({
                f"attention/concentration_ratio{log_suffix}": attention_concentration,
                f"attention/mean_entropy{log_suffix}": mean_entropy,
                f"attention/max_attention_weight{log_suffix}": token_attention_sum.max().item(),
                f"attention/min_attention_weight{log_suffix}": token_attention_sum.min().item(),
            })
            
            # Log top sink tokens as a table with batch information
            sink_data = []
            for i, token_idx in enumerate(top_5_sinks):
                sink_data.append([
                    batch_id if batch_id is not None else "Unknown",  # Batch ID
                    i + 1,  # Rank
                    token_idx,  # Token index
                    token_attention_sum[token_idx].item(),  # Total attention
                    max_attention_per_token[token_idx].item()  # Max attention
                ])
            
            sink_table = wandb.Table(
                columns=["Batch_ID", "Rank", "Token_Index", "Total_Attention", "Max_Attention"],
                data=sink_data
            )
            wandb.log({f"attention/top_sink_tokens{log_suffix}": sink_table})
            
            print(f"✅ Attention analysis for batch {batch_id} logged to wandb")
            print(f"Top 5 attention sink tokens for batch {batch_id}: {top_5_sinks}")
            print(f"Attention concentration ratio: {attention_concentration:.3f}")
            print(f"Mean attention entropy: {mean_entropy:.3f}")
            
        else:
            print(f"❌ No attention scores captured for batch {batch_id}!")
            wandb.log({f"attention/capture_failed{log_suffix}": 1})