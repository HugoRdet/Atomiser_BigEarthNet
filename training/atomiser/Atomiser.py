from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt
import numpy as np
from pytorch_optimizer import Lamb
import torch
from torch import nn, einsum
import torch.nn.functional as F
import seaborn as sns
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import wandb
import time
from torch.profiler import record_function


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



import torch



def pruning(tokens: torch.Tensor,
            attention_mask: torch.Tensor,
            percent: float):
    """
    Randomly drop `percent`% of the *valid* tokens (those
    where attention_mask is True in any batch element).

    Returns:
      - pruned_tokens:        Tensor of shape (B, K, D)
      - pruned_attention_mask:Tensor of shape (B, K)
      - keep_idx:             LongTensor of shape (K,), the indices in [0..N)
    """
    B, N, D = tokens.shape
    device = tokens.device

   
    #valid = attention_mask.any(dim=0)           
    #valid_idx = valid.nonzero(as_tuple=True)[0]   

    keep_frac = 1.0 - (percent / 100.0)
    #num_valid = valid_idx.size(0)
    num_keep  = max(1, int(N * keep_frac))

    perm      = torch.randperm(N, device=device)
    kept_perm = perm[:num_keep]                   
    #keep_idx  = valid_idx[kept_perm]             

    pruned_tokens        = tokens[:, kept_perm, :].clone()      
    pruned_attention_mask= attention_mask[:, kept_perm].clone()    

    return pruned_tokens, pruned_attention_mask






class Atomiser(pl.LightningModule):
    def __init__(
        self,
        *,
        config,
        transform,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        
        self.config=config
        self.transform=transform
        depth=int(self.config["Atomiser"]["depth"])
        num_latents=int(self.config["Atomiser"]["num_latents"])
        latent_dim=int(self.config["Atomiser"]["latent_dim"])
        cross_heads=int(self.config["Atomiser"]["cross_heads"])
        latent_heads=int(self.config["Atomiser"]["latent_heads"])
        cross_dim_head=int(self.config["Atomiser"]["cross_dim_head"])
        latent_dim_head=int(self.config["Atomiser"]["latent_dim_head"])
        num_classes=self.config["trainer"]["num_classes"]
        attn_dropout=self.config["Atomiser"]["attn_dropout"]
        ff_dropout=self.config["Atomiser"]["ff_dropout"]
        weight_tie_layers=self.config["Atomiser"]["weight_tie_layers"]
        self_per_cross_attn=self.config["Atomiser"]["self_per_cross_attn"]
        final_classifier_head=self.config["Atomiser"]["final_classifier_head"]             
        self.input_axis = 2
        
        self.max_tokens_forward = self.config["trainer"]["max_tokens_forward"]
        self.max_tokens_val = self.config["trainer"]["max_tokens_val"]
        self.max_tokens_reconstruction = self.config["trainer"]["max_tokens_reconstruction"]
        
        



        # Compute input dim from encodings
        
        dx = self.get_shape_attributes_config("pos")
        dy = self.get_shape_attributes_config("pos")
        dw = self.get_shape_attributes_config("wavelength")
        db = self.get_shape_attributes_config("bandvalue")
        #ok
        input_dim = dx + dy + dw + db
        query_dim_recon = dx + dy + dw 

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

  

        # Decoder TODO
        recon_dim = 1 #we just reconstruct the reflectance
        
        
        
        self.recon_cross = PreNorm(
            query_dim_recon,   
            CrossAttention(
                query_dim   = query_dim_recon,   
                context_dim = latent_dim,   
                heads       = latent_heads,
                dim_head    = latent_dim_head,
                dropout=0.0
            ),
            context_dim=latent_dim
        )
        
        self.recon_dim = recon_dim  # keep for reference
        hidden = max(128, query_dim_recon * 2)
        self.recon_mlp = nn.Sequential(
            nn.LayerNorm(query_dim_recon),
            nn.Linear(query_dim_recon, hidden),
            nn.GELU(),                # ReLU works too; GELU is smoother
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.recon_dim)  # linear output for regression
        )

        self.recon_ff = PreNorm(
            query_dim_recon, 
            FeedForward(query_dim_recon, dropout=ff_dropout)  
        )

       
        self.recon_head = nn.Sequential(
            nn.LayerNorm(query_dim_recon),      
            nn.Linear(query_dim_recon, recon_dim) 
        )

        
        
        # Classifier
        if final_classifier_head:
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        
            
    def _set_requires_grad(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_encoder(self):
        # freeze all attention/ffn layers used by self.encoder
        self._set_requires_grad(self.layers, False)
        # freeze latents parameter
        if hasattr(self, "latents"):
            self.latents.requires_grad = False

    def unfreeze_encoder(self):
        self._set_requires_grad(self.layers, True)
        if hasattr(self, "latents"):
            self.latents.requires_grad = True
            
    def _set_requires_grad(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_decoder(self):
        for m in (self.recon_cross, self.recon_ff, self.recon_head):
            self._set_requires_grad(m, False)

    def unfreeze_decoder(self):
        for m in (self.recon_cross, self.recon_ff, self.recon_head):
            self._set_requires_grad(m, True)

    def freeze_classifier(self):
        self._set_requires_grad(self.to_logits, False)

    def unfreeze_classifier(self):
        self._set_requires_grad(self.to_logits, True)

    


       


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
        
        
    def reconstruct(self, latents, mae_tokens, mae_tokens_mask):
        """
        latents:        [B, L, D]  (output of encoder)
        mae_tokens:     [B, N, Din]  raw tokens for positions you want to reconstruct
        mae_tokens_mask:[B, N]  boolean, True where padding/invalid (same convention as encoder)
        returns:
            preds: [B, K, recon_dim]  predictions (K <= N if limited)
            out_mask: [B, K]          boolean mask aligned to preds (True = invalid)
        """

        
        query_tokens = mae_tokens.clone()
        query_mask   = mae_tokens_mask.clone()
        
        query_tokens, query_mask = self.transform.process_data(query_tokens, query_mask,query=True)
        
        preds = self.recon_cross(query_tokens, context=latents, mask=None)  # [B, K, D]
        preds= self.recon_mlp(preds)
        
        return preds, query_mask


    
                
    
    
    def encoder(self,x,tokens,mask, training=True):
        #performs the encoding of the latents through the multiple cross attention and self attention layers
        #tokens: [B, K, D]
        #mask: [B, K]
        #latents: [N, D]
        
        for idx_layer, (cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            token_limit=self.max_tokens_forward
            if not training:
                token_limit=self.max_tokens_val
            
            permutation = torch.randperm(tokens.shape[1], device=tokens.device)
            tmp_tokens = tokens[:,permutation[:token_limit]].clone()
            tmp_mask = mask[:,permutation[:token_limit]].clone()
            
            tokens, tokens_mask = self.transform.process_data(tmp_tokens, tmp_mask)
            tokens_mask= tokens_mask.bool()
            tokens = tokens.masked_fill_(tokens_mask.unsqueeze(-1), 0.)
            
            
            
            x_ca = cross_attn(x, context=tokens, mask=None,id=idx_layer)
            
            x = x_ca + x
            
            # Feed-forward after CA
            x_ff = cross_ff(x)
            x = x_ff + x
            
            # Self-attention + FF blocks
            for blk_idx, (sa, ff) in enumerate(self_attns):
                x_sa = sa(x)
                
                x = x_sa + x
                

                x_ff2 = ff(x)
                
                x = x_ff2 + x
                

        return x



    
    def forward(self, data, mask,mae_tokens,mae_tokens_mask, training=True, reconstruction=True):
        #extraction of the batch size
        b = data.shape[0]   
        
        
        

        # Initialize latents
        x = repeat(self.latents, 'n d -> b n d', b=b) 
        
        #x=self.encoder(x, data, mask, training=training)
        
        #reconstruction
        if reconstruction:
            preds, out_mask = self.reconstruct(x, mae_tokens, mae_tokens_mask)
            
            return preds, out_mask
        else:
            x = self.to_logits(x)
            return x
