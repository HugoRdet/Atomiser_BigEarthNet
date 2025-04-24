from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

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



class Atomiser(nn.Module):
    def __init__(
        self,
        *,
        config,
        transform,
        depth,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        lattent_attn_depth = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        masking=0,
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
        self.masking=masking
        self.config=config
        self.transform=transform

        
        #fourier_channels = (input_axis * ((self.num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        shape_input_dim_x = self.get_shape_attributes_config("pos")
        shape_input_dim_y = self.get_shape_attributes_config("pos")

      

        
        shape_input_year = self.get_shape_attributes_config("year")
        shape_input_day = self.get_shape_attributes_config("day")
        shape_input_wavelength=self.get_shape_attributes_config("wavelength")
        shape_input_bandvalue=self.get_shape_attributes_config("bandvalue")
        self.nb_classes=config["dataset"]["classes"]

        self.wavelength_bits_size=shape_input_wavelength

        input_dim=shape_input_dim_x+shape_input_dim_y+shape_input_year+shape_input_day+shape_input_wavelength+shape_input_bandvalue
        self.coordinates_start_wl=shape_input_bandvalue
        self.coordinates_end_wl=shape_input_bandvalue+shape_input_wavelength


        
        
        
        # Initialize your parameter
        self.VV = nn.Parameter(torch.empty(shape_input_wavelength))
        nn.init.trunc_normal_(self.VV, mean=0.0, std=0.02, a=-2.0, b=2.0)

        self.VH = nn.Parameter(torch.empty(shape_input_wavelength))
        nn.init.trunc_normal_(self.VH, mean=0.0, std=0.02, a=-2.0, b=2.0)

        
        
        
        # Apply truncated normal initialization with specified parameters




        #https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, mean=0.0, std=0.02, a=-2.0, b=2.0)


        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_rev_cross_attn = lambda: ReverseCrossAttentionBlock(input_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout)
        get_input_attn = lambda: PreNorm(input_dim, LinearAttention(input_dim, heads=4, dim_head=64, dropout=attn_dropout))
        get_rev_ff = lambda: PreNorm(input_dim, FeedForward(input_dim, dropout=ff_dropout))



        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_rev_cross_attn,get_input_attn, get_rev_ff = map(
            cache_fn, (
                get_cross_attn,
                get_cross_ff,
                get_latent_attn,
                get_latent_ff,
                get_rev_cross_attn,
                get_input_attn,
                get_rev_ff
            )
        )
        
        self.layers = nn.ModuleList([])
        #weights are shared except for the first blocs of cross attention / latent attention
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
                get_rev_cross_attn(**cache_args),
                get_input_attn(**cache_args),
                get_rev_ff(**cache_args),
                self_attns
            ]))

        self.lattent_attn_layers = nn.ModuleList([])

        for i in range(lattent_attn_depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.lattent_attn_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

     

        #self.to_logits = nn.Sequential(
        #    Reduce('b n d -> b d', 'mean'),
        #    nn.LayerNorm(latent_dim),
        #    nn.Linear(latent_dim, num_classes)
        #) if final_classifier_head else nn.Identity()

        self.to_logits = nn.Sequential(
            LatentAttentionPooling(latent_dim, heads=4, dim_head=64, dropout=0.1),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

       


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
        

    def get_tokens(self,img,date,mask,mode="optique",modality="s2",wave_encoding=None):
        
  

        if mode=="optique":
            return self.transform.apply_transformations_optique(img,date,mask,modality)
        if mode=="sar":
            return self.transform.apply_transformations_SAR(img,date,mask,modality,wave_encoding=wave_encoding)
                
    def process_data(self,batch):
        L_tokens=[]
        L_masks=[]
        
        img_s2,img_l7,img_mo,img_s1,img_al,date_s2,date_l7,date_mo,date_s1,date_al,mask_s2,mask_l7,mask_mo,mask_s1,mask_al = batch
        
        if self.config["dataset"]["S2"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_s2, mask_s2)
            tokens_s2,tokens_mask_s2=self.get_tokens(tmp_img,date_s2,tmp_mask,mode="optique",modality="s2")
            L_masks.append(tokens_mask_s2)
            L_tokens.append(tokens_s2)





        
        if self.config["dataset"]["L7"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_l7, mask_l7)
            tokens_l7,tokens_mask_l7=self.get_tokens(tmp_img,date_l7,tmp_mask,mode="optique",modality="l7")
            L_masks.append(tokens_mask_l7)
            L_tokens.append(tokens_l7)



        if self.config["dataset"]["MODIS"]:
            tokens_mo,tokens_mask_mo=self.get_tokens(img_mo,date_mo,mask_mo,mode="optique",modality="modis")
            L_masks.append(tokens_mask_mo)
            L_tokens.append(tokens_mo)


        if self.config["dataset"]["S1"]:
            tokens_s1,tokens_mask_s1=self.get_tokens(img_s1,date_s1,mask_s1,mode="sar",modality="s1",wave_encoding=(self.VV,self.VH))
            L_masks.append(tokens_mask_s1)
            L_tokens.append(tokens_s1)


        if self.config["dataset"]["ALOS"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_al, mask_al)
            tokens_al,tokens_mask_al=self.get_tokens(tmp_img,date_al,tmp_mask,mode="sar",modality="alos",wave_encoding=(self.VV,self.VH))
            L_masks.append(tokens_mask_al)
            L_tokens.append(tokens_al)
      



        tokens=torch.cat(L_tokens,dim=1)
        tokens_mask=torch.cat(L_masks,dim=1)


            
        return tokens,tokens_mask




    def forward(self, data, training=False):
        data, tokens_masks = self.process_data(data)
     

        b, *_, device, dtype = *data.shape, data.device, data.dtype

        x = repeat(self.latents, 'n d -> b n d', b=b)

        tokens_masks = tokens_masks.to(bool)
        data[tokens_masks == 0] = 0

  





        # Proceed with the rest of the forward pass
        for cross_attn, cross_ff, rev_cross_attn,input_attn,rev_ff, self_attns in self.layers:

            masked_data = data.clone()
            masked_attention_mask = tokens_masks.clone()
            idxs_masking=None
            if self.masking > 0:
                masked_data, masked_attention_mask,idxs_masking = pruning(masked_data, masked_attention_mask, self.masking)

            x = cross_attn(x, context=masked_data, mask=masked_attention_mask) + x
            x = cross_ff(x) + x


            masked_data = rev_cross_attn(masked_data, latents=x)
            masked_data = input_attn(masked_data) + masked_data
            masked_data = rev_ff(masked_data) + masked_data
            data[:,idxs_masking]=masked_data

            
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        for self_attn, self_ff in self.lattent_attn_layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return self.to_logits(x)

