import torch
from .utils_dataset import read_yaml,save_yaml
from .image_utils import *
from .files_utils import*
from math import pi
import einops 
import datetime
import numpy as np
import datetime
from torchvision.transforms.functional import rotate, hflip, vflip
import random
import torch.nn as nn
import time
import math
from tqdm import tqdm
from torch.profiler import record_function



def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands==-1:
        scales = torch.linspace(1., max_freq , max_freq, device = device, dtype = dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class transformations_config(nn.Module):

    

    def __init__(self,bands_yaml,config,lookup_table):
        super().__init__()
      
        self.bands_yaml=read_yaml(bands_yaml)
        self.bands_sen2_infos=self.bands_yaml["bands_sen2_info"]
        self.s2_waves=self.get_wavelengths_infos(self.bands_sen2_infos)
        self.s2_res_tmp=self.get_resolutions_infos(self.bands_sen2_infos)
        self.register_buffer("positional_encoding_s2", None)
        self.register_buffer('wavelength_processing_s2', None)
        #self.register_buffer("positional_encoding_fourrier", None)
        self.resolutions_x_sizes_cached={}
        
        
        


        
        self.lookup_table=lookup_table
        

        self.register_buffer(
            "s2_res",
            torch.tensor(self.s2_res_tmp, dtype=torch.float32),
            persistent=True
        )
  

        self.config=config
  
        self.nb_tokens_limit=config["trainer"]["max_tokens"]

        self.gaussian_means=[]
        self.gaussian_stds=[]

        if "wavelengths_encoding" in self.config:
            for gaussian_idx in self.config["wavelengths_encoding"]:
                self.gaussian_means.append(self.config["wavelengths_encoding"][gaussian_idx]["mean"])
                self.gaussian_stds.append(self.config["wavelengths_encoding"][gaussian_idx]["std"])
        
        self.gaussian_means=torch.Tensor(np.array(self.gaussian_means)).to(torch.float32).view(1, -1)
        self.gaussian_stds=torch.Tensor(np.array(self.gaussian_stds)).to(torch.float32).view(1, -1)
        
        
        
        
        

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
        

    def get_wavelengths_infos(self,bands_info):
        bandwidth=[]
        central_w=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            bandwidth.append(band["bandwidth"])
            central_w.append(band["central_wavelength"])

        return np.array(bandwidth),np.array(central_w)
    
    def get_resolutions_infos(self,bands_info):
        res=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            res.append(band["resolution"])


        return torch.from_numpy(np.array([20 for _ in range(12)]))
   


        

        




    def get_band_identifier(self,bands,channel_idx):
        for band_key in bands:
            band=bands[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None

    
    def get_band_infos(self,bands,band_identifier):
        return bands[band_identifier]



    
    def pos_encoding(self,size,positional_scaling=None,max_freq=4,num_bands = 4,device="cpu"):


        axis = torch.linspace(-positional_scaling/2.0, positional_scaling/2.0, steps=size,device=device)
        pos=fourier_encode(axis,max_freq=max_freq,num_bands = num_bands)
        
        return pos
    
    
    def get_positional_encoding_fourrier(
        self,
        size,             # e.g. (B_size, T_size, H, W, C)
        resolution: float,      # shape [C], base resolution per band
        resolution_factor: float,  # shape [B_size], factor per sample
        device
        ):
        """
        Returns: Tensor [B_size, T_size, H, W, C, D]
        resolution: [C]
        resolution_factor: [B_size]
        """

        # spatial size (assume H == W)
   

        # -- 1) compute per-sample, per-band new resolution: [B, C]
        #    new_res[i, b] = resolution[b] / resolution_factor[i]
        new_res = resolution / resolution_factor

        # -- 2) compute positional scaling per band: [B, C]
        pos_scalings = (size * new_res) / 400.0
  
        max_freq  = self.config["Atomiser"]["pos_max_freq"]
        num_bands = self.config["Atomiser"]["pos_num_freq_bands"]

        
        
        raw = self.pos_encoding(
            size,
            positional_scaling=pos_scalings,
            max_freq=max_freq,
            num_bands=num_bands,
            device=device
        )
        
        return raw

    def get_gaussian_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        num_gaussians: int,        
        sigma: float,
        device=None
    ):
        """
        Compute Gaussian encoding for tokens with explicit position information.
        
        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        device = device or token_data.device
        batch, tokens, _ = token_data.shape
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_{num_gaussians}_{sigma}_{device}"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_gaussian_encodings(num_gaussians, sigma, device, cache_key)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_x_indices = token_data[..., 1].long()  # [batch, tokens]
        global_y_indices = token_data[..., 2].long()  # [batch, tokens]
        
        # Direct lookup using global indices
        encoding_x = cached_encoding[global_x_indices]  # [batch, tokens, num_gaussians]
        encoding_y = cached_encoding[global_y_indices]  # [batch, tokens, num_gaussians]
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_x, encoding_y], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        return result

    def _precompute_global_gaussian_encodings(self, num_gaussians, sigma, device, cache_key):
        """
        Precompute Gaussian encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        
        # Create Gaussian centers (same for all positions)
        centers = torch.linspace(-1200.0, 1200.0, num_gaussians, device=device)
        
        # Initialize global encoding tensor
        global_encoding = torch.zeros(max_global_index, num_gaussians, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            # Get global offset for this modality
            
            
            
            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            
            # Create physical coordinates for this modality's pixels
            physical_coords = torch.linspace(
                (-image_size/2.) * resolution, 
                (image_size/2.) * resolution, 
                steps=image_size, 
                device=device
            )
            
            # Compute Gaussian encoding for this modality
            modality_encoding = self._compute_1d_gaussian_encoding_vectorized(
                physical_coords, resolution/2.0, centers, sigma, device
            )  # [image_size, num_gaussians]
            
            # Place encodings at correct global indices
            global_encoding[global_offset:global_offset + image_size] = modality_encoding
        
        # Store the global encoding
        setattr(self, cache_key, global_encoding)
        
    def get_fourrier_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        device=None
    ):
        """
        Compute Fourier encoding for tokens with explicit position information.

        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_fourrier"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_fourrier_encodings(device)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_x_indices = token_data[..., 1].long()  # [batch, tokens]
        global_y_indices = token_data[..., 2].long()  # [batch, tokens]
        
        # Direct lookup using global indices
        encoding_x = cached_encoding[global_x_indices]  # [batch, tokens, num_gaussians]
        encoding_y = cached_encoding[global_y_indices]  # [batch, tokens, num_gaussians]
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_x, encoding_y], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        
        return result

    def _precompute_global_gaussian_encodings(self, num_gaussians, sigma, device, cache_key):
        """
        Precompute Gaussian encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        
        # Create Gaussian centers (same for all positions)
        centers = torch.linspace(-1200.0, 1200.0, num_gaussians, device=device)
        
        # Initialize global encoding tensor
        global_encoding = torch.zeros(max_global_index, num_gaussians, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            # Get global offset for this modality
            
            
            
            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            
            # Create physical coordinates for this modality's pixels
            physical_coords = torch.linspace(
                (-image_size/2.) * resolution, 
                (image_size/2.) * resolution, 
                steps=image_size, 
                device=device
            )
            
            # Compute Gaussian encoding for this modality
            modality_encoding = self._compute_1d_gaussian_encoding_vectorized(
                physical_coords, resolution/2.0, centers, sigma, device
            )  # [image_size, num_gaussians]
            
            # Place encodings at correct global indices
            global_encoding[global_offset:global_offset + image_size] = modality_encoding
        
        # Store the global encoding
        setattr(self, cache_key, global_encoding)
        
        
    def _precompute_global_fourrier_encodings(self, device):
        """
        Precompute fourrier encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        
        # Initialize global encoding tensor
        num_bands = self.config["Atomiser"]["pos_num_freq_bands"]* 2 + 1
        global_encoding = torch.zeros(max_global_index, num_bands, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            # Get global offset for this modality
            pos=self.get_positional_encoding_fourrier(image_size,10.0,resolution,device)
            
            
            
            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            
            
            
            # Place encodings at correct global indices
            global_encoding[global_offset:global_offset + image_size] = pos
        
        # Store the global encoding
        cache_key = f"positional_encoding_fourrier"
        setattr(self, cache_key, global_encoding)
        

    def _compute_1d_gaussian_encoding_vectorized(self, positions, half_res, centers, sigma, device):
        """
        Vectorized computation of 1D Gaussian encoding.
        
        Args:
            positions: [size] - physical positions for each pixel
            half_res: float - half resolution
            centers: [num_gaussians] - Gaussian centers
            sigma: float - Gaussian sigma
            
        Returns:
            encoding: [size, num_gaussians]
        """
        sqrt_2 = math.sqrt(2)
        norm_factor = sigma * math.sqrt(math.pi / 2)
        inv_sqrt2_sigma = 1.0 / (sqrt_2 * sigma)
        
        # Expand dimensions for vectorized computation
        positions_exp = positions.unsqueeze(-1)  # [size, 1]
        centers_exp = centers.unsqueeze(0)       # [1, num_gaussians]
        
        # Compute pixel bounds
        lower_bound = positions_exp - half_res   # [size, 1]
        upper_bound = positions_exp + half_res   # [size, 1]
        
        # Vectorized computation of error function bounds
        lower_erf_input = (lower_bound - centers_exp) * inv_sqrt2_sigma  # [size, num_gaussians]
        upper_erf_input = (upper_bound - centers_exp) * inv_sqrt2_sigma  # [size, num_gaussians]
        
        # Compute integral using error function
        encoding = norm_factor * (torch.erf(upper_erf_input) - torch.erf(lower_erf_input))
        
        # L2 normalize
        encoding = encoding / (encoding.norm(dim=-1, keepdim=True) + 1e-8)
        
        return encoding  # [size, num_gaussians]




    def apply_temporal_spatial_transforms(self, img, mask):
        """
        Apply random 90-degree rotation and flip to each item in a batch of 
        [B, T, H, W, C] images and masks. Rotation and flip are applied consistently 
        across all channels and time steps for each sample.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [B, T, H, W, C]
            mask (torch.Tensor): Input mask tensor of shape [B, T, H, W, C]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and mask tensors
        """

        B, T, H, W, C = img.shape
        img = img.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        mask = mask.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

        img_out = img.clone()
        mask_out = mask.clone()

        for b in range(B):
            # --- Choose random 90-degree rotation ---
            k = random.randint(0, 3)  # rotate 0, 90, 180, or 270 degrees
            if k > 0:
                img_out[b] = torch.rot90(img_out[b], k=k, dims=(-2, -1))  # rotate over H-W
                mask_out[b] = torch.rot90(mask_out[b], k=k, dims=(-2, -1))

            # --- Random horizontal flip ---
            if random.random() > 0.5:
                img_out[b] = img_out[b].flip(-1)  # flip W
                mask_out[b] = mask_out[b].flip(-1)

            # --- Random vertical flip ---
            if random.random() > 0.5:
                img_out[b] = img_out[b].flip(-2)  # flip H
                mask_out[b] = mask_out[b].flip(-2)

        # Back to [B, T, H, W, C]
        img_out = img_out.permute(0, 2, 3, 4, 1)
        mask_out = mask_out.permute(0, 2, 3, 4, 1)

        return img_out, mask_out






    def fourier_encode_scalar(self,scalar,size,max_freq,num_bands):
        
        tmp_encoding=fourier_encode(torch.Tensor(scalar), max_freq=max_freq, num_bands = num_bands) # B T C
        
        tmp_encoding = tmp_encoding.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, C]
        tmp_encoding = tmp_encoding.expand(-1,    # keep B
                                   -1,    # keep T
                                    size, # H = size
                                    size, # W = size
                                   -1)    # keep C
  
        return tmp_encoding
    
    def scaling_frequencies(self,x):
        return (1000/x)-1.5
    
    
    


    def compute_gaussian_band_max_encoding(self, lambda_centers, bandwidths, num_points=50,modality="S2"):

        
        # Ensure inputs are PyTorch tensors and on the same device as self.gaussian_means.
        device = self.gaussian_means.device

        
        lambda_centers = torch.as_tensor(lambda_centers, dtype=torch.float32, device=device)
        bandwidths = torch.as_tensor(bandwidths, dtype=torch.float32, device=device)

        
        lambda_min = lambda_centers - bandwidths / 2 
        lambda_max = lambda_centers + bandwidths / 2 
        
        
        t = torch.linspace(0, 1, num_points, device=device)  
        
        # Compute the sampled wavelengths for each spectral band using broadcasting.
        # Each spectral band gets its own set of sample points.
        # sampled_lambdas shape: [s, num_points]

        sampled_lambdas = lambda_min.unsqueeze(1) + (lambda_max - lambda_min).unsqueeze(1) * t.unsqueeze(0)

        gaussians = torch.exp(
            -0.5 * (
                ((sampled_lambdas.unsqueeze(2) - self.gaussian_means.unsqueeze(0).unsqueeze(0)) / 
                self.gaussian_stds.unsqueeze(0).unsqueeze(0)) ** 2
            )
        )
        
        # For each spectral band and each Gaussian, find the maximum activation across the sampled points.
        # The max is taken along dim=1 (the num_points axis), returning a tensor of shape [s, num_gaussians].
        
        encoding = gaussians.max(dim=-2).values
       

        
        return encoding


    
    def get_bvalue_processing(self,img):
        if self.config["Atomiser"]["bandvalue_encoding"]=="NATURAL":
                return img.unsqueeze(-1)
        
        elif self.config["Atomiser"]["bandvalue_encoding"]=="FF":
            num_bands=self.config["Atomiser"]["bandvalue_num_freq_bands"]
            max_freq=self.config["Atomiser"]["bandvalue_max_freq"]
            fourier_encoded_values=fourier_encode(img, max_freq, num_bands)
            
            return fourier_encoded_values

    def wavelength_processing(self,device,wavelength,bandwidth,modality="s2",tokens=None):

        id_cache="wavelength_processing_s2"
        encoded = getattr(self, id_cache)

        if encoded is None:
            encoded=self.compute_gaussian_band_max_encoding(wavelength, bandwidth, num_points=50)
            
            tmp_wavelength=torch.from_numpy(wavelength).clone().unsqueeze(-1)
            tmp_bandwidth =torch.from_numpy(bandwidth).clone().unsqueeze(-1)
            encoded=encoded.squeeze(0)

            indexing_array=torch.cat([tmp_bandwidth,tmp_wavelength,encoded],dim=-1)

            
            encoded=indexing_array.to(device)
            id_cache="wavelength_processing_s2"
            setattr(self,id_cache, encoded)
        
      

        # tokens: [B, N, 5]
        # encoded: [M, D] (here D=21)
        B, N, _ = tokens.shape
        M = encoded.shape[0]
        D = encoded.shape[1] - 2

        res = torch.zeros(B, N, D, device=tokens.device)

        # Extract wavelength and bandwidth from tokens
        token_wl = tokens[:, :, 4].unsqueeze(-1)  # [B, N, 1]
        token_bw = tokens[:, :, 3].unsqueeze(-1)  # [B, N, 1]

        # Compare with encoded wavelength and bandwidth
        encoded_wl = encoded[:, 0]  # [M]
        encoded_bw = encoded[:, 1]  # [M]

        # Build a matching mask [B, N, M]
        mask = (token_wl == encoded_wl) & (token_bw == encoded_bw)

        # For each token, get the index of the matching row
        idx = mask.float().argmax(dim=-1)  # [B, N] (index of the match)

        # Gather the encoding values
        res = encoded[idx, 2:]  # Will broadcast to [B, N, D]
        

        
            
        
        return res
    

    def time_processing(self,time_stamp,img_size=-1):
        dt = time_stamp.astype('datetime64[s]')

        years = dt.astype('datetime64[Y]').astype(int) + 1970
        norm_year = (years - 1999) / 27.0

        day_of_year = (dt - dt.astype('datetime64[Y]')) / np.timedelta64(1, 'D')
        norm_day = (day_of_year - 1) / 366.0
       
        return (norm_day.astype(np.float32),norm_year.astype(np.float32))
    
    
    def time_encoding(self,time_stamp,img_size=-1):
 
        norm_year = time_stamp[1]

       
        norm_day = time_stamp[0]
       
        


        y_max_freq=self.config["Atomiser"]["year_max_freq"]
        y_num_bands=self.config["Atomiser"]["year_num_freq_bands"]
        d_max_freq=self.config["Atomiser"]["day_max_freq"]
        d_num_bands=self.config["Atomiser"]["day_num_freq_bands"]

      

            

        if self.config["Atomiser"]["day_encoding"]=="FF":
            y_max_freq=self.config["Atomiser"]["year_max_freq"]
            y_num_bands=self.config["Atomiser"]["year_num_freq_bands"]
            d_max_freq=self.config["Atomiser"]["day_max_freq"]
            d_num_bands=self.config["Atomiser"]["day_num_freq_bands"]

            year_encoding=self.fourier_encode_scalar(norm_year,img_size,y_max_freq,y_num_bands)
            day_encoding=self.fourier_encode_scalar(norm_day,img_size,d_max_freq,d_num_bands)

            time_encoding=torch.cat([year_encoding,day_encoding],dim=-1)

            
            
            return  time_encoding
        return None
         


    def apply_transformations_optique(self, im_sen, mask_sen,resolution,size, mode):
        # --- select band info based on mode ---
        if mode=="s2":
            tmp_infos = self.bands_sen2_infos
            res = self.s2_res

            tmp_bandwidth, tmp_central_wavelength = self.s2_waves
            
        


        #tokens of shape [batch,nb_tokens,5]
        #last dimension:
        #    - bandvalue
        #    - xpos
        #    - ypos
        #    - bandwidth 
        #    - central wavelength

        B_size,T, C = im_sen.shape
    
        # 2) Wavelength encoding
        #with record_function("Atomizer/process_data/get_tokens/wavelength_processing"):

        central_wavelength_processing = self.wavelength_processing(
            im_sen.device,
            tmp_central_wavelength,
            tmp_bandwidth,
            modality=mode,
            tokens=im_sen
        )
        
        #with record_function("Atomizer/process_data/get_tokens/get_bvalue_processing"):

        # 3) Band‑value encoding
        value_processed = self.get_bvalue_processing(im_sen[:,:,0])
        
        #p_x=fourier_encode(im_sen[:,:,1], max_freq=64, num_bands=64)
        #p_y=fourier_encode(im_sen[:,:,2], max_freq=64, num_bands=64)
        p_x=self.get_fourrier_encoding(im_sen,device=im_sen.device)
        

        


        
        
        
        
            
        #print("value ",value_processed.shape,"  ",central_wavelength_processing.shape,"  ",band_post_proc_0.shape,band_post_proc_3.shape)
        

       
        
        
        #with record_function("Atomizer/process_data/get_tokens/cat"):
        tokens = torch.cat([
            value_processed,
            central_wavelength_processing,
            p_x.zero_()
        ], dim=-1)
        
        


      



        return tokens, mask_sen
    
    
    def get_tokens(self,img,mask,resolution,size,mode="optique",modality="s2",wave_encoding=None):
        
  

        if mode=="optique":
            return self.apply_transformations_optique(img,mask,resolution,size,modality)
        

    def process_data(self,img,mask,resolution,size):
        
        L_tokens=[]
        L_masks=[]

        
        
        if self.config["dataset"]["S2"]:
            #with record_function("Atomizer/process_data/apply_temporal_spatial_transforms"):
            #    tmp_img,tmp_mask=self.apply_temporal_spatial_transforms(img, mask)
            
            #with record_function("Atomizer/process_data/get_tokens"):
            tokens_s2,tokens_mask_s2=self.get_tokens(img,mask,resolution,size,mode="optique",modality="s2")
            
            return tokens_s2,tokens_mask_s2


  



