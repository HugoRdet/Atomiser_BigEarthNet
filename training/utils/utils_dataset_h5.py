import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from .FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
from .lookup_positional import*

 
def _init_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset
    # open the file once per worker, and keep it around on `ds.h5`
    ds.h5 = h5py.File(ds.file_path, 'r')

def del_file(path):
    if os.path.exists(path):
        os.remove(path)

def normalize_image(img, means, stds):
    return (img - means) / stds

def rotate_image_torch(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return image.rot90(1, [-2, -1])
    elif angle == 180:
        return image.rot90(2, [-2, -1])
    elif angle == 270:
        return image.rot90(3, [-2, -1])
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270 degrees.")
    
def flip_image_torch(image, horizontal=False, vertical=False):
    if horizontal:
        image = image.flip(-1)
    if vertical:
        image = image.flip(-2)
    return image

def compute_channel_mean_std_dico(dico_idxs,ds):
    """
    Computes the mean and std for each of the 12 channels in a large list of tensors.
    Each tensor in the list has shape (12, 120, 120).

    Args:
        tensor_list (list): A list of PyTorch tensors, each with shape (12, 120, 120).

    Returns:
        A tensor of shape (12, 2), where [:, 0] contains the means and [:, 1] the std.
    """

    # We assume each tensor is (12, 120, 120)
    # so each channel has 120 * 120 elements per tensor.
    # We'll accumulate sums and sums of squares in double precision to reduce numerical issues.
    
    num_channels = 14
    sums = torch.zeros(num_channels, dtype=torch.float64)
    sums_of_squares = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0  # total number of pixels across all tensors for each channel

    for key in dico_idxs:
        
        L_idxs=dico_idxs[key]
        
        
        for idx in L_idxs:
            
            tensor,_=ds[idx]
            # Ensure the tensor is on CPU and in float64 for stable accumulations
            t = tensor.to(dtype=torch.float64)
    
            # Sum over spatial dimensions (dim=1,2) => shape [12]
            channel_sums = t.sum(dim=(1, 2))
            # Sum of squares over spatial dimensions
            channel_sums_of_squares = (t * t).sum(dim=(1, 2))
    
            sums += channel_sums
            sums_of_squares += channel_sums_of_squares
            # Each tensor contributes 120 * 120 pixels per channel
            total_pixels += t.shape[1] * t.shape[2]

    # Compute mean per channel
    mean = sums / total_pixels

    # Compute variance = E[X^2] - (E[X])^2
    # E[X^2] = sums_of_squares / total_pixels
    var = (sums_of_squares / total_pixels) - mean**2

    # To be safe against negative rounding errors, clamp var to 0
    var = var.clamp_min(0.0)
    std = var.sqrt()

    # Stack into (12, 2): first column is mean, second is std
    stats = torch.stack((mean, std), dim=-1).float()

    


    return stats

import torch

def random_rotate_flip(image: torch.Tensor):
    """
    Apply random rotation (0°, 90°, 180°, 270°) and random horizontal/vertical flip
    to a satellite image of shape [C=12, H=120, W=120].

    Args:
        image (torch.Tensor): Tensor of shape [12, H, W]

    Returns:
        torch.Tensor: Transformed image with same shape
    """
    assert image.ndim == 3 and image.shape[0] == 12, "Expected shape [12, H, W]"

    # Random rotation: 0, 90, 180, 270 degrees (implemented as number of 90° rotations)
    k = torch.randint(0, 4, (1,)).item()
    image = torch.rot90(image, k=k, dims=[1, 2])

    # Random horizontal flip
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[2])  # Flip width (left-right)

    # Random vertical flip
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[1])  # Flip height (top-bottom)

    return image


def compute_channel_mean_std(dico_idxs,ds):
    if dico_idxs!=None:
        return compute_channel_mean_std_dico(dico_idxs,ds)
    
    num_channels = 14
    sums = torch.zeros(num_channels, dtype=torch.float64)
    sums_of_squares = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0  # total number of pixels across all tensors for each channel

    for idx in range(len(ds)):
        
        tensor,_=ds[idx]
        # Ensure the tensor is on CPU and in float64 for stable accumulations
        t = tensor.to(dtype=torch.float64)

        # Sum over spatial dimensions (dim=1,2) => shape [12]
        channel_sums = t.sum(dim=(1, 2))
        # Sum of squares over spatial dimensions
        channel_sums_of_squares = (t * t).sum(dim=(1, 2))

        sums += channel_sums
        sums_of_squares += channel_sums_of_squares
        # Each tensor contributes 120 * 120 pixels per channel
        total_pixels += t.shape[1] * t.shape[2]

    # Compute mean per channel
    mean = sums / total_pixels

    # Compute variance = E[X^2] - (E[X])^2
    # E[X^2] = sums_of_squares / total_pixels
    var = (sums_of_squares / total_pixels) - mean**2

    # To be safe against negative rounding errors, clamp var to 0
    var = var.clamp_min(0.0)
    std = var.sqrt()

    # Stack into (12, 2): first column is mean, second is std
    stats = torch.stack((mean, std), dim=-1).float()
    torch.save(stats, "./data/stats.pt")
    return stats


def create_dataset(dico_idxs, ds,df, name="tiny", mode="train",trans_config=None,trans_tokens=None, stats=None,max_len=-1):
    if dico_idxs!=None:
        return create_dataset_dico(dico_idxs, ds, name, mode,trans_config,trans_tokens=trans_tokens, stats=stats)
    
    # 1) Clean up any existing file
    

    h5_path = f'./data/Tiny_BigEarthNet/{name}_{mode}.h5'
    del_file(h5_path)
    db = h5py.File(h5_path, 'w')
    
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
        if os.path.exists("data/normalisation/stats.pt"):
            stats=torch.load("data/normalisation/stats.pt",weights_only=True)
            print("ok!")
        else:
            stats = compute_channel_mean_std( dico_idxs,ds)

    ids=set()
    dico_stats=dict()


        

    # Make sure stats is a torch float tensor
    stats = stats.float()
    # Separate mean/std for easy broadcasting
    means = stats[:, 0].view(-1, 1, 1)  # shape: [12, 1, 1]
    stds = stats[:, 1].view(-1, 1, 1)   # shape: [12, 1, 1]

    # 3) Create a new HDF5 file
    cpt_train = 0
    

    if max_len!=-1:
        max_len=int(len(ds)*max_len)    



    for elem_id in tqdm(range(len(ds))):
        
        if cpt_train>max_len and max_len!=-1:
            print("oui Milgram",cpt_train,"   ",max_len)
            print(cpt_train)
            break

        

        split=get_split(df,elem_id)

        if split!=mode:
            continue
        
        img, label = ds[elem_id]  # shape (12, 120, 120)


        tmp_id=get_one_hot_indices(label)
        for tmp_id_elem in tmp_id:
            if not tmp_id_elem in dico_stats:
                dico_stats[tmp_id_elem]=1
            else:
                dico_stats[tmp_id_elem]+=1
        

        if trans_config!=None:
            trans_config.create_transform_image_dico(int(elem_id),mode="train",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="test",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="validation",modality_folder=mode)

      

    

        # Convert to float (if needed) before normalization
        img = img.float()

        # Apply per-channel normalization
        # normalized_value = (value - mean[channel]) / std[channel]
        img = (img - means) / stds



        # Convert back to numpy to store in HDF5
        db.create_dataset(f'image_{cpt_train}', data=img.numpy().astype(np.float16))
        db.create_dataset(f'label_{cpt_train}', data=label.numpy().astype(np.float16))
        db.create_dataset(f'id_{cpt_train}', data=int(elem_id))


        cpt_train += 1


    db.close()

    return stats


def get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode,modality_mode):
    tmp_img=img[2:,:,:]


    tmp_mask=torch.ones(tmp_img.shape)
    tmp_img,tmp_mask=trans_config.apply_transformations(tmp_img,tmp_mask,int(elem_id),mode=mode,modality_mode=modality_mode)
    tmp_img,tmp_mask=trans_tokens.process_data(tmp_img.unsqueeze(0),tmp_mask.unsqueeze(0))
    tmp_img=tmp_img.squeeze(0)
    tmp_mask=tmp_mask.squeeze(0)

    cond=tmp_mask==1
    image=tmp_img[cond]
    return int(image.shape[0])

def create_dataset_dico(dico_idxs, ds, name="tiny", mode="train",trans_config=None,trans_tokens=None, stats=None):
    """
    Creates an HDF5 dataset using the given sample indices (dico_idxs) from ds.
    If stats (per-channel mean/std) is None, computes it on-the-fly in a streaming fashion.
    Then applies normalization to each image: (image - mean) / std

    Args:
        dico_idxs (dict): Mapping from some key to a list of sample indices
        ds: A dataset that supports ds[idx] -> (image, label)
            where image is shape (12, 120, 120).
        name (str): HDF5 file prefix
        mode (str): e.g. "train" or "test"
        stats (torch.Tensor or None): shape (12,2) with [:,0] as mean, [:,1] as std

    Returns:
        stats (torch.Tensor): The per-channel mean/std used for normalization
    """

    # 1) Clean up any existing file
    h5_path = f'./data/Tiny_BigEarthNet/{name}_{mode}.h5'
    del_file(h5_path)
    ids=set()
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
   
        stats = compute_channel_mean_std( dico_idxs,ds)
        

    # Make sure stats is a torch float tensor
    stats = stats.float()
    # Separate mean/std for easy broadcasting
    means = stats[:, 0].view(-1, 1, 1)  # shape: [12, 1, 1]
    stds = stats[:, 1].view(-1, 1, 1)   # shape: [12, 1, 1]

    # 3) Create a new HDF5 file
    db = h5py.File(h5_path, 'w')

    cpt_train = 0

    # 4) Iterate through your dictionary of IDs, fetch images, and store them
    for idx in tqdm(dico_idxs.keys()):
        l_samples = dico_idxs[idx]

        for elem_id in l_samples:
            img, label = ds[elem_id]  # shape (12, 120, 120)
        

            trans_config.create_transform_image_dico(int(elem_id),mode="train",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="test",modality_folder=mode)
            trans_config.create_transform_image_dico(int(elem_id),mode="validation",modality_folder=mode)

            # Convert to float (if needed) before normalization
            img = img.float()

            # Apply per-channel normalization
            # normalized_value = (value - mean[channel]) / std[channel]
            img = (img - means) / stds

            #shape_train=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="train")
            #shape_validation=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="validation")
            #shape_test=get_img_shape_trans(img,elem_id,trans_config,trans_tokens,mode=mode,modality_mode="test")

            
            

            # Convert back to numpy to store in HDF5
            db.create_dataset(f'image_{cpt_train}', data=img.numpy().astype(np.float16))
            db.create_dataset(f'label_{cpt_train}', data=label.numpy().astype(int))
            db.create_dataset(f'id_{cpt_train}', data=int(elem_id))
            db.create_dataset(f'shape_train_{cpt_train}', data=int(0))
            db.create_dataset(f'shape_test_{cpt_train}', data=int(0))
            db.create_dataset(f'shape_validation_{cpt_train}', data=int(0))

            

            cpt_train += 1

    db.close()
    return stats










class Tiny_BigEarthNet(Dataset):
    def __init__(self, file_path, 
                 transform,
                 transform_tokens=None,
                 model="None",
                 mode="train",
                 modality_mode=None,
                 fixed_size=None,
                 fixed_resolution=None,
                 dataset_config=None,
                 config_model=None,
                 look_up=None):
        
        self.file_path = file_path
        self.num_samples = None
        self.mode=mode
        self.shapes=[]
        self._initialize_file()
        self.transform=transform
        self.model=model
        self.transform_tokens=transform_tokens
        self.original_mode=mode
        self.fixed_size=fixed_size
        self.fixed_resolution=fixed_resolution
        self.bands_info=dataset_config
        self.bandwidths=torch.zeros(12)
        self.wavelengths=torch.zeros(12)
        self.config_model=config_model
        self.nb_tokens=self.config_model["trainer"]["max_tokens"]
        self.look_up=look_up

        self.prepare_band_infos()
        

        if modality_mode==None:
            self.modality_mode=mode
            self.original_mode=mode
        else:
            self.modality_mode=modality_mode
            self.original_mode=modality_mode

        self.h5=None




    def _initialize_file(self):
     
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 6  # Nombre d'échantillons

            #for idx in range(self.num_samples):
            #    shape_key = int(f[f'shape_{self.mode}_{idx}'][()])
            #    self.shapes.append(shape_key)


  

    def __len__(self):
        return self.num_samples
    

    def set_modality_mode(self,mode):
        self.modality_mode=mode

    def reset_modality_mode(self):
        self.modality_mode=self.original_mode

    def prepare_band_infos(self):
        
        for idx,band in enumerate(self.bands_info["bands_sen2_info"]):
            band_data=self.bands_info["bands_sen2_info"][band]
            self.bandwidths[idx]=band_data["bandwidth"]
            self.wavelengths[idx]=band_data["central_wavelength"]
            
    


            

    def __getitem__(self, idx):


        image=None
        label=None
        id_img=None



        f = self.h5


        image = torch.tensor(f[f'image_{idx}'][:]) #14;120;120
        image =random_rotate_flip(image[2:,:,:])
        attention_mask=torch.ones(image.shape)
        label = torch.tensor(f[f'label_{idx}'][:])
        id_img = int(f[f'id_{idx}'][()])



        image,attention_mask,new_resolution=self.transform.apply_transformations(image,attention_mask,id_img,mode=self.mode,modality_mode=self.modality_mode,f_s=self.fixed_size,f_r=self.fixed_resolution)

        new_size=image.shape[1]

        
        

        if self.model == "Atomiser":
            #12:size:size
            image_size = image.shape[-1]

            tmp_resolution = int(10.0/new_resolution*1000)
            resolution_tmp=10.0/new_resolution
            
            
            
            
            # Get global offset for this modality
            global_offset = self.look_up.table[(tmp_resolution, image_size)]
            
            p_x=torch.linspace(-image_size/2.0*resolution_tmp,image_size/2.0*resolution_tmp,image_size)
            p_x=p_x/1200
            
            p_y=torch.linspace(-image_size/2.0*resolution_tmp,image_size/2.0*resolution_tmp,image_size)
            p_y=p_y/1200
            
            # Create LOCAL pixel indices (0 to image_size-1)
            #y_indices, x_indices = torch.meshgrid(
            #    torch.arange(image_size), 
            #    torch.arange(image_size), 
            #    indexing="ij"
            #)
            
            # Create LOCAL pixel indices (0 to image_size-1)
            y_indices, x_indices = torch.meshgrid(
                p_x,
                p_y,
                indexing="ij"
            )
            
            # Convert to GLOBAL indices by adding offset
            #x_indices = x_indices + global_offset
            #y_indices = y_indices + global_offset
            
            # Expand for all bands
            x_indices = repeat(x_indices.unsqueeze(0), "u h w -> (u r) h w", r=12).unsqueeze(-1)
            y_indices = repeat(y_indices.unsqueeze(0), "u h w -> (u r) h w", r=12).unsqueeze(-1)
            
            # Prepare other token data
            tmp_bandwidths = repeat(
                self.bandwidths.clone().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size
            )
            tmp_wavelengths = repeat(
                self.wavelengths.clone().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size
            )
            
            # Concatenate all token data
            image = torch.cat([
                image.unsqueeze(-1),      # Band values
                x_indices.float(),        # Global X indices
                y_indices.float(),        # Global Y indices  
                tmp_bandwidths,           # Bandwidths
                tmp_wavelengths,          # Wavelengths
            ], dim=-1)
            
            # Reshape and sample tokens
            image = rearrange(image, "b h w c -> (b h w) c")
            tmp_rand = torch.randperm(image.shape[0])
            image = image[tmp_rand[:self.nb_tokens]]
            attention_mask = torch.ones(image.shape[0])
            
            # Handle padding if needed
            if image.shape[0] < self.nb_tokens:
                padding_tokens = repeat(
                    image[0].clone().unsqueeze(0),
                    "n d -> (n r) d", r=self.nb_tokens-image.shape[0]
                )
                padding_mask = torch.zeros((self.nb_tokens-image.shape[0]))
                
                image = torch.cat([image, padding_tokens], dim=0)
                attention_mask = torch.cat([attention_mask, padding_mask], dim=0)
            
            return image, attention_mask, new_resolution, new_size, label, id_img
        
        return image, attention_mask, new_resolution, new_size, label, id_img
    
  
    

import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist
import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist

class DistributedShapeBasedBatchSampler(Sampler):
    """
    A distributed batch sampler that groups samples by shape and partitions the batches
    across GPUs. Each process only sees a subset of the batches based on its rank.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, rank=None, world_size=None,mode="train"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.mode=mode


        # Set up distributed parameters.
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank=0
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        self.rank = rank
        self.world_size = world_size

        # Group indices by image shape.
        self.shape_to_indices = {}
        # Use a temporary DataLoader to iterate over the dataset (batch_size=1).
        #loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
        for idx in tqdm(range(len(dataset)), desc="Sampler initialization"):
            # Assuming each sample returns (image, label, ...); adjust as needed.
            image_shape = dataset.Sampler_building(idx,mode=self.mode)
            # Convert image.shape (a torch.Size) to a tuple so it can be used as a key.
            shape_key = image_shape
            self.shape_to_indices.setdefault(shape_key, []).append(idx)
        
        # Create batches from the groups.
        self.batches = []
        for indices in tqdm(self.shape_to_indices.values(), desc="Batch creation"):
            random.shuffle(indices)
            # Create batches for this shape group.
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    self.batches.append(batch)
        
        if self.shuffle:
            random.shuffle(self.batches)
        
        # Make sure total number of batches is divisible by the number of processes.
        total_batches = len(self.batches)
        remainder = total_batches % self.world_size
        if remainder != 0:
            if not self.drop_last:
                # Pad with extra batches (repeating from the beginning) so each process has equal work.
                pad_size = self.world_size - remainder
                self.batches.extend(self.batches[:pad_size])
                total_batches = len(self.batches)
            else:
                # If dropping last incomplete batches, remove the excess.
                total_batches = total_batches - remainder
                self.batches = self.batches[:total_batches]
        self.total_batches = total_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for i in range(self.rank, self.total_batches, self.world_size):
            batch = self.batches[i]
            yield batch

    def __len__(self):
        # Number of batches that this process will iterate over.
        return self.total_batches // self.world_size



import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist


class Tiny_BigEarthNetDataModule(pl.LightningDataModule):
    def __init__(self, path,
                trans_modalities,
                trans_tokens=None, 
                model="None", 
                batch_size=32, 
                num_workers=4,
                modality=None,
                ds=None,
                dataset_config=None,
                config_model=None,
                look_up=None):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_validation.h5"
        self.test_file = path + "_test.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_modalities = trans_modalities
        self.model = model
        self.modality=modality
        self.trans_tokens=trans_tokens
        self.dataset_config=dataset_config
        self.config_model=config_model
        self.look_up=look_up
        
 

    def setup(self, stage=None):
        

        
        
        self.train_dataset = Tiny_BigEarthNet(
            self.train_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="train",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

            
        self.val_dataset = Tiny_BigEarthNet(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        self.val_dataset_mode_train = Tiny_BigEarthNet(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        self.val_dataset_mode_train.modality_mode="train"

        if self.modality!=None:
            self.val_dataset.modality_mode=self.modality
            
        self.test_dataset = Tiny_BigEarthNet(
            self.test_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="test",
            modality_mode=self.modality,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up
        )

        if self.modality!=None:
            self.test_dataset.modality_mode=self.modality


    def train_dataloader(self):
        # Create the custom distributed sampler inside the DataLoader call.

        if self.modality==None:
      
            self.modality="train"

    
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )


    def val_dataloader(self):

        if self.modality==None:
            self.modality="validation"

        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")

        val_mod_val=DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )

        val_mod_train=DataLoader(
            self.val_dataset_mode_train,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )
        return [val_mod_val,val_mod_train]


    def test_dataloader(self):


        if self.modality==None:
            self.modality="test"

        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            worker_init_fn=_init_worker,
            drop_last=False,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            prefetch_factor=4,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )



class Tiny_BigEarthNetDataModule_test_RS(pl.LightningDataModule):
    def __init__(self, path,
                trans_modalities,
                trans_tokens=None, 
                model="None", 
                batch_size=32, 
                num_workers=4,
                modality=None,
                ds=None,
                fixed_resolution=None,
                fixed_size=None,
                dataset_config=None):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_validation.h5"
        self.test_file = path + "_test.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_modalities = trans_modalities
        self.model = model
        self.modality=modality
        self.trans_tokens=trans_tokens
        self.fixed_resolution=fixed_resolution
        self.fixed_size=fixed_size
        self.dataset_config=dataset_config
 

    def setup(self, stage=None):

        
        
        self.train_dataset = Tiny_BigEarthNet(
            self.train_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="train",
            fixed_resolution=self.fixed_resolution,
            fixed_size=self.fixed_size,
            dataset_config=self.dataset_config
        )

        

            
        self.val_dataset = Tiny_BigEarthNet(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config
        )

        self.val_dataset_mode_train = Tiny_BigEarthNet(
            self.val_file,
            transform=self.trans_modalities,
            transform_tokens=self.trans_tokens,
            model=self.model,
            mode="validation",
            dataset_config=self.dataset_config
        )

        self.val_dataset_mode_train.modality_mode="train"

        if self.modality!=None:
            self.val_dataset.modality_mode=self.modality


        if self.fixed_resolution!=None:
            self.test_dataset=[]

            for resolution in self.fixed_resolution:
            
                test_dataset = Tiny_BigEarthNet(
                    self.test_file,
                    transform=self.trans_modalities,
                    transform_tokens=self.trans_tokens,
                    model=self.model,
                    mode="test",
                    modality_mode=self.modality,
                    fixed_resolution=resolution,
                    fixed_size=-1,
                    dataset_config=self.dataset_config
                )

                self.test_dataset.append(test_dataset)

        if self.fixed_resolution==None:
            self.test_dataset=[]

            for size in self.fixed_size:
            
                test_dataset = Tiny_BigEarthNet(
                    self.test_file,
                    transform=self.trans_modalities,
                    transform_tokens=self.trans_tokens,
                    model=self.model,
                    mode="test",
                    modality_mode=self.modality,
                    fixed_resolution=-1,
                    fixed_size=size,
                    dataset_config=self.dataset_config
                )

                self.test_dataset.append(test_dataset)

        if self.modality!=None:
            for k in self.test_dataset:
                k.modality_mode=self.modality

    def train_dataloader(self):
        # Create the custom distributed sampler inside the DataLoader call.

        if self.modality==None:
      
            self.modality="train"

      
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )


    def val_dataloader(self):

        if self.modality==None:
            self.modality="validation"

        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")

        val_mod_val=DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )

        val_mod_train=DataLoader(
            self.val_dataset_mode_train,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            #batch_sampler=batch_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
            prefetch_factor=8,  # increased prefetch for smoother transfers
            persistent_workers=True  # avoid worker restart overhead
        )
        return [val_mod_val,val_mod_train]


    def test_dataloader(self):


        if self.modality==None:
            self.modality="test"

  
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")

        res=[]

        for dataset in self.test_dataset:
            tmp_dataloader=DataLoader(
                dataset,
                num_workers=1,
                batch_size=self.batch_size,
                worker_init_fn=_init_worker,
                drop_last=False,
            )

            res.append(tmp_dataloader)
        
        return res
