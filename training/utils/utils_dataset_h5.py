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


def create_dataset(dico_idxs, ds,df, name="tiny", mode="train",trans_config=None, stats=None,max_len=-1):
    if dico_idxs!=None:
        return create_dataset_dico(dico_idxs, ds, name, mode,trans_config, stats)
    
    # 1) Clean up any existing file
    

    h5_path = f'./data/Tiny_BigEarthNet/{name}_{mode}.h5'
    del_file(h5_path)
    db = h5py.File(h5_path, 'w')
    
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
        if os.path.exists("data/stats.pt"):
            stats=torch.load("data/stats.pt",weights_only=True)
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

    if max_len!=None:
        max_len=int(len(ds)*max_len)    



    for elem_id in tqdm(range(len(ds))):
        if cpt_train>max_len and max_len!=-1:
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


def create_dataset_dico(dico_idxs, ds, name="tiny", mode="train",trans_config=None, stats=None):
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
    for idx in dico_idxs.keys():
        l_samples = dico_idxs[idx]

        for elem_id in l_samples:
            img, label = ds[elem_id]  # shape (12, 120, 120)

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
            db.create_dataset(f'label_{cpt_train}', data=label.numpy().astype(int))
            db.create_dataset(f'id_{cpt_train}', data=int(elem_id))

            cpt_train += 1

    db.close()
    return stats










class Tiny_BigEarthNet(Dataset):
    def __init__(self, file_path, transform,model="None",mode="train"):
        self.file_path = file_path
        self.num_samples = None
        self.mode=mode
        self._initialize_file()
        self.transform=transform
        self.model=model
        
        self.modality_mode=mode
        self.original_mode=mode

        self.h5=None




    def _initialize_file(self):
     
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 3  # Nombre d'échantillons

  

    def __len__(self):
        return self.num_samples
    

    def set_modality_mode(self,mode):
        self.modality_mode=mode

    def reset_modality_mode(self):
        self.modality_mode=self.original_mode

    def __getitem__(self, idx):


        image=None
        label=None
        id_img=None


        f = self.h5

        image = torch.tensor(f[f'image_{idx}'][:]) #14;120;120
        image =image[2:,:,:]
        attention_mask=torch.ones(image.shape)
        label = torch.tensor(f[f'label_{idx}'][:])
        id_img = int(f[f'id_{idx}'][()])

        
        image,attention_mask=self.transform.apply_transformations(image,attention_mask,id_img,mode=self.mode,modality_mode=self.modality_mode)

        

        return image,attention_mask, label, id_img
    
    def Sampler_building(self, idx, mode="train"):

        return self.shapes[idx]
  
import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist


class Tiny_BigEarthNetDataModule(pl.LightningDataModule):
    def __init__(self, path, trans_modalities, model="None", batch_size=32, num_workers=4,modality=None,ds=None):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_validation.h5"
        self.test_file = path + "_test.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_modalities = trans_modalities
        self.model = model
        self.modality=modality

    def setup(self, stage=None):
        # Define transformations for the training phase.
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])

        # Initialize the datasets with their transformations.
        # Note: This setup() method is called on each process so the dataset and sampler
        # will be created in the proper distributed context.
        
        self.train_dataset = Tiny_BigEarthNet(
            self.train_file,
            transform=self.trans_modalities,
            model=self.model,
            mode="train",
        )
        if self.modality!=None:
            self.train_dataset.modality_mode=self.modality
            
        self.val_dataset = Tiny_BigEarthNet(
            self.val_file,
            transform=self.trans_modalities,
            model=self.model,
            mode="validation",
        )

        if self.modality!=None:
            self.val_dataset.modality_mode=self.modality
            
        self.test_dataset = Tiny_BigEarthNet(
            self.test_file,
            transform=self.trans_modalities,
            model=self.model,
            mode="test",
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
            batch_size=self.batch_size
            #pin_memory=True,
            #prefetch_factor=8,  # increased prefetch for smoother transfers
            #persistent_workers=True  # avoid worker restart overhead
        )

    def val_dataloader(self):

        if self.modality==None:
            self.modality="validation"

 
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            batch_size=self.batch_size
            #pin_memory=True,
            #prefetch_factor=8,  # increased prefetch for smoother transfers
            #persistent_workers=True  # avoid worker restart overhead
        )

    def test_dataloader(self):


        if self.modality==None:
            self.modality="test"


        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            worker_init_fn=_init_worker,
            batch_size=self.batch_size
            #pin_memory=True,
            #prefetch_factor=4,  # increased prefetch for smoother transfers
            #persistent_workers=True  # avoid worker restart overhead
        )