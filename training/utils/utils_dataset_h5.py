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


def get_dates_data(time_stamps):
    res_months=np.zeros(time_stamps.shape[0])
    res_year=np.zeros(time_stamps.shape[0])

    for id_date in range(time_stamps.shape[0]):
        dt = datetime.fromtimestamp(time_stamps[id_date] / 1000.0, tz=timezone.utc)
        res_months[id_date]=dt.month-1
        res_year[id_date]=(dt.year-2013)/4.0
    
    return res_months,res_year

def create_dataset_prec(name="tiny", mode="train",max_imgs=-1):
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
    h5_path = f'./data/custom_planted/{name}_{mode}.h5'
    del_file(h5_path)

    dico_labels=load_json_to_dict("./data/labels.json")
  

    s1_mean=torch.load("./data/normalisation/s1_mean.pt").numpy()
    s1_std = torch.load("./data/normalisation/s1_std.pt").numpy()

    s2_mean=torch.load("./data/normalisation/s2_mean.pt").numpy()
    s2_std = torch.load("./data/normalisation/s2_std.pt").numpy()

    l7_mean= torch.load("./data/normalisation/l7_mean.pt").numpy()
    l7_std = torch.load("./data/normalisation/l7_std.pt").numpy()
    
    modis_mean= torch.load("./data/normalisation/modis_mean.pt").numpy()
    modis_std = torch.load("./data/normalisation/modis_std.pt").numpy()

    alos_mean= torch.load("./data/normalisation/alos_mean.pt").numpy()
    alos_std = torch.load("./data/normalisation/alos_std.pt").numpy()
 

        



    # 3) Create a new HDF5 file
    db = h5py.File(h5_path, 'w')

    cpt_imgs=0
    ids=get_ids_in_folder(base_path=f"./data/public/1.0.1/{mode}/")
    for file_id in ids:
        data=load_npz_by_id(file_id,base_path=f"./data/public/1.0.1/{mode}/")
        for id in tqdm(range(data["s2"].shape[0])):
            s1=(data["s1"][id]-s1_mean)/s1_std #shape (8,12,12,3) 
            db.create_dataset(f's1_{cpt_imgs}', data=s1.astype(np.float16))

            s2=(data["s2"][id]-s2_mean)/s2_std #shape (8,12,12,10) 
            db.create_dataset(f's2_{cpt_imgs}', data=s2.astype(np.float16))

            l7=(data["l7"][id]-l7_mean)/l7_std #shape (20,4,4,6) 
            db.create_dataset(f'l7_{cpt_imgs}', data=l7.astype(np.float16))

            modis=(data["modis"][id]-modis_mean)/modis_std #shape (60,1,1,7)
            db.create_dataset(f'modis_{cpt_imgs}', data=modis.astype(np.float16))

            alos=(data["alos"][id]-alos_mean)/alos_std #shape (4,4,4,3)
            db.create_dataset(f'alos_{cpt_imgs}', data=alos.astype(np.float16))
            ###################################

            s1_timestamps=data["s1_timestamps"][id] 
            db.create_dataset(f's1_timestamps_{cpt_imgs}', data=s1_timestamps.astype(int))

            s2_timestamps=data["s2_timestamps"][id]
            db.create_dataset(f's2_timestamps_{cpt_imgs}', data=s2_timestamps.astype(int))
                                             
            l7_timestamps=data["l7_timestamps"][id] 
            db.create_dataset(f'l7_timestamps_{cpt_imgs}', data=l7_timestamps.astype(int))

            modis_timestamps=data["modis_timestamps"][id] 
            db.create_dataset(f'modis_timestamps_{cpt_imgs}', data=modis_timestamps.astype(int))

            alos_timestamps=data["alos_timestamps"][id]
            db.create_dataset(f'alos_timestamps_{cpt_imgs}', data=alos_timestamps.astype(int))

            s1_mask=data["s1_mask"][id] 
            db.create_dataset(f's1_mask_{cpt_imgs}', data=s1_mask.astype(np.uint8))

            s2_mask=data["s2_mask"][id] 
            db.create_dataset(f's2_mask_{cpt_imgs}', data=s2_mask.astype(np.uint8))

            l7_mask=data["l7_mask"][id] 
            db.create_dataset(f'l7_mask_{cpt_imgs}', data=l7_mask.astype(np.uint8))

            modis_mask=data["modis_mask"][id]
            db.create_dataset(f'modis_mask_{cpt_imgs}', data=modis_mask.astype(np.uint8))

            alos_mask=data["alos_mask"][id] 
            db.create_dataset(f'alos_mask_{cpt_imgs}', data=alos_mask.astype(np.uint8))

         
            label_id=int(dico_labels[str(data["species"][id])]["id"])
            db.create_dataset(f'label_{cpt_imgs}', data=label_id)
            cpt_imgs+=1

            if max_imgs!=-1 and cpt_imgs>max_imgs:
                db.close()
                return
        



def create_dataset_ancien(name="tiny", mode="train", max_imgs=-1):
    h5_path = f'./data/custom_planted/{name}_{mode}.h5'
    del_file(h5_path)

    dico_labels = load_json_to_dict("./data/labels.json")
    ids = get_ids_in_folder(base_path=f"./data/public/1.0.1/{mode}/")

    # Load normalization stats
    def load_norm(name):
        mean = torch.load(f"./data/normalisation/{name}_mean.pt").numpy()
        std = torch.load(f"./data/normalisation/{name}_std.pt").numpy()
        return mean, std

    s1_mean, s1_std = load_norm("s1")
    s2_mean, s2_std = load_norm("s2")
    l7_mean, l7_std = load_norm("l7")
    modis_mean, modis_std = load_norm("modis")
    alos_mean, alos_std = load_norm("alos")

    # Pre-collect to count total number of samples
    total_samples = sum(load_npz_by_id(fid, base_path=f"./data/public/1.0.1/{mode}/")["s2"].shape[0] for fid in ids)

    if max_imgs != -1:
        total_samples = min(max_imgs, total_samples)

    print(f"Total samples to write: {total_samples}")

    BATCH_SIZE = 128
    buffer = {k: [] for k in shape_dict}


    # Create datasets
    with h5py.File(h5_path, 'w') as db:
        shape_dict = {
            "s1":       (total_samples, 8, 12, 12, 3),
            "s2":       (total_samples, 8, 12, 12, 10),
            "l7":       (total_samples, 20, 4, 4, 6),
            "modis":    (total_samples, 60, 1, 1, 7),
            "alos":     (total_samples, 4, 4, 4, 3),
            "s1_mask":  (total_samples, 8, 12, 12, 3),
            "s2_mask":  (total_samples, 8, 12, 12, 10),
            "l7_mask":  (total_samples, 20, 4, 4, 6),
            "modis_mask": (total_samples, 60, 1, 1, 7),
            "alos_mask":  (total_samples, 4, 4, 4, 3),
            "s1_timestamps": (total_samples, 8),
            "s2_timestamps": (total_samples, 8),
            "l7_timestamps": (total_samples, 20),
            "modis_timestamps": (total_samples, 60),
            "alos_timestamps": (total_samples, 4),
            "label": (total_samples,)
        }

        # Initialize all datasets
        datasets = {
            key: db.create_dataset(key, shape=shape, dtype=np.float16 if "mask" not in key and "timestamps" not in key and key != "label" else np.uint8 if "mask" in key else int,
                                   compression="gzip", compression_opts=4)
            for key, shape in shape_dict.items()
        }

        cpt_imgs = 0
        for file_id in ids:
            data = load_npz_by_id(file_id, base_path=f"./data/public/1.0.1/{mode}/")
            num_samples = data["s2"].shape[0]

            for i in tqdm(range(num_samples)):
                if cpt_imgs >= total_samples:
                    break

                datasets["s1"][cpt_imgs] = ((data["s1"][i] - s1_mean) / s1_std).astype(np.float16)
                datasets["s2"][cpt_imgs] = ((data["s2"][i] - s2_mean) / s2_std).astype(np.float16)
                datasets["l7"][cpt_imgs] = ((data["l7"][i] - l7_mean) / l7_std).astype(np.float16)
                datasets["modis"][cpt_imgs] = ((data["modis"][i] - modis_mean) / modis_std).astype(np.float16)
                datasets["alos"][cpt_imgs] = ((data["alos"][i] - alos_mean) / alos_std).astype(np.float16)

                datasets["s1_mask"][cpt_imgs] = data["s1_mask"][i].astype(bool)
                datasets["s2_mask"][cpt_imgs] = data["s2_mask"][i].astype(bool)
                datasets["l7_mask"][cpt_imgs] = data["l7_mask"][i].astype(bool)
                datasets["modis_mask"][cpt_imgs] = data["modis_mask"][i].astype(bool)
                datasets["alos_mask"][cpt_imgs] = data["alos_mask"][i].astype(bool)

                datasets["s1_timestamps"][cpt_imgs] = (data["s1_timestamps"][i]/1000).astype(np.uint32)
                datasets["s2_timestamps"][cpt_imgs] = (data["s2_timestamps"][i]/1000).astype(np.uint32)
                datasets["l7_timestamps"][cpt_imgs] = (data["l7_timestamps"][i]/1000).astype(np.uint32)
                datasets["modis_timestamps"][cpt_imgs] = (data["modis_timestamps"][i]/1000).astype(np.uint32)
                datasets["alos_timestamps"][cpt_imgs] = (data["alos_timestamps"][i]/1000).astype(np.uint32)

                label_id = int(dico_labels[str(data["species"][i])]["id"])
                datasets["label"][cpt_imgs] = label_id

                cpt_imgs += 1

            if cpt_imgs >= total_samples:
                break


def create_dataset(name="tiny", mode="train", max_imgs=-1):
    import h5py
    import numpy as np
    import torch
    from tqdm import tqdm

    # Remove any existing file
    h5_path = f'./data/custom_planted/{name}_{mode}.h5'
    del_file(h5_path)

    dico_labels = load_json_to_dict("./data/labels.json")
    ids = get_ids_in_folder(base_path=f"./data/public/1.0.1/{mode}/")

    # Load normalization statistics (assumes torch tensors saved as .pt files)
    def load_norm(band_name):
        mean = torch.load(f"./data/normalisation/{band_name}_mean.pt").numpy()
        std = torch.load(f"./data/normalisation/{band_name}_std.pt").numpy()
        return mean, std

    s1_mean, s1_std = load_norm("s1")
    s2_mean, s2_std = load_norm("s2")
    l7_mean, l7_std = load_norm("l7")
    modis_mean, modis_std = load_norm("modis")
    alos_mean, alos_std = load_norm("alos")

    # Pre-calculate total number of samples (from all files)
    total_samples = 0
    for fid in ids:
        data = load_npz_by_id(fid, base_path=f"./data/public/1.0.1/{mode}/")
        total_samples += data["s2"].shape[0]
    if max_imgs != -1:
        total_samples = min(max_imgs, total_samples)
    print(f"Total samples to write: {total_samples}")

    # Define dataset shapes
    shape_dict = {
        "s1":           (total_samples, 8, 12, 12, 3),
        "s2":           (total_samples, 8, 12, 12, 10),
        "l7":           (total_samples, 20, 4, 4, 6),
        "modis":        (total_samples, 60, 1, 1, 7),
        "alos":         (total_samples, 4, 4, 4, 3),
        "s1_mask":      (total_samples, 8, 12, 12, 3),
        "s2_mask":      (total_samples, 8, 12, 12, 10),
        "l7_mask":      (total_samples, 20, 4, 4, 6),
        "modis_mask":   (total_samples, 60, 1, 1, 7),
        "alos_mask":    (total_samples, 4, 4, 4, 3),
        "s1_timestamps":(total_samples, 8),
        "s2_timestamps":(total_samples, 8),
        "l7_timestamps":(total_samples, 20),
        "modis_timestamps": (total_samples, 60),
        "alos_timestamps":  (total_samples, 4),
        "label":        (total_samples,),
        "frequencies":        (total_samples,)
    }

    # Open the HDF5 file and create datasets (no compression)
    with h5py.File(h5_path, 'w') as db:
        datasets = {}
        for key, shape in shape_dict.items():
            if key == "label" or key=="frequencies":
                dtype = int
            elif "mask" in key:
                dtype = np.uint8  # storing booleans as uint8
            elif "timestamps" in key:
                dtype = int
            else:
                dtype = np.float16
            datasets[key] = db.create_dataset(key, shape=shape, dtype=dtype)

        global_index = 0
        # Process each file in a vectorized manner
        for file_id in tqdm(ids, desc="Processing files"):
            data = load_npz_by_id(file_id, base_path=f"./data/public/1.0.1/{mode}/")
            num_samples_file = data["s2"].shape[0]

            # If adding the entire file overshoots max_imgs, trim the arrays accordingly
            if global_index + num_samples_file > total_samples:
                num_samples_file = total_samples - global_index

            alos = data["alos"][:num_samples_file]
            alos_mask = data["alos_mask"][:num_samples_file]

            # Prepare masked log-transform for the first two channels
            alos_db = np.copy(alos[:, :, :, :, :2])
            mask = (alos_mask[:, :, :, :, :2] == 1) & (alos_db != 0)
            alos_db[mask] = 10 * np.log(alos_db[mask] ** 2) - 83.0
            alos[:, :, :, :, :2] = alos_db


            # Process normalization over all samples from this file at once
            s1_norm = ((data["s1"][:num_samples_file] - s1_mean) / s1_std).astype(np.float16)
            s2_norm = ((data["s2"][:num_samples_file] - s2_mean) / s2_std).astype(np.float16)
            l7_norm = ((data["l7"][:num_samples_file] - l7_mean) / l7_std).astype(np.float16)
            modis_norm = ((data["modis"][:num_samples_file] - modis_mean) / modis_std).astype(np.float16)
            alos_norm = ((alos- alos_mean) / alos_std).astype(np.float16)

            # Process masks (cast to uint8 for storage)
            s1_mask = data["s1_mask"][:num_samples_file].astype(np.uint8)
            s2_mask = data["s2_mask"][:num_samples_file].astype(np.uint8)
            l7_mask = data["l7_mask"][:num_samples_file].astype(np.uint8)
            modis_mask = data["modis_mask"][:num_samples_file].astype(np.uint8)
            alos_mask = data["alos_mask"][:num_samples_file].astype(np.uint8)

            # Process timestamps (dividing by 1000 and converting)
            s1_timestamps = (data["s1_timestamps"][:num_samples_file] / 1000).astype(np.uint32)
            s2_timestamps = (data["s2_timestamps"][:num_samples_file] / 1000).astype(np.uint32)
            l7_timestamps = (data["l7_timestamps"][:num_samples_file] / 1000).astype(np.uint32)
            modis_timestamps = (data["modis_timestamps"][:num_samples_file] / 1000).astype(np.uint32)
            alos_timestamps = (data["alos_timestamps"][:num_samples_file] / 1000).astype(np.uint32)

            # Process labels and frequencies.
            labels = np.empty(num_samples_file, dtype=int)
            frequencies = np.empty(num_samples_file, dtype=int)
            for i in range(num_samples_file):
                labels[i] = int(dico_labels[str(data["species"][i])]["id"])
                tmp_freq = dico_labels[str(data["species"][i])]["Frequency"]
                if tmp_freq == "frequent":
                    frequencies[i] = 0
                elif tmp_freq == "common":
                    frequencies[i] = 1
                elif tmp_freq == "rare":
                    frequencies[i] = 2

            # Write all processed arrays to the corresponding slice in the HDF5 datasets
            start_idx = global_index
            
            end_idx = global_index + num_samples_file
            datasets["s1"][start_idx:end_idx] = s1_norm
            datasets["s2"][start_idx:end_idx] = s2_norm
            datasets["l7"][start_idx:end_idx] = l7_norm
            datasets["modis"][start_idx:end_idx] = modis_norm
            datasets["alos"][start_idx:end_idx] = alos_norm



            datasets["s1_mask"][start_idx:end_idx] = s1_mask
            datasets["s2_mask"][start_idx:end_idx] = s2_mask
            datasets["l7_mask"][start_idx:end_idx] = l7_mask
            datasets["modis_mask"][start_idx:end_idx] = modis_mask
            datasets["alos_mask"][start_idx:end_idx] = alos_mask

            datasets["s1_timestamps"][start_idx:end_idx] = s1_timestamps
            datasets["s2_timestamps"][start_idx:end_idx] = s2_timestamps
            datasets["l7_timestamps"][start_idx:end_idx] = l7_timestamps
            datasets["modis_timestamps"][start_idx:end_idx] = modis_timestamps
            datasets["alos_timestamps"][start_idx:end_idx] = alos_timestamps

            datasets["label"][start_idx:end_idx] = labels
            datasets["frequencies"][start_idx:end_idx] = frequencies  # Corrected
            global_index=end_idx

        









import random
import torch
from torchvision.transforms.functional import rotate, hflip, vflip

import torch
from torch.utils.data import Dataset
import h5py
import random
from torchvision.transforms.functional import rotate, hflip, vflip

class CustomPlanted_prec(Dataset):
    def __init__(self, file_path, config=None, trans_config=None, augment=True):
        self.file_path = file_path
        self.config = config
        self.trans_config = trans_config
        self.augment = augment
        self.max_tokens=self.config["trainer"]["max_tokens"]

        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = f["s1"].shape[0]  # all modalities have the same first dimension

    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            # Load modalities (you can combine or subset these as needed)
            #s1 = torch.tensor(f["s1"][idx], dtype=torch.float32)        # [8,12,12,3]


            label= f["label"][idx]
            frequency=f["frequencies"][idx]

            tokens_list=[]
            token_masks_wavelength=[]
            if self.config["dataset"]["S2"]:
                s2 = torch.tensor(f["s2"][idx], dtype=torch.float32)        # [8,12,12,10]
                s2_dates = f["s2_timestamps"][idx]  # Just example — you can build years/months/days here if needed
                s2_mask = torch.tensor(f["s2_mask"][idx], dtype=torch.float32)  # e.g., first timestep, shape [12,12]
                tokens_s2=self.trans_config.apply_transformations_optique(s2,s2_dates,s2_mask,"s2")
                token_masks_wavelength.append(torch.ones(tokens_s2.shape[0]))
                tokens_list.append(tokens_s2)

            if self.config["dataset"]["L7"]:
                l7 = torch.tensor(f["l7"][idx], dtype=torch.float32)        # [20,4,4,6]
                l7_dates = f["l7_timestamps"][idx]  # Just example — you can build years/months/days here if needed
                l7_mask = torch.tensor(f["l7_mask"][idx], dtype=torch.float32)  # e.g., first timestep, shape [12,12]
                tokens_l7=self.trans_config.apply_transformations_optique(l7,l7_dates,l7_mask,"l7")
                token_masks_wavelength.append(torch.ones(tokens_l7.shape[0]))
                tokens_list.append(tokens_l7)

            if self.config["dataset"]["MODIS"]:
                modis = torch.tensor(f["modis"][idx], dtype=torch.float32)        # [20,4,4,6]
                modis_dates = f["modis_timestamps"][idx]  # Just example — you can build years/months/days here if needed
                modis_mask = torch.tensor(f["modis_mask"][idx], dtype=torch.float32)  # e.g., first timestep, shape [12,12]
                tokens_modis=self.trans_config.apply_transformations_optique(modis,modis_dates,modis_mask,"modis")
                token_masks_wavelength.append(torch.ones(tokens_modis.shape[0]))
                tokens_list.append(tokens_modis)

            if self.config["dataset"]["S1"]:
                s1 = torch.tensor(f["s1"][idx], dtype=torch.float32)        # [20,4,4,6]
                s1_dates = f["s1_timestamps"][idx]  # Just example — you can build years/months/days here if needed
                s1_mask = torch.tensor(f["s1_mask"][idx], dtype=torch.float32)  # e.g., first timestep, shape [12,12]
                tokens_s1=self.trans_config.apply_transformations_SAR(s1,s1_dates,s1_mask,"s1")
                token_masks_wavelength.append(torch.ones(tokens_s1.shape[0])*2)
                tokens_list.append(tokens_s1)

            if self.config["dataset"]["ALOS"]:
                alos = torch.tensor(f["alos"][idx], dtype=torch.float32)        # [20,4,4,6]
                alos_dates = f["alos_timestamps"][idx]  # Just example — you can build years/months/days here if needed
                alos_mask = torch.tensor(f["alos_mask"][idx], dtype=torch.float32)  # e.g., first timestep, shape [12,12]
                tokens_alos=self.trans_config.apply_transformations_SAR(alos,alos_dates,alos_mask,"alos")
                token_masks_wavelength.append(torch.ones(tokens_alos.shape[0])*3)
                tokens_list.append(tokens_alos)


            tokens=torch.cat(tokens_list,dim=0)
            token_masks_w=torch.cat(token_masks_wavelength,dim=0)

            
            if tokens.shape[0]<self.max_tokens:
                tokens_padding=torch.zeros((self.max_tokens-tokens.shape[0],tokens.shape[1]))
                tokens_masks=torch.zeros((self.max_tokens-tokens.shape[0]))

            

                tokens=torch.cat([tokens,tokens_padding],dim=0)
                token_masks_w=torch.cat([token_masks_w,tokens_masks])

  

        return tokens,token_masks_w,label,frequency
    

class CustomPlanted(Dataset):
    def __init__(self, file_path, config=None, trans_config=None, augment=True):
        self.file_path = file_path
        self.config = config
        self.trans_config = trans_config
        self.augment = augment
        self.max_tokens=self.config["trainer"]["max_tokens"]
        self.h5 = None

        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = f["s1"].shape[0]  # all modalities have the same first dimension

    def __len__(self):
        return self.num_samples
    

    def get_modality(self,f,modality,idx):
        imgs=torch.tensor(f[modality][idx], dtype=torch.float32)
        time_stamps=f[f"{modality}_timestamps"][idx]
    
        time_stamps=self.trans_config.time_processing(time_stamps)
   
        mask=torch.tensor(f[f"{modality}_mask"][idx], dtype=torch.float32)

        return (imgs,time_stamps,mask)


    def __getitem__(self, idx):
        f = self.h5
            # Load modalities (you can combine or subset these as needed)
            #s1 = torch.tensor(f["s1"][idx], dtype=torch.float32)        # [8,12,12,3]


        label= f["label"][idx]
        frequency=f["frequencies"][idx]

        tokens_list=[]
        token_masks_wavelength=[]

        
        
        img_s2,date_s2,mask_s2=self.get_modality(f,"s2",idx)
        img_l7,date_l7,mask_l7=self.get_modality(f,"l7",idx)
        img_mo,date_mo,mask_mo=self.get_modality(f,"modis",idx)
        img_s1,date_s1,mask_s1=self.get_modality(f,"s1",idx)
        img_al,date_al,mask_al=self.get_modality(f,"alos",idx)


        arr = self.h5["s2"][idx]    # or however you read your first modality


        # 2) tensor conversion
        x = torch.from_numpy(arr)


                    


         

  

        return img_s2,img_l7,img_mo,img_s1,img_al,date_s2,date_l7,date_mo,date_s1,date_al,mask_s2,mask_l7,mask_mo,mask_s1,mask_al,label,frequency




import torch.distributed as dist

class CustomPlantedDataModule(pl.LightningDataModule):
    def __init__(self, path,config,trans_config, batch_size=8, num_workers=16):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_validation.h5"
        self.test_file = path + "_test.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_config = trans_config
        self.config=config

    def setup(self,stage=None):
        # Define transformations for the training phase.
        #self.train_transform = T.Compose([
        #    T.RandomHorizontalFlip(),
        #    T.RandomVerticalFlip(),
        #])

        # Initialize the datasets with their transformations.
        # Note: This setup() method is called on each process so the dataset and sampler
        # will be created in the proper distributed context.

        self.train_dataset = CustomPlanted(
            self.train_file,
            config=self.config,
            trans_config=self.trans_config,
        )

        self.val_dataset = CustomPlanted(
            self.val_file,
            config=self.config,
            trans_config=self.trans_config,
        )
   
        self.test_dataset = CustomPlanted(
            self.test_file,
            config=self.config,
            trans_config=self.trans_config,
        )


    def train_dataloader(self):
        # Create the custom distributed sampler inside the DataLoader call.

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,  # Enable pinned memory
            worker_init_fn=_init_worker,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,  # Enable pinned memory
            worker_init_fn=_init_worker,
            prefetch_factor=2,
        )

    def test_dataloader(self):

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,  # Enable pinned memory
            worker_init_fn=_init_worker,
            prefetch_factor=2,
        )