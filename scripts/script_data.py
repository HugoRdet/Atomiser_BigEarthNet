from training.perceiver import*
from training.utils import*
from training.losses import*
from training.VIT import*
from training.ResNet import*
from collections import defaultdict
from training import*

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from pytorch_lightning.profilers import AdvancedProfiler
import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages


from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule

torch.manual_seed(0)

mode="validation"

def prepare_BENv2(mode=None,max_len=None):
    dico_paths={"images_lmdb":"data/Encoded-BigEarthNet/",
    "metadata_parquet":"data/Encoded-BigEarthNet/metadata.parquet",
    "metadata_snow_cloud_parquet":"data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet"}

    df=open_parquet(dico_paths["metadata_parquet"])

    if max_len!=None:
        return BENv2_DataSet.BENv2DataSet(data_dirs=dico_paths, img_size=(14, 120, 120),split=mode,max_len=max_len),df

    
    return BENv2_DataSet.BENv2DataSet(data_dirs=dico_paths, img_size=(14, 120, 120),split=mode),df

def create_datasets(name,trans_conf,trans_tokens,sizes=(2,2,2),max_len=None,max_len_h5=-1):

   
    mode="train"
    ds,df=prepare_BENv2(max_len=max_len)

    #idxs,_=get_tiny_dataset(ds,df,MAX_IDs=sizes[0],mode=mode)
    idxs=None
    stats=create_dataset(idxs, ds,df, name=name, mode=mode, trans_config=trans_conf,trans_tokens=trans_tokens,stats=None,max_len=max_len_h5)

    mode="validation"
    ds,df=prepare_BENv2(max_len=max_len)
    
    
    #idxs,_=get_tiny_dataset(ds,df,MAX_IDs=sizes[1],mode=mode)
    idxs=None
    create_dataset(idxs, ds,df, name=name, mode=mode,trans_config=trans_conf,trans_tokens=trans_tokens,stats=stats,max_len=max_len_h5)

    mode="test"
    ds,_=prepare_BENv2(max_len=max_len)
    #idxs,_=get_tiny_dataset(ds,df,MAX_IDs=sizes[2],mode=mode)
    idxs=None
    create_dataset(idxs, ds,df, name=name, mode=mode,trans_config=trans_conf,trans_tokens=trans_tokens,stats=stats,max_len=max_len_h5)

bands_yaml="./data/bands_info/bands.yaml"
configs_dataset="./data/Tiny_BigEarthNet/configs_dataset_full.yaml"
config_dico = read_yaml("./training/configs/config_test-Atomiser_Atos.yaml")
#test_conf= transformations_config(config_dico,bands_yaml,configs_dataset,path_imgs_config="./data/Tiny_BigEarthNet/",name_config="BigEarthPart")
#(self,configs_dataset,path_imgs_config,name_config=""):
modalities_trans= modalities_transformations_config(configs_dataset,name_config="full")
test_conf= transformations_config(bands_yaml,config_dico)

create_datasets("full",modalities_trans,trans_tokens=test_conf,sizes=(100000,100000,100000),max_len_h5=-1)