import torch
from math import pi
import einops as einops
from training.utils.FLAIR_2 import*
import matplotlib.pyplot as plt
from training.perceiver import*
from training.utils import*
from training.losses import*
from training.VIT import*
from training.ResNet import*
from collections import defaultdict
from training import*


from pytorch_lightning import Trainer,seed_everything
seed_everything(42, workers=True)

#config_path = "./data/flair_2_dataset/flair-2-config.yml" # Change to yours
config_path = "./data/flair_2_toy_dataset/flair-2-config.yml" # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
# Creation of the train, val and test dictionnaries with the data file paths
d_train, d_val, d_test = load_data(config)

images = d_train["PATH_IMG"]
labels = d_train["PATH_LABELS"]
sentinel_images = d_train["PATH_SP_DATA"]
sentinel_masks = d_train["PATH_SP_MASKS"] # Cloud masks
sentinel_products = d_train["PATH_SP_DATES"] # Needed to get the dates of the sentinel images
centroids = d_train["SP_COORDS"] # Position of the aerial image in the sentinel super area
aerial_mtds=d_train["MTD_AERIAL"]

create_dataset_flair(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtds, name="tiny", mode="train", stats=None)


from training.perceiver import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
from sklearn.metrics import average_precision_score
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce

# ← new imports for the PyTorch profiler
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity

import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import argparse



xp_name = "bouhouhou_SAD"
config_model = read_yaml("./training/configs/config_test-Atomiser_Atos_One.yaml")
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"
lookup_table=Lookup_encoding(read_yaml(configs_dataset),read_yaml(bands_yaml))
modalities_trans = modalities_transformations_config(configs_dataset,model=config_model["encoder"], name_config="regular")
test_conf=None
if config_model["encoder"] == "Atomiser_tradi":
    test_conf        = transformations_config_tradi(bands_yaml, config_model,lookup_table=lookup_table)
else:
    test_conf        = transformations_config(bands_yaml, config_model,lookup_table=lookup_table)

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"],
        project="MAE_debug",
        config=config_model
    )
    wandb_logger = WandbLogger(project="MAE_debug")
    

model = Model_MAE(
    config_model,
    wand=True,
    name=xp_name,
    transform=test_conf
)

data_module = Tiny_BigEarthNetDataModule(
    f"./data/custom_flair/tiny",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=FLAIR_MAE
)


# Call setup to actually create datasets
data_module.setup("fit")

# Access the training dataset directly
train_dataset = data_module.train_dataset

#print(type(train_dataset))
#print(train_dataset[0][0].shape)  # inspect one sample

