from training.perceiver import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.loggers import WandbLogger

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce

import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages
seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random


import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Evaluation script")




# Add the --run_id argument
parser.add_argument("--run_id", type=str, required=True, help="WandB run id from training")

# Add the --run_id argument
parser.add_argument("--xp_name", type=str, required=True, help="Experiment name")

# Add the --run_id argument
parser.add_argument("--config_model", type=str, required=True, help="Model config yaml file")

# Add the --run_id argument
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset used")


# Parse the arguments
args = parser.parse_args()

# Access the run id
run_id = args.run_id
xp_name=args.xp_name
config_model = args.config_model
config_name_dataset = args.dataset_name

print("Using WandB Run ID:", run_id)



seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

config_model = read_yaml("./training/configs/config_test-Atomiser_Atos.yaml")
labels = load_json_to_dict("./data/labels.json")
bands_yaml = "./data/bands_info/bands.yaml"

trans_config = transformations_config_flair(bands_yaml, config_model)


 
wand = True
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            id=run_id,            # Pass the run ID from the training run
            resume='allow',       # Allow resuming the existing run
            name=config_model['encoder'],
            project="PLANTED",
            config=config_model
        )
        wandb_logger = WandbLogger(project="PLANTED")


checkpoint_dir = "./checkpoints"
all_ckpt_files = [
    os.path.join(checkpoint_dir, f) 
    for f in os.listdir(checkpoint_dir) 
    if f.endswith(".ckpt")
]

# Try to filter files by xp_name if any checkpoint includes it.
checkpoint_files = [f for f in all_ckpt_files if xp_name in os.path.basename(f)]

# If no file contains xp_name, fallback to using all checkpoint files.
if not checkpoint_files:
    checkpoint_files = all_ckpt_files

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}.")

# Sort the selected checkpoint files by modification time and choose the most recent one.
checkpoint_path = sorted(checkpoint_files, key=os.path.getmtime)[-1]
print(f"Loading best checkpoint: {checkpoint_path}")

model = Model.load_from_checkpoint(checkpoint_path, config=config_model, wand=wand, name=xp_name, labels=labels,transform=trans_config)


data_module = CustomPlantedDataModule(
    "./data/custom_planted/"+config_name_dataset,
    config=config_model,
    trans_config=trans_config,
    batch_size=config_model['dataset']['batchsize'],
)




# Trainer
test_trainer = Trainer(
    use_distributed_sampler=False,
    max_epochs=config_model["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=1,
    accelerator="gpu",
    default_root_dir="./checkpoints/"
)

# Evaluate the model
test_trainer.test(model, datamodule=data_module)


