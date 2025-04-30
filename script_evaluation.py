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


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    
    # Get the state dictionary
    state_dict = ckpt["state_dict"]
    
    # Create a new state dict that only contains keys that exist in the model
    model_state_dict = model.state_dict()
    
    # Filter out unexpected keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the filtered state dict
    missing_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # Print information about what was loaded and what was missed
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    
    print(f"Loaded {len(filtered_state_dict)} parameters")
    print(f"Missing {len(missing_keys.missing_keys)} parameters")
    print(f"Ignored {len(unexpected_keys)} unexpected parameters")
    
    if len(unexpected_keys) > 0:
        print("First few unexpected keys:", list(unexpected_keys)[:5])
    if len(missing_keys.missing_keys) > 0:
        print("First few missing keys:", missing_keys.missing_keys[:5])
    
    return model

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

config_model = read_yaml("./training/configs/"+config_model)
configs_dataset=f"./data/Tiny_BigEarthNet/configs_dataset_{config_name_dataset}.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

modalities_trans= modalities_transformations_config(configs_dataset,name_config=config_name_dataset)
test_conf= transformations_config(bands_yaml,config_model)


 
wand = True
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        run=wandb.init(
            #id=run_id,            # Pass the run ID from the training run
            #resume='allow',       # Allow resuming the existing run
            name=config_model['encoder'],
            project="Atomizer_BigEarthNet",
            config=config_model,
            tags=["evaluation", xp_name, config_model['encoder']]
        )
        #wandb_logger = WandbLogger(project="Atomizer_BigEarthNet")
        wandb_logger = WandbLogger(project="Atomizer_BigEarthNet", experiment=run)

# … everything up through your wandb / logger setup is unchanged …

checkpoint_dir = "./checkpoints"
all_ckpt_files = [
    os.path.join(checkpoint_dir, f)
    for f in os.listdir(checkpoint_dir)
    if f.endswith(".ckpt")
]
if not all_ckpt_files:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

def latest_ckpt_for(prefix: str):
    # filter to files that start with the prefix,
    # then pick the most recently‐modified one
    matches = [f for f in all_ckpt_files if os.path.basename(f).startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No checkpoints matching {prefix}* in {checkpoint_dir}")
    return max(matches, key=os.path.getmtime)

# 1) Best model according to val_mod_train AP
ckpt_train = latest_ckpt_for(config_model["encoder"]+"-best_model_val_mod_train")
print("→ Testing on ckpt (val_mod_train):", ckpt_train)

# 2) Best model according to val_mod_val AP
ckpt_val = latest_ckpt_for(config_model["encoder"]+"-best_model_val_mod_val")
print("→ Testing on ckpt (val_mod_val):", ckpt_val)


data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/{config_name_dataset}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    modality="test"
)

# One Trainer is enough; we'll just call .test twice
test_trainer = Trainer(
    accelerator="gpu",
    devices=[1],
    logger=wandb_logger,
    precision="16-mixed",
    #default_root_dir="./checkpoints/",
)

# Instantiate your model and datamodule just once
model = Model(config_model, wand=wand, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_train)
model = model.float()
model.comment_log="train_best mod_test "

# Test the “train‐best” checkpoint
test_results_train = test_trainer.test(
    model=model,
    datamodule=data_module,
    verbose=True,
    ckpt_path=None,
    
)




# Instantiate your model and datamodule just once
model = Model(config_model, wand=wand, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_val)
model = model.half()
model.comment_log="val_best mod_test "
# Test the “val‐best” checkpoint
# (Lightning will re-load the model from the new checkpoint)
test_results_val = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_val,
    verbose=True
)



print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)

# Instantiate your model and datamodule just once
model = Model(config_model, wand=wand, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_train)
model = model.half()
model.comment_log="val_best mod_test "

data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/{config_name_dataset}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    modality="validation"
)

#=====================
# One Trainer is enough; we'll just call .test twice
test_trainer = Trainer(
    accelerator="gpu",
    devices=[1],
    precision="16-mixed",
    logger=wandb_logger,
)

model.comment_log="train_best mod_val "
# Test the “train‐best” checkpoint
test_results_train = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_train,
    verbose=True
)




# Instantiate your model and datamodule just once
model = Model(config_model, wand=wand, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_val)
model = model.half()
model.comment_log="val_best mod_val "
# Test the “val‐best” checkpoint
# (Lightning will re-load the model from the new checkpoint)
test_results_val = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_val,
    verbose=True
)




print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)
