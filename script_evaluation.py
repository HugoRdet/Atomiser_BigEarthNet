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
        wandb.init(
            #id=run_id,            # Pass the run ID from the training run
            resume='allow',       # Allow resuming the existing run
            name=config_model['encoder'],
            project="Atomizer_BigEarthNet",
            config=config_model
        )
        wandb_logger = WandbLogger(project="Atomizer_BigEarthNet")

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

# Instantiate your model and datamodule just once
model = Model(config_model, wand=wand, name=xp_name, transform=test_conf)
ckpt = torch.load(ckpt_train, map_location="cuda")
model.load_state_dict(ckpt["state_dict"], strict=True)
model = model.half()

data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/{config_name_dataset}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
)

# One Trainer is enough; we'll just call .test twice
test_trainer = Trainer(
    use_distributed_sampler=False,
    accelerator="gpu",
    devices=-1,
    logger=wandb_logger,
    default_root_dir="./checkpoints/",
)


# Test the “train‐best” checkpoint
test_results_train = test_trainer.test(
    model=model,
    datamodule=data_module,
    verbose=True,
    ckpt_path=None,
)

# 1) Record which checkpoint you just tested
wandb_logger.experiment.summary["train_best_ckpt"] = os.path.basename(ckpt_train)
# 2) Lift all of its test metrics into the summary with a prefix
for metric_name, val in test_results_train[0].items():
    wandb_logger.experiment.summary[f"mod_test_train_best_{metric_name}"] = val



# Test the “val‐best” checkpoint
# (Lightning will re-load the model from the new checkpoint)
test_results_val = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_val,
    verbose=True
)

# 1) Record which ckpt
wandb_logger.experiment.summary["val_best_ckpt"] = os.path.basename(ckpt_val)
# 2) Push its test metrics
for metric_name, val in test_results_val[0].items():
    wandb_logger.experiment.summary[f"mod_test_val_best_{metric_name}"] = val

print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)

#=====================
# One Trainer is enough; we'll just call .test twice
test_trainer = Trainer(
    use_distributed_sampler=False,
    accelerator="gpu",
    devices=[0],
    precision="16-mixed",
    logger=wandb_logger,
)

data_module.test_dataset.set_modality_mode("validation")

# Test the “train‐best” checkpoint
test_results_train = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_train,
    verbose=True
)

# 1) Record which checkpoint you just tested
wandb_logger.experiment.summary["train_best_ckpt"] = os.path.basename(ckpt_train)
# 2) Lift all of its test metrics into the summary with a prefix
for metric_name, val in test_results_train[0].items():
    wandb_logger.experiment.summary[f"mod_val_train_best_{metric_name}"] = val



# Test the “val‐best” checkpoint
# (Lightning will re-load the model from the new checkpoint)
test_results_val = test_trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path=ckpt_val,
    verbose=True
)

# 1) Record which ckpt
wandb_logger.experiment.summary["val_best_ckpt"] = os.path.basename(ckpt_val)
# 2) Push its test metrics
for metric_name, val in test_results_val[0].items():
    wandb_logger.experiment.summary[f"mod_val_val_best_{metric_name}"] = val

print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)
