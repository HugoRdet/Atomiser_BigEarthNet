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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,GradientAccumulationScheduler
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
parser = argparse.ArgumentParser(description="Training script")

# Add the --run_id argument
parser.add_argument("--xp_name", type=str, required=True, help="Experiment name")

# Add the --run_id argument
parser.add_argument("--config_model", type=str, required=True, help="Model config yaml file")

# Add the --run_id argument
parser.add_argument("--dataset_name", type=str, required=True, help="name of the dataset used")

# Parse the arguments
args = parser.parse_args()

# Access the run id
xp_name=args.xp_name
config_model = args.config_model
config_name_dataset = args.dataset_name

seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

config_model = read_yaml("./training/configs/"+config_model)
labels = load_json_to_dict("./data/labels.json")
bands_yaml = "./data/bands_info/bands.yaml"

trans_config = transformations_config_flair(bands_yaml, config_model)

wand = True
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name=config_model['encoder'],
            project="PLANTED",
            config=config_model
        )
        wandb_logger = WandbLogger(project="PLANTED")

model = Model(config_model, wand=wand, name=xp_name, labels=labels,transform=trans_config)

data_module = CustomPlantedDataModule(
    "./data/custom_planted/"+config_name_dataset,
    config=config_model,
    trans_config=trans_config,
    batch_size=config_model['dataset']['batchsize'],
)


# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename="best_model-{epoch:02d}-{val_F1:.4f}",
    monitor="val_F1",
    mode="max",
    save_top_k=1,
    verbose=True,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=75,
    verbose=False,
    mode="max"
)


accumulator = GradientAccumulationScheduler(scheduling={0: 20})

# Trainer
trainer = Trainer(
    use_distributed_sampler=False,
    strategy="ddp",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=256,
    accelerator="gpu",
    callbacks=[early_stop_callback, checkpoint_callback,accumulator],
    default_root_dir="./checkpoints/",
    val_check_interval=0.3,
    precision="bf16-mixed"
)


# Fit the model
trainer.fit(model, datamodule=data_module)






# ... after training completes, within your "if wand" block:
if wand and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    
    # Create the directory for storing wandb run IDs if it doesn't exist
    runs_dir = "training/wandb_runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    
    # Save the run ID to a file inside wandb_runs
    run_file = os.path.join(runs_dir, xp_name+".txt")
    with open(run_file, "w") as f:
        f.write(run_id)






