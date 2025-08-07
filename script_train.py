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

# instantiate the PyTorchProfiler
#profiler = PyTorchProfiler(
#    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#    record_shapes=True,
#    profile_memory=True,
#    export_to_chrome=True,       # dumps a trace.json for Chrome/TensorBoard
#    dirpath="profiling",         # where to write trace.json
#    filename="trace"             # will produce "profiling/trace.json"
#)

# Create the parser
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"
lookup_table=Lookup_positional_encoding(read_yaml(configs_dataset))
modalities_trans = modalities_transformations_config(configs_dataset,model=config_model["encoder"], name_config=args.dataset_name)
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
        project="Atomizer_BigEarthNet_debug",
        config=config_model
    )
    wandb_logger = WandbLogger(project="Atomizer_BigEarthNet_debug")

model = Model(
    config_model,
    wand=True,
    name=xp_name,
    transform=test_conf
)

data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/{args.dataset_name}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table
)


checkpoint_val_mod_train = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-best_val_mod_train-{{epoch:02d}}-{{val_mod_train_ap:.4f}}",
    monitor="val_mod_train_ap",
    mode="max",
    save_top_k=1,
    verbose=True,
)
accumulator = GradientAccumulationScheduler(scheduling={0:64})

# Trainer
trainer = Trainer(
    strategy="ddp",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[checkpoint_val_mod_train, accumulator],
    default_root_dir="./checkpoints/",
    #profiler=profiler,           # ← attach the PyTorchProfiler here
)

# Fit the model
trainer.fit(model, datamodule=data_module)

# Save wandb run ID if needed
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
