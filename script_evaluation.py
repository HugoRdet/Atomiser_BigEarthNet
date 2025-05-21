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

# Add option to disable wandb
parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

# Parse the arguments
args = parser.parse_args()

# Access the run id
run_id = None#args.run_id
xp_name=args.xp_name
config_model = args.config_model
config_name_dataset = args.dataset_name
use_wandb = not args.no_wandb

print("Using WandB Run ID:", run_id)

# Helper function to handle loading of checkpoints with mismatched architectures
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

def setup_wandb(config_model, xp_name, run_id=None):
    """Set up W&B logging with error handling and reconnection logic"""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            # Set environment variables that might help with connection issues
            os.environ["WANDB_CONSOLE"] = "off"  # Disable console logging to reduce pipe traffic
            os.environ["WANDB_RECONNECT_ATTEMPTS"] = "5"  # Attempt to reconnect up to 5 times
            
            import wandb
            # Initialize wandb with more robust settings
            run = wandb.init(
                id=run_id if run_id else None,
                resume="allow" if run_id else None,
                name=config_model['encoder'],
                project="Atomizer_BigEarthNet",
                config=config_model,
                tags=["evaluation", xp_name, config_model['encoder']],
                settings=wandb.Settings(
                    _service_wait=300,  # Wait up to 5 minutes for service to respond
                    _file_stream_buffer=8192,  # Increase buffer size
                )
            )
            
            # Create logger with the run
            logger = WandbLogger(project="Atomizer_BigEarthNet", experiment=run)
            
            print("W&B logging successfully initialized")
            return logger
            
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Continuing without wandb logging")
            return None
    return None

def run_with_wandb_fallback(callback, *args, **kwargs):
    """Run a function with W&B logging, but gracefully fallback if W&B fails"""
    try:
        return callback(*args, **kwargs)
    except Exception as e:
        if "BrokenPipeError" in str(e) or "wandb" in str(e).lower():
            print(f"W&B error occurred: {e}")
            print("Disabling W&B and continuing with local logging only...")
            
            # Cleanup wandb
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except:
                pass
            
            # Set a global flag to disable wandb
            global use_wandb
            use_wandb = False
            
            # Re-run the callback with wandb disabled
            kwargs['logger'] = None
            return callback(*args, **kwargs)
        else:
            # If it's not a wandb error, re-raise it
            raise

def run_test(trainer, model, datamodule, ckpt_path=None):
    """Run the test with wandb fallback"""
    return run_with_wandb_fallback(
        trainer.test,
        model=model,
        datamodule=datamodule,
        verbose=True,
        ckpt_path=ckpt_path
    )

seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

config_model = read_yaml("./training/configs/"+config_model)
configs_dataset=f"./data/Tiny_BigEarthNet/configs_dataset_{config_name_dataset}.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

modalities_trans= modalities_transformations_config(configs_dataset,name_config=config_name_dataset)
test_conf= transformations_config(bands_yaml,config_model)

 
# Initialize W&B if enabled
wandb_logger = None
if use_wandb:
    wandb_logger = setup_wandb(config_model, xp_name, run_id)

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
#prefix
prefix=config_model["encoder"]
if prefix=="Atomiser":
    prefix="Atomiserxp_20250516_234544"
ckpt_train = latest_ckpt_for(config_model["encoder"])
print("→ Testing on ckpt (val_mod_train):", ckpt_train)

# 2) Best model according to val_mod_val AP
#ckpt_val = latest_ckpt_for(config_model["encoder"]+"-best_model_val_mod_val")
#print("→ Testing on ckpt (val_mod_val):", ckpt_val)

# Set up data module for testing
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
)

# Test with the "train‐best" checkpoint
print("\n===== Testing model from train-best checkpoint on test data =====")
model = Model(config_model, wand=use_wandb, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_train)
model = model.float()
model.comment_log="train_best mod_test "

test_results_train = run_test(
    test_trainer, 
    model, 
    data_module
)

# Test with the "val‐best" checkpoint
#print("\n===== Testing model from val-best checkpoint on test data =====")
#model = Model(config_model, wand=use_wandb, name=xp_name, transform=test_conf)
#model = load_checkpoint(model, ckpt_val)
#model = model.float()
#model.comment_log="val_best mod_test "

test_results_val = run_test(
    test_trainer, 
    model, 
    data_module
)

#print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)

# Now test on validation data
print("\n===== Setting up validation data module =====")
data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/{config_name_dataset}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    modality="validation"
)

# Test with the "train‐best" checkpoint on validation data
print("\n===== Testing model from train-best checkpoint on validation data =====")
model = Model(config_model, wand=use_wandb, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_train)
model = model.float()
model.comment_log="train_best mod_val "

test_results_train_val = run_test(
    test_trainer, 
    model, 
    data_module
)

# Test with the "val‐best" checkpoint on validation data
#print("\n===== Testing model from val-best checkpoint on validation data =====")
#model = Model(config_model, wand=use_wandb, name=xp_name, transform=test_conf)
#model = load_checkpoint(model, ckpt_val)
#model = model.float()
#model.comment_log="val_best mod_val "

#test_results_val_val = run_test(
#    test_trainer, 
#    model, 
#    data_module
#)

# Clean up wandb
if use_wandb:
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

print("\n===== Final Results =====")
print("Test dataset results:")
print("Results for best_model_val_mod_train:", test_results_train)
print("Results for best_model_val_mod_val:  ", test_results_val)

