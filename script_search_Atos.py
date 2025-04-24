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

import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages


from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import os
import subprocess



def run_sweep(config_path):
    def train_fn():
        torch.manual_seed(0)

        config = read_yaml(config_path)
        tmp_name=tmp_name=get_xp_name(config["encoder"])

        wandb.init(
                name=tmp_name,
                project="trial_pos_encodings",
                config=config
            )
        
        # Update config from wandb sweep
        tmp_encoder=config["encoder"]
        sweep_config = wandb.config
       
        config[tmp_encoder]["num_latents"] = sweep_config.num_latents
        config[tmp_encoder]["latent_dim"] = sweep_config.latent_dim
        #config[tmp_encoder]["self_per_cross_attn"] = sweep_config.self_per_cross_attn

        save_yaml("./training/configs/tmp_sweep_config.yaml",config)

        subprocess.call("sh TrainEval.sh test_sweep tmp_sweep_config.yaml regular", shell=True)


    # Sweep configuration
    sweep_config = {
        "method": "grid",  # Choose "random", "grid", or "bayes"
        "parameters": {
            "query_num_freq_bands:": {"values": [64,512]},
            "query_max_freq": {"values": [64,512]},
            #"self_per_cross_attn": {"values": [1,4,8]},
            #"depth":{"values": [4,8]},
        }
    }

    # Initialize WandB sweep
    sweep_id = wandb.sweep(sweep_config, project="trial_pos_encodings")

    # Run sweep
    wandb.agent(sweep_id, function=train_fn)

config_path = "./training/configs/config_test-Atomiser_Atos.yaml"
run_sweep(config_path)

