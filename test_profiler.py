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
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.callbacks import EarlyStopping

seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')


torch.manual_seed(0)

bands_yaml="./data/bands_info/bands.yaml"
configs_dataset="./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
config_dico = read_yaml("./training/configs/config_test-Atomiser_Atos.yaml")
#test_conf= transformations_config(config_dico,bands_yaml,configs_dataset,path_imgs_config="./data/Tiny_BigEarthNet/",name_config="BigEarthPart")
#(self,configs_dataset,path_imgs_config,name_config=""):
modalities_trans= modalities_transformations_config(configs_dataset,name_config="regular")
test_conf= transformations_config(bands_yaml,config_dico)

#create_datasets("regular",modalities_trans,sizes=(5,5,5),max_len_h5=-1)



data_module=Tiny_BigEarthNetDataModule( "./data/Tiny_BigEarthNet/regular", 
                                       batch_size=16, 
                                       num_workers=4,
                                       trans_modalities=modalities_trans,
                                       trans_tokens=None,
                                       model="Atomiser")

data_module.setup()
# Prepare dataloaders
train_loader = data_module.train_dataloader()
#val_loader = data_module.val_dataloader()
#test_loader = data_module.test_dataloader()


xp_name="test_xp"
config_model = "Atomiser_Atos"
config_name_dataset = "tiny"
config_name_dataset= "./data/custom_flair/"+config_name_dataset


config_model = read_yaml("./training/configs/config_test-"+config_model+".yaml")
#labels=load_json_to_dict("./data/flair_2_toy_dataset/flair_labels.json")
bands_yaml = "./data/Tiny_BigEarthNet/bands.yaml"

bands_yaml="./data/bands_info/bands.yaml"
configs_dataset="./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
config_dico = read_yaml("./training/configs/config_test-Atomiser_Atos.yaml")

# Prepare dataloaders

wand = False
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name=config_model['encoder'],
            project=config_name_dataset+"_modalities",
            config=config_model
        )
        wandb_logger = WandbLogger(project=config_name_dataset+"_modalities")

#def __init__(self, config, wand, name)
model = Model(config_model,wand=wand, name=xp_name,transform=test_conf)


early_stop_callback = EarlyStopping(monitor="val_ap", min_delta=0.00, patience=15, verbose=False, mode="max")

profiler = AdvancedProfiler(dirpath="profiling", filename="profiler_output.txt")

# Configure the trainer for distributed training.
trainer = Trainer(
    use_distributed_sampler=False,  # we use our custom sampler
    strategy="ddp",
    max_epochs=config_model["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=1,
    devices=-1,
    accelerator="gpu",
    callbacks=[early_stop_callback],
    profiler=profiler  
)

trainer.fit(model, datamodule=data_module)
 
