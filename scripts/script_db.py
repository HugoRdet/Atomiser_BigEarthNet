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

#import matplotlib.pyplot as plt

import cProfile, pstats, io
from pstats import SortKey

torch.manual_seed(0)

compute_channel_mean_std(mode="train")

create_dataset(name="tiny", mode="train",max_imgs=-1)
create_dataset(name="tiny", mode="test",max_imgs=-1)
create_dataset(name="tiny", mode="validation",max_imgs=-1)
