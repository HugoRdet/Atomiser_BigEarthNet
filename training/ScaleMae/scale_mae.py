from torchgeo.models import ScaleMAE
import pytorch_lightning as pl
from torch import nn

class CustomScaleMAE(pl.LightningModule):
    def __init__(self,
                 *,
                 num_classes=19):
        super().__init__()

        

        self.encoder=ScaleMAE(
                res=20,
                img_size=120,
                patch_size=15,
                in_chans=12
            )
        self.to_logits = nn.Sequential(
                nn.Linear(768,num_classes)
            )
        
    def forward(self,x):
        x=self.encoder(x)
        x=self.to_logits(x)

        return x