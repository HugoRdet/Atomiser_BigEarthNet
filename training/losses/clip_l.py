import torch
import torch.nn.functional as F
import torch.nn as nn

class CLIP_loss(nn.Module):

    def __init__(self,latent_dim,joint_dim,p=0.5):
        super().__init__()

        #sentinel 1 joint space
        self.W_S1_1 = nn.Linear(latent_dim, joint_dim, bias=False)
        self.W_S1_2 = nn.Linear(joint_dim, joint_dim, bias=False)
        self.layer_norm_S1 = nn.LayerNorm(joint_dim) #layernorm has learnable parameters

        #sentinel 2 joint space
        self.W_S2_1 = nn.Linear(latent_dim, joint_dim, bias=False)
        self.W_S2_2 = nn.Linear(joint_dim, joint_dim, bias=False)
        self.layer_norm_S2 = nn.LayerNorm(joint_dim)

        self.drop = nn.Dropout(p)


        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07))  # Default temperature 0.07

    def projection_joint(self,x,device,id="S1"):
        if id=="S1":
            embed=self.W_S1_1(x)
            embed=self.W_S1_2(embed)
            embed=self.layer_norm_S1(embed)
            return self.drop(embed)
        else:
            embed=self.W_S2_1(x)
            embed=self.W_S2_2(embed)
            embed=self.layer_norm_S2(embed)
            return self.drop(embed)

    def forward(self, latent_S1, latent_S2):
        """ Compute the CLIP loss. """
        # Normalize embeddings
        z_S1 = F.normalize(self.projection_joint(latent_S1, id="S1"), dim=-1)
        z_S2 = F.normalize(self.projection_joint(latent_S2, id="S2"), dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(z_S1, z_S2.T) * self.logit_scale.exp()

        # Create labels for contrastive learning
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Cross-entropy loss
        loss_S1 = F.cross_entropy(logits, labels)
        loss_S2 = F.cross_entropy(logits.T, labels)

        # Return average loss
        return (loss_S1 + loss_S2) / 2
        
        
