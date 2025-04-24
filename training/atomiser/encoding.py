import torch
from math import pi
import einops as einops

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands==-1:
        scales = torch.linspace(1., max_freq , max_freq, device = device, dtype = dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]

    x = x * scales * 2*pi
        
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

def pos_encoding(size,positional_scaling=None,max_freq=4,num_bands = 4):
        if positional_scaling==None:
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size), (size,size)))
        else:
            axis_pos = list(map(lambda size: torch.linspace(-positional_scaling/2.0, positional_scaling/2.0, steps=size), (size,size)))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
        
        pos_x=fourier_encode(pos[:,:,0],max_freq=max_freq,num_bands = num_bands)
        pos_y=fourier_encode(pos[:,:,1],max_freq=max_freq,num_bands = num_bands)
        
       
        pos_x=einops.rearrange(pos_x,"h w f -> (h w) f")
        pos_y=einops.rearrange(pos_y,"h w f -> (h w) f")

        pos=torch.cat([pos_x,pos_y],dim=1)
        return pos

def get_positional_processing(size,resolution,max_freq,num_bands):

        positional_scaling=None
        if resolution!=None:
            positional_scaling=(size * resolution)/400.0

   
        

        encoding=pos_encoding(size,positional_scaling=positional_scaling,max_freq=max_freq,num_bands = num_bands)
        

      
        return encoding
