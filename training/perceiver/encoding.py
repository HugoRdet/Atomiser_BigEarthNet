import torch
from math import pi
import einops as einops

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]
    
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

def fourier_encode_xy(x, max_freq, num_bands = 4):
    _,height,width=x.shape
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(1., max_freq / 2, height, device=device, dtype=dtype),
        torch.linspace(1., max_freq / 2, width, device=device, dtype=dtype),
        indexing="ij",
    )

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]
    
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

def positional_encoding_2d_to_1d(image, max_freq, num_bands=4):
    """
    Generate positional encoding for a 2D image based on the X coordinate.
    Args:
        image (torch.Tensor): Input tensor of shape (channels, height, width).
        max_freq (float): Maximum frequency for the positional encoding.
        num_bands (int): Number of frequency bands for Fourier encoding.
    Returns:
        torch.Tensor: Tensor with positional encoding added, shape (channels + new_channels, height, width).
    """
    channels, height, width = image.shape
    device = image.device

    # Flatten the image to shape (height * width, channels)
    flattened_image = einops.rearrange(image, "c h w -> (h w) c")

    # Create X coordinate positions (indices for the flattened vector)
    x_coords = torch.arange(height * width, device=device, dtype=torch.float32)

    # Apply Fourier positional encoding to the X coordinates
    x_pos_enc = fourier_encode(x_coords, max_freq=max_freq, num_bands=num_bands)  # Shape: (height * width, new_channels)

    # Concatenate the positional encodings with the flattened image
    encoded_flat = torch.cat([flattened_image, x_pos_enc], dim=-1)  # Shape: (height * width, channels + new_channels)

    # Reshape back to (channels + new_channels, height, width)
    new_channels = encoded_flat.shape[-1]
    encoded_image = einops.rearrange(encoded_flat, "(h w) c -> c h w", h=height, w=width)

    return encoded_image

def positional_encoding_2d_to_1d_batch(images, max_freq, num_bands=4):
    """
    Generate positional encoding for a batch of 2D images based on the X coordinate.
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        max_freq (float): Maximum frequency for the positional encoding.
        num_bands (int): Number of frequency bands for Fourier encoding.
    Returns:
        torch.Tensor: Tensor with positional encoding added, shape (batch_size, channels + new_channels, height, width).
    """
    batch_size, channels, height, width = images.shape
    device = images.device

    # Flatten the image to shape (batch_size, height * width, channels)
    flattened_images = einops.rearrange(images, "b c h w -> b (h w) c")

    # Create X coordinate positions (indices for the flattened vector)
    x_coords = torch.arange(height * width, device=device, dtype=torch.float32)

    # Apply Fourier positional encoding to the X coordinates
    x_pos_enc = fourier_encode(x_coords, max_freq=max_freq, num_bands=num_bands)  # Shape: (height * width, new_channels)

    # Repeat positional encodings for each image in the batch
    x_pos_enc_batch = x_pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, height * width, new_channels)

    # Concatenate the positional encodings with the flattened images
    encoded_flat_batch = torch.cat([flattened_images, x_pos_enc_batch], dim=-1)  # Shape: (batch_size, height * width, channels + new_channels)

    # Reshape back to (batch_size, channels + new_channels, height, width)
    new_channels = encoded_flat_batch.shape[-1]
    encoded_images = einops.rearrange(encoded_flat_batch, "b (h w) c -> b c h w", h=height, w=width)

    return encoded_images

def fourier_encode_2D_batch(x, max_freq, num_bands=4):
    """
    Apply Fourier positional encoding to a batch of 2D images.
    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        max_freq (float): Maximum frequency for the positional encoding.
        num_bands (int): Number of frequency bands.
    Returns:
        torch.Tensor: Tensor with positional encoding added, shape 
                      (batch, channels + new_channels, height, width).
    """
    b, c, h, w = x.shape
    device, dtype = x.device, x.dtype

    # Generate grid of normalized coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij"
    )  # Shape: (h, w)
    
    # Stack the coordinates and apply Fourier encoding
    pos_coords = torch.stack([x_coords, y_coords], dim=0)  # Shape: (2, h, w)
    pos_coords = pos_coords.unsqueeze(0).expand(b, -1, -1, -1)  # Shape: (b, 2, h, w)

    scales = torch.linspace(1.0, max_freq, num_bands, device=device, dtype=dtype)  # (num_bands)
    scales = scales.view(-1, 1, 1)  # Shape: (num_bands, 1, 1)

    pos_coords = pos_coords.unsqueeze(2) * scales * pi  # Shape: (b, 2, num_bands, h, w)
    encoded_coords = torch.cat([pos_coords.sin(), pos_coords.cos()], dim=2)  # (b, 2, 2*num_bands, h, w)

    # Reshape to (b, new_channels, h, w)
    encoded_coords = encoded_coords.view(b, -1, h, w)  # (b, 2 * num_bands * 2, h, w)

    # Concatenate with input along the channel dimension
    return torch.cat([x, encoded_coords], dim=1)  # Shape: (b, c + new_channels, h, w)










def fourier_encode_2D(x, max_freq, num_bands=4):
    """
    Apply Fourier positional encoding to both X and Y coordinates of a 2D tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (channels, height, width).
        max_freq (float): Maximum frequency for the encoding.
        num_bands (int): Number of frequency bands.
    
    Returns:
        torch.Tensor: Tensor with positional encoding added, shape 
                      (channels + new_channels, height, width).
    """
    channels, height, width = x.shape
    device, dtype = x.device, x.dtype

    # Create positional coordinates for X and Y
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij"
    )  # Shape: [height, width]

    # Stack X and Y coordinates along a new dimension
    position_tensor = torch.stack([x_coords, y_coords], dim=0)  # Shape: [2, height, width]

    # Create frequency scales for Fourier encoding
    scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)  # Shape: [num_bands]
    scales = scales.view(-1, 1, 1)  # Shape: [num_bands, 1, 1] for broadcasting

    # Apply scales and Fourier transformations to positional tensor
    position_tensor = position_tensor.unsqueeze(1) * scales * pi  # Shape: [2, num_bands, height, width]
    sin_enc = position_tensor.sin()  # Shape: [2, num_bands, height, width]
    cos_enc = position_tensor.cos()  # Shape: [2, num_bands, height, width]

    # Combine sine and cosine encodings
    pos_enc = torch.cat([sin_enc, cos_enc], dim=1)  # Shape: [2, 2*num_bands, height, width]

    # Reshape to [new_channels, height, width]
    pos_enc = pos_enc.view(-1, height, width)  # Shape: [new_channels, height, width]

    # Concatenate positional encodings to the input tensor
    x = torch.cat([x, pos_enc], dim=0)  # Shape: [channels + new_channels, height, width]

    return x

