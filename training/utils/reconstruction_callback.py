
import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import seaborn as sns
from typing import Optional, Union, List
from .image_utils import*



class CustomMAEReconstructionCallback(pl.Callback):

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: Configuration dict containing visualization parameters
        """
        super().__init__()
        self.config=config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
        # Sentinel-2 band definitions (12 bands)
        self.sentinel2_bands = {
            'B01': 0,   # Coastal aerosol (443 nm)
            'B02': 1,   # Blue (490 nm)
            'B03': 2,   # Green (560 nm)
            'B04': 3,   # Red (665 nm)
            'B05': 4,   # Vegetation Red Edge (705 nm)
            'B06': 5,   # Vegetation Red Edge (740 nm)
            'B07': 6,   # Vegetation Red Edge (783 nm)
            'B08': 7,   # NIR (842 nm)
            'B8A': 8,   # Vegetation Red Edge (865 nm)
            'B09': 9,   # Water vapour (945 nm)
            'B11': 10,  # SWIR (1610 nm)
            'B12': 11,  # SWIR (2190 nm)
        }
        
        # Band combinations for visualization
        self.rgb_bands = [self.sentinel2_bands['B04'], self.sentinel2_bands['B03'], self.sentinel2_bands['B02']]  # Red, Green, Blue
        self.infrared_bands = [self.sentinel2_bands['B08'], self.sentinel2_bands['B11'], self.sentinel2_bands['B12']]  # NIR, SWIR1, SWIR2
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform reconstruction visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # **CRITICAL: Ensure we only run on GPU 0 (rank 0) to avoid multiple executions**
        if trainer.is_global_zero == False:
            return
            
        # Alternative way to check for single GPU execution
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            # Only execute on rank 0
            if wandb.run is not None:
                self._perform_custom_reconstruction(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping reconstruction logging")
        except Exception as e:
            print(f"Error in custom reconstruction callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_custom_reconstruction(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform reconstruction using your custom data loading."""
        
        # Get the dataset from the trainer
        dataset = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(sample_idx, i, dataset, pl_module, trainer.current_epoch)
            
    def _process_single_sample(self, dataset_idx: int, viz_idx: int, dataset, pl_module, epoch: int):
        """Process a single sample using your get_samples_to_viz function."""
        
        # Get data from your custom method
        image_tokens, attention_mask, mae_tokens, mask_MAE_res = dataset.get_samples_to_viz(dataset_idx)
        
        # Move to device
        mae_tokens_mask = torch.ones(mae_tokens.shape[0])
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mask_MAE_res = mask_MAE_res.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions to match batch format
                image_tokens_batch = image_tokens.unsqueeze(0) 
                image_tokens_mask = attention_mask.unsqueeze(0)
                mae_tokens_batch = mae_tokens.clone().unsqueeze(0)
                mae_tokens_mask_batch = mae_tokens_mask.unsqueeze(0)
                
                # Perform reconstruction
                y_hat, y_mask = pl_module.forward(
                    image_tokens_batch,
                    image_tokens_mask,
                    mae_tokens_batch,
                    mae_tokens_mask_batch,
                    training=False,
                    task="reconstruction"
                )
                
                # Remove batch dimension for visualization
                y_hat = y_hat.squeeze(0)
                
                # Reshape to spatial format
                ground_truth = mae_tokens[:,0]
                ground_truth = rearrange(ground_truth, "(b h w) -> b h w", b=12, h=120, w=120)
                
                y_hat = rearrange(y_hat, "(b h w) c -> b h w c", b=12, h=120, w=120, c=1).squeeze(-1)
                
                print(f"✓ Successfully reconstructed sample {dataset_idx}")
                
            except Exception as e:
                print(f"Error in model forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_spatial_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                ground_truth=ground_truth,      # [12, 120, 120]
                prediction=y_hat,               # [12, 120, 120]
                spatial_mask=mask_MAE_res,      # [120, 120]
                epoch=epoch
            )
        except Exception as e:
            print(f"Error creating spatial visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_and_upload_spatial_visualization(self, sample_idx, viz_idx, ground_truth, prediction, spatial_mask, epoch):
        """Create and upload spatial reconstruction visualization to wandb."""
        
        # Convert to numpy for visualization
        gt_np = ground_truth.cpu().numpy()          # [12, 120, 120]
        pred_np = prediction.cpu().numpy()          # [12, 120, 120]
        mask_np = spatial_mask.cpu().numpy()        # [120, 120]
        
        # Calculate basic metrics for logging
        mse = np.mean((gt_np - pred_np) ** 2)
        mae = np.mean(np.abs(gt_np - pred_np))
        correlation = np.corrcoef(gt_np.flatten(), pred_np.flatten())[0, 1]
        
        # Create the main spatial visualization
        spatial_fig = self._create_spatial_reconstruction_figure(
            gt_np, pred_np, mask_np, sample_idx, epoch, mse, mae, correlation
        )
        
        # Prepare wandb data
        wandb_data = {
            # Main spatial visualization
            f"spatial_reconstruction/epoch_{epoch}_sample_{sample_idx}": wandb.Image(
                spatial_fig, 
                caption=f"Spatial MAE Reconstruction - Epoch {epoch}, Sample {sample_idx}, MSE: {mse:.6f}"
            ),
            
            # Basic metrics
            f"reconstruction_metrics/epoch": epoch,
            f"reconstruction_metrics/sample_{viz_idx}_mse": mse,
            f"reconstruction_metrics/sample_{viz_idx}_mae": mae,
            f"reconstruction_metrics/sample_{viz_idx}_correlation": correlation,
        }
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Successfully uploaded spatial reconstruction for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
        
        plt.close(spatial_fig)
    
    def _create_spatial_reconstruction_figure(self, ground_truth, prediction, spatial_mask, 
                                            sample_idx, epoch, mse, mae, correlation):
        """Create clean spatial reconstruction visualization with RGB and Infrared bands."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MAE Reconstruction - Sample {sample_idx}, Epoch {epoch}\n'
                    f'MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}', 
                    fontsize=16, fontweight='bold')
        
        # Row 1: RGB bands (Red, Green, Blue)
        rgb_gt = self._create_rgb_composite(ground_truth, self.rgb_bands)
        rgb_pred = self._create_rgb_composite(prediction, self.rgb_bands)
        
        # Ground Truth RGB
        axes[0, 0].imshow(rgb_gt)
        axes[0, 0].set_title('Ground Truth\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction RGB
        axes[0, 1].imshow(rgb_pred)
        axes[0, 1].set_title('Prediction\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Spatial Mask
        axes[0, 2].imshow(spatial_mask, cmap='RdYlBu_r', alpha=0.9)
        axes[0, 2].set_title(f'Spatial Mask\n(Masked: {np.mean(spatial_mask):.1%})', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Infrared bands (NIR, SWIR1, SWIR2)
        infrared_gt = self._create_rgb_composite(ground_truth, self.infrared_bands)
        infrared_pred = self._create_rgb_composite(prediction, self.infrared_bands)
        
        # Ground Truth Infrared
        axes[1, 0].imshow(infrared_gt)
        axes[1, 0].set_title('Ground Truth\n(False Color: B08-B11-B12)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Prediction Infrared
        axes[1, 1].imshow(infrared_pred)
        axes[1, 1].set_title('Prediction\n(False Color: B08-B11-B12)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Error Map (using RGB bands)
        rgb_error = np.abs(rgb_gt.astype(float) - rgb_pred.astype(float))
        rgb_error = rgb_error / rgb_error.max() if rgb_error.max() > 0 else rgb_error
        axes[1, 2].imshow(rgb_error)
        axes[1, 2].set_title('RGB Error Map\n(Absolute Difference)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_composite(self, data, band_indices):
        """Create RGB composite from selected bands."""
        # data: [12, 120, 120], band_indices: [R_idx, G_idx, B_idx]
        composite = np.stack([
            data[band_indices[0]],  # Red channel
            data[band_indices[1]],  # Green channel  
            data[band_indices[2]]   # Blue channel
        ], axis=-1)  # [120, 120, 3]
        
        # Normalize using the provided function
        composite_tensor = torch.from_numpy(composite).permute(2, 0, 1)  # [3, 120, 120]
        normalized = normalize(composite_tensor)  # [3, 120, 120]
        
        return normalized.permute(1, 2, 0).numpy()  # [120, 120, 3]
    
    





class CustomMAEReconstructionCallback_FLAIR(pl.Callback):

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: Configuration dict containing visualization parameters
        """
        super().__init__()
        self.config=config
        
        self.log_every_n_epochs = config["debug"]["viz_every_n_epochs"]
        self.sample_indices = config["debug"]["idxs_to_viz"]
        self.num_samples = len(self.sample_indices)
        
        # Sentinel-2 band definitions (12 bands)
        self.sentinel2_bands = {
            'B01': 0,   # Coastal aerosol (443 nm)
            'B02': 1,   # Blue (490 nm)
            'B03': 2,   # Green (560 nm)
            'B04': 3,   # Red (665 nm)
            'B05': 4,   # Vegetation Red Edge (705 nm)
            'B06': 5,   # Vegetation Red Edge (740 nm)
            'B07': 6,   # Vegetation Red Edge (783 nm)
            'B08': 7,   # NIR (842 nm)
            'B8A': 8,   # Vegetation Red Edge (865 nm)
            'B09': 9,   # Water vapour (945 nm)
            'B11': 10,  # SWIR (1610 nm)
            'B12': 11,  # SWIR (2190 nm)
        }
        
        # Band combinations for visualization
        self.rgb_bands = [2, 1, 0]  # Red, Green, Blue
        self.infrared_bands = [4,4,4]  # NIR, SWIR1, SWIR2
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Perform reconstruction visualization at the end of validation epoch."""
        
        # Only log every N epochs and skip first few epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or trainer.current_epoch < 2:
            return
        
        # **CRITICAL: Ensure we only run on GPU 0 (rank 0) to avoid multiple executions**
        if trainer.is_global_zero == False:
            return
            
        # Alternative way to check for single GPU execution
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        pl_module.eval()
        
        try:
            # Only execute on rank 0
            if wandb.run is not None:
                self._perform_custom_reconstruction(trainer, pl_module)
            else:
                print("Warning: wandb not active, skipping reconstruction logging")
        except Exception as e:
            print(f"Error in custom reconstruction callback: {e}")
            import traceback
            traceback.print_exc()
        
        pl_module.train()
    
    def _perform_custom_reconstruction(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform reconstruction using your custom data loading."""
        
        # Get the dataset from the trainer
        dataset = trainer.datamodule.val_dataset
        
        for i, sample_idx in enumerate(self.sample_indices):
            self._process_single_sample(sample_idx, i, dataset, pl_module, trainer.current_epoch)
            
    def _process_single_sample(self, dataset_idx: int, viz_idx: int, dataset, pl_module, epoch: int):
        """Process a single sample using your get_samples_to_viz function."""
        
        # Get data from your custom method
        image_tokens, attention_mask, mae_tokens, mask_MAE_res = dataset.get_samples_to_viz(dataset_idx)
        
        # Move to device
        mae_tokens_mask = torch.ones(mae_tokens.shape[0])
        device = pl_module.device
        image_tokens = image_tokens.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        mae_tokens = mae_tokens.to(device)
        mask_MAE_res = mask_MAE_res.to(device)
        
        with torch.no_grad():
            try:
                # Expand dimensions to match batch format
                image_tokens_batch = image_tokens.unsqueeze(0) 
                image_tokens_mask = attention_mask.unsqueeze(0)
                mae_tokens_batch = mae_tokens.clone().unsqueeze(0)
                mae_tokens_mask_batch = mae_tokens_mask.unsqueeze(0)
                
                # Perform reconstruction
                y_hat, y_mask = pl_module.forward(
                    image_tokens_batch,
                    image_tokens_mask,
                    mae_tokens_batch,
                    mae_tokens_mask_batch,
                    training=False,
                    task="reconstruction"
                )
                
                # Remove batch dimension for visualization
                y_hat = y_hat.squeeze(0)
                
                # Reshape to spatial format
                ground_truth = mae_tokens[:,0]
                ground_truth = rearrange(ground_truth, "(b h w) -> b h w", b=5, h=512, w=512)
                
                y_hat = rearrange(y_hat, "(b h w) c -> b h w c", b=5, h=512, w=512, c=1).squeeze(-1)
                
                print(f"✓ Successfully reconstructed sample {dataset_idx}")
                
            except Exception as e:
                print(f"Error in model forward pass for sample {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Create and upload visualization
        try:
            self._create_and_upload_spatial_visualization(
                sample_idx=dataset_idx,
                viz_idx=viz_idx,
                ground_truth=ground_truth,      # [12, 120, 120]
                prediction=y_hat,               # [12, 120, 120]
                spatial_mask=mask_MAE_res,      # [120, 120]
                epoch=epoch
            )
        except Exception as e:
            print(f"Error creating spatial visualization for sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_and_upload_spatial_visualization(self, sample_idx, viz_idx, ground_truth, prediction, spatial_mask, epoch):
        """Create and upload spatial reconstruction visualization to wandb."""
        
        # Convert to numpy for visualization
        gt_np = ground_truth.cpu().numpy()          # [12, 120, 120]
        pred_np = prediction.cpu().numpy()          # [12, 120, 120]
        mask_np = spatial_mask.cpu().numpy()        # [120, 120]
        
        # Calculate basic metrics for logging
        mse = np.mean((gt_np - pred_np) ** 2)
        mae = np.mean(np.abs(gt_np - pred_np))
        correlation = np.corrcoef(gt_np.flatten(), pred_np.flatten())[0, 1]
        
        # Create the main spatial visualization
        spatial_fig = self._create_spatial_reconstruction_figure(
            gt_np, pred_np, mask_np, sample_idx, epoch, mse, mae, correlation
        )
        
        # Prepare wandb data
        wandb_data = {
            # Main spatial visualization
            f"spatial_reconstruction/epoch_{epoch}_sample_{sample_idx}": wandb.Image(
                spatial_fig, 
                caption=f"Spatial MAE Reconstruction - Epoch {epoch}, Sample {sample_idx}, MSE: {mse:.6f}"
            ),
            
            # Basic metrics
            f"reconstruction_metrics/epoch": epoch,
            f"reconstruction_metrics/sample_{viz_idx}_mse": mse,
            f"reconstruction_metrics/sample_{viz_idx}_mae": mae,
            f"reconstruction_metrics/sample_{viz_idx}_correlation": correlation,
        }
        
        # Upload to wandb
        try:
            wandb.log(wandb_data)
            print(f"✓ Successfully uploaded spatial reconstruction for sample {sample_idx} at epoch {epoch}")
        except Exception as e:
            print(f"✗ Failed to upload to wandb: {e}")
        
        plt.close(spatial_fig)
    
    def _create_spatial_reconstruction_figure(self, ground_truth, prediction, spatial_mask, 
                                            sample_idx, epoch, mse, mae, correlation):
        """Create clean spatial reconstruction visualization with RGB and Infrared bands."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MAE Reconstruction - Sample {sample_idx}, Epoch {epoch}\n'
                    f'MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}', 
                    fontsize=16, fontweight='bold')
        
        # Row 1: RGB bands (Red, Green, Blue)
        rgb_gt = self._create_rgb_composite(ground_truth, self.rgb_bands)
        rgb_pred = self._create_rgb_composite(prediction, self.rgb_bands)
        
        # Ground Truth RGB
        axes[0, 0].imshow(rgb_gt)
        axes[0, 0].set_title('Ground Truth\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction RGB
        axes[0, 1].imshow(rgb_pred)
        axes[0, 1].set_title('Prediction\n(RGB: B04-B03-B02)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Spatial Mask
        axes[0, 2].imshow(spatial_mask, cmap='RdYlBu_r', alpha=0.9)
        axes[0, 2].set_title(f'Spatial Mask\n(Masked: {np.mean(spatial_mask):.1%})', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Infrared bands (NIR, SWIR1, SWIR2)
        infrared_gt = self._create_elevation_composite(ground_truth, self.infrared_bands)
        infrared_pred = self._create_elevation_composite(prediction, self.infrared_bands)
        
        # Ground Truth Infrared
        axes[1, 0].imshow(infrared_gt)
        axes[1, 0].set_title('Ground Truth\n(False Color: B08-B11-B12)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Prediction Infrared
        axes[1, 1].imshow(infrared_pred)
        axes[1, 1].set_title('Prediction\n(Elevation)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Error Map (using RGB bands)
        rgb_error = np.abs(rgb_gt.astype(float) - rgb_pred.astype(float))
        rgb_error = rgb_error / rgb_error.max() if rgb_error.max() > 0 else rgb_error
        axes[1, 2].imshow(rgb_error)
        axes[1, 2].set_title('RGB Error Map\n(Absolute Difference)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_rgb_composite(self, data, band_indices):
        """Create RGB composite from selected bands."""
        # data: [12, 120, 120], band_indices: [R_idx, G_idx, B_idx]
        composite = np.stack([
            data[band_indices[0]],  # Red channel
            data[band_indices[1]],  # Green channel  
            data[band_indices[2]]   # Blue channel
        ], axis=-1)  # [120, 120, 3]
        
        # Normalize using the provided function
        composite_tensor = torch.from_numpy(composite).permute(2, 0, 1)  # [3, 120, 120]
        normalized = normalize(composite_tensor)  # [3, 120, 120]
        
        return normalized.permute(1, 2, 0).numpy()  # [120, 120, 3]
    
    def _create_elevation_composite(self, data, band_indices):
        """Create RGB composite from selected bands."""
        # data: [12, 120, 120], band_indices: [R_idx, G_idx, B_idx]
        composite = np.stack([
            data[band_indices[-1]],  # Red channel
            data[band_indices[-1]],  # Green channel  
            data[band_indices[-1]]   # Blue channel
        ], axis=-1)  # [120, 120, 3]
        
        # Normalize using the provided function
        composite_tensor = torch.from_numpy(composite).permute(2, 0, 1)  # [3, 120, 120]
        normalized = normalize(composite_tensor)  # [3, 120, 120]
        
        return normalized.permute(1, 2, 0).numpy()  # [120, 120, 3]

