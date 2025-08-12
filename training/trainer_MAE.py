from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ScaleMae import*
from training.ResNet import *
from collections import defaultdict
from training import *
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
util.MESSAGE_LEVEL = util.MessageLevel.INFO
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import torchmetrics
import warnings
import wandb
from transformers import get_cosine_schedule_with_warmup
import seaborn as sns

#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model_MAE(pl.LightningModule):
    def __init__(self, config, wand, name,transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform=transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        self.labels_idx = load_json_to_dict("./data/Encoded-BigEarthNet/labels.json")
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table=False
        self.comment_log=""
        
        # Add manual loss tracking to avoid NaN/inf issues
        self.train_losses = []
        self.val_losses = []
        
        self.metric_MSE_train = torchmetrics.MeanSquaredError(squared=False)
        self.metric_MSE_val_mod_val = torchmetrics.MeanSquaredError(squared=False)
        self.metric_MSE_val_mod_train = torchmetrics.MeanSquaredError(squared=False)
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0
        
        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(config=self.config,transform=self.transform)

        self.loss = nn.MSELoss(reduction='mean')  # Explicitly set reduction
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, image, attention_mask,mae_tokens,mae_tokens_mask, training=False):
        return self.encoder(image, attention_mask,mae_tokens,mae_tokens_mask, training=training)

    def training_step(self, batch, batch_idx):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch
        
        y_hat, y_mask = self.forward(image.clone(), attention_mask.clone(), mae_tokens.clone(), mae_tokens_mask.clone(), training=True)
        
        y_hat_masked = y_hat.clone()
        mae_tokens_masked = mae_tokens.clone()
        
        # Apply masking
        y_hat_masked[y_mask == 1.0] = 0.0
        mae_tokens_masked[y_mask == 1.0] = 0.0
        
        # Compute loss
        loss = self.loss(y_hat_masked[:, :, 0], mae_tokens_masked[:, :, 0])
        
        # Update metrics
        self.metric_MSE_train.update(y_hat_masked[:, :, 0].detach(), mae_tokens_masked[:,:,0].detach())
        
        # Store loss for epoch-end logging
        self.train_losses.append(loss.detach().cpu().item())
        
        # IMPORTANT: Return the loss so PyTorch Lightning can call backward()
        return loss   
    
    def on_fit_start(self):
        # if starting with MAE
        self.encoder.unfreeze_encoder()
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
    
    def on_train_epoch_start(self):
        self.encoder.unfreeze_encoder()    
        self.encoder.unfreeze_decoder()
        self.encoder.freeze_classifier()
        self.train_losses = []  # Reset for new epoch
        
    def on_train_epoch_end(self):
        # Calculate average training loss manually
        if len(self.train_losses) > 0:
            avg_train_loss = np.mean(self.train_losses)
            # Check for NaN/inf
            if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
                avg_train_loss = 0.0
        else:
            avg_train_loss = 0.0
            
        self.log("train_reconstruction_loss", avg_train_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        #self.check_gradients()
        # Compute MSE metric
        try:
            train_mse = self.metric_MSE_train.compute()
            if torch.isnan(train_mse) or torch.isinf(train_mse):
                train_mse = torch.tensor(0.0)
        except:
            train_mse = torch.tensor(0.0)
            
        self.log("train_MSE", train_mse, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_train.reset()
        
        return {"train_reconstruction_loss": avg_train_loss}
    
    def on_validation_epoch_start(self):
        self.trainer.datamodule.val_dataset.set_modality_mode("validation")
        self.val_losses = []  # Reset for new epoch
        
    def on_after_backward(self):
        """Check gradients after each backward pass."""
        if self.global_step % 50 == 0:  # Check every 50 steps
            self.check_encoder_decoder_separately()
            
        # Optional: Check for gradient explosion/vanishing
        total_norm = 0
        param_count = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        
        # Log gradient norm
        if param_count > 0:
            self.log("grad_norm", total_norm, on_step=True, on_epoch=False, logger=True)
            
            # Warning for gradient issues
            if total_norm > 10.0:
                print(f"⚠️ Large gradient norm detected: {total_norm:.4f}")
            elif total_norm < 1e-6:
                print(f"⚠️ Very small gradient norm detected: {total_norm:.8f}")
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch

        y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=False)
        
        # Create copies to avoid in-place operations
        y_hat_masked = y_hat.clone()
        mae_tokens_masked = mae_tokens.clone()
        
        # Apply masking
        y_hat_masked[y_mask == 1.0] = 0.0
        mae_tokens_masked[mae_tokens_mask == 1.0] = 0.0

        # Only compute loss on non-masked tokens to avoid NaN
        valid_mask = (mae_tokens_mask == 0.0)
        
        if valid_mask.sum() == 0:
            # If no valid tokens, return a small loss to avoid NaN
            loss = torch.tensor(0.0, device=self.device)
        else:
            # Only compute loss on valid (non-masked) tokens
            y_hat_valid = y_hat_masked[:, :, 0][valid_mask]
            mae_tokens_valid = mae_tokens_masked[:, :, 0][valid_mask]
            
            loss = self.loss(y_hat_valid, mae_tokens_valid)
        
        # Check for NaN/inf and handle gracefully
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected in validation step {batch_idx}")
            loss = torch.tensor(0.0, device=self.device)
        
        # Update metrics only with valid data
        if valid_mask.sum() > 0:
            if dataloader_idx == 0:  # Validation set
                self.metric_MSE_val_mod_val.update(y_hat_valid.detach(), mae_tokens_valid.detach())
            elif dataloader_idx == 1:  # Training set
                self.metric_MSE_val_mod_train.update(y_hat_valid.detach(), mae_tokens_valid.detach())

        # Store loss for epoch-end logging
        self.val_losses.append(loss.detach().cpu().item())
        
        # Log step-wise (optional, can remove if too verbose)
        self.log("val_reconstruction_loss_step", loss, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        return loss    

    def on_validation_epoch_end(self):
        
            
        
        
        # Compute MSE metric
        val_mse_val = self.metric_MSE_val_mod_val.compute()
        self.log("val mod val MSE", val_mse_val, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_reconstruction_loss", val_mse_val, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_val_mod_val.reset()

        val_mse_train = self.metric_MSE_val_mod_train.compute()
        self.log("val mod train MSE", val_mse_train, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.metric_MSE_val_mod_train.reset()

        return {"val_reconstruction_loss": val_mse_val}

    def test_step(self, batch, batch_idx):
        pass
        
    def save_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        

    def configure_optimizers(self):
        # Import LAMB optimizer
        
        
        # LAMB optimizer optimized for MAE training
        # Consider increasing batch size when using LAMB
        optimizer = Lamb(
            self.parameters(), 
            lr=self.lr * 2.0,    # LAMB often works with higher learning rates
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),  
            eps=1e-6,            
            # For MAE, you might want to experiment with:
            # - Higher learning rates (2x-10x of Adam)
            # - Larger batch sizes (if memory allows)
        )

        accumulate_grad_batches = 64
        batches_per_epoch = self.trainer.estimated_stepping_batches/self.config["trainer"]["epochs"]
        steps_per_epoch = batches_per_epoch // accumulate_grad_batches

        total_steps = self.config["trainer"]["epochs"] * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'val_reconstruction_loss'
            }
        }
        
    def check_encoder_decoder_separately(self):
        """
        Specifically check encoder vs decoder parameters.
        """
        print(f"\n{'='*60}")
        print(f"GRADIENT CHECK - Step {self.global_step}")
        print(f"{'='*60}")
        
        encoder_params = {'total': 0, 'with_grad': 0, 'without_grad': 0, 'frozen': 0, 'grad_norms': []}
        decoder_params = {'total': 0, 'with_grad': 0, 'without_grad': 0, 'frozen': 0, 'grad_norms': []}
        other_params = {'total': 0, 'with_grad': 0, 'without_grad': 0, 'frozen': 0, 'grad_norms': []}
        
        for name, param in self.named_parameters():
            # Categorize parameters
            if any(x in name for x in ['layers', 'latents']):  # Encoder
                category = encoder_params
            elif any(x in name for x in ['decoder', 'recon']):  # Decoder
                category = decoder_params
            else:  # Other (classifier, etc.)
                category = other_params
            
            category['total'] += 1
            
            if not param.requires_grad:
                category['frozen'] += 1
            elif param.grad is None:
                category['without_grad'] += 1
                # Print specific parameters without gradients for debugging
                if 'decoder' in name or 'recon' in name:
                    print(f"⚠️ DECODER param without gradient: {name}")
            else:
                category['with_grad'] += 1
                grad_norm = param.grad.norm().item()
                category['grad_norms'].append(grad_norm)
        
        # Print results with more details
        for cat_name, stats in [("ENCODER", encoder_params), ("DECODER", decoder_params), ("OTHER", other_params)]:
            print(f"\n{cat_name}:")
            print(f"  Total parameters: {stats['total']}")
            print(f"  ✅ With gradients: {stats['with_grad']}")
            print(f"  ⚠️  Without gradients: {stats['without_grad']}")
            print(f"  ❌ Frozen: {stats['frozen']}")
            
            if stats['total'] > 0:
                active_ratio = stats['with_grad'] / stats['total'] * 100
                print(f"  📊 Active ratio: {active_ratio:.1f}%")
                
                # Show gradient statistics
                if stats['grad_norms']:
                    grad_norms = stats['grad_norms']
                    avg_norm = np.mean(grad_norms)
                    max_norm = np.max(grad_norms)
                    min_norm = np.min(grad_norms)
                    print(f"  📈 Grad norm - avg: {avg_norm:.6f}, max: {max_norm:.6f}, min: {min_norm:.6f}")
                    
                    # Warning for problematic gradients
                    if avg_norm > 1.0:
                        print(f"  ⚠️ High average gradient norm in {cat_name}")
                    elif avg_norm < 1e-6:
                        print(f"  ⚠️ Very low average gradient norm in {cat_name}")

    def check_decoder_usage(self):
        """
        Check if decoder components are actually being used in forward pass.
        Call this method to debug decoder connectivity.
        """
        print(f"\n{'='*60}")
        print("DECODER CONNECTIVITY CHECK")
        print(f"{'='*60}")
        
        # Check if decoder modules exist
        decoder_modules = [
            ('decoder_cross_attn', hasattr(self.encoder, 'decoder_cross_attn')),
            ('decoder_ff', hasattr(self.encoder, 'decoder_ff')),
            ('recon_head', hasattr(self.encoder, 'recon_head'))
        ]
        
        print("Decoder modules present:")
        for module_name, exists in decoder_modules:
            status = "✅ Present" if exists else "❌ Missing"
            print(f"  {module_name}: {status}")
        
        # Check if reconstruction mode is being used
        print(f"\nForward pass info:")
        print(f"  Current mode: {'MAE training' if hasattr(self, 'mode') else 'Unknown'}")
        
        # Recommendation
        print(f"\n💡 Debugging recommendations:")
        print(f"  1. Verify decoder is called with reconstruction=True")
        print(f"  2. Check if mae_tokens are properly processed")
        print(f"  3. Ensure transform.process_data works with query=True")
        print(f"  4. Add print statements in reconstruct() method")

    # Add a simple debug method to trace the forward pass
    def debug_forward_pass(self, batch):
        """
        Debug method to trace what happens in the forward pass.
        Call this manually to understand the data flow.
        """
        image, attention_mask, mae_tokens, mae_tokens_mask, labels = batch
        
        print(f"\n{'='*60}")
        print("FORWARD PASS DEBUG")
        print(f"{'='*60}")
        
        print(f"Input shapes:")
        print(f"  image: {image.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  mae_tokens: {mae_tokens.shape}")
        print(f"  mae_tokens_mask: {mae_tokens_mask.shape}")
        
        # Call forward pass
        try:
            y_hat, y_mask = self.forward(image, attention_mask, mae_tokens, mae_tokens_mask, training=True)
            print(f"\nOutput shapes:")
            print(f"  y_hat: {y_hat.shape}")
            print(f"  y_mask: {y_mask.shape}")
            
            print(f"\nOutput statistics:")
            print(f"  y_hat mean: {y_hat.mean().item():.6f}")
            print(f"  y_hat std: {y_hat.std().item():.6f}")
            print(f"  y_hat min: {y_hat.min().item():.6f}")
            print(f"  y_hat max: {y_hat.max().item():.6f}")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Usage: Add this to your training_step for debugging
    def training_step_debug(self, batch, batch_idx):
        # Debug only on first few steps
        if batch_idx < 3:
            self.debug_forward_pass(batch)
            self.check_decoder_usage()
        
        # Your normal training step
        return self.training_step(batch, batch_idx)