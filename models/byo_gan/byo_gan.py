# GAN architecture
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple, List, Dict, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
from sklearn.manifold import TSNE
import timm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# Main training script
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import models.byol.contrastive_dataset as dataset

def display_tensor_image(tensor, title="Image", normalize=True):
    """
    Display a tensor as an image in a pop-up window.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape [C, H, W] or [B, C, H, W]
        title (str): Title for the window
        normalize (bool): Whether to normalize the image for display
    """
    import matplotlib.pyplot as plt
    
    # Make a copy to avoid modifying the original tensor
    img = tensor.clone().detach()
    
    # Handle batch dimension if present
    if img.dim() == 4:
        img = img[0]  # Take the first image in the batch
    
    # Move to CPU if on another device
    img = img.cpu()
    
    # Convert to numpy and transpose from [C, H, W] to [H, W, C]
    img_np = img.permute(1, 2, 0).numpy()
    
    # Normalize to [0, 1] range if requested
    if normalize:
        if img_np.max() > 1.0 or img_np.min() < 0.0:
            # If image is not in [0, 1] range, normalize it
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Create figure and display
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


class Generator(nn.Module):
    """Generator for inpainting with cross-attention."""
    def __init__(self, latent_dim=256, channels=64, byol_output_dim=512):
        super().__init__()
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Initial linear projection for random latent
        self.fc = nn.Linear(latent_dim, 4 * 4 * channels * 8)
        
        # Linear projection for BYOL embedding
        self.byol_proj = nn.Linear(byol_output_dim, latent_dim)
        
        # Upsampling blocks
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(channels * 8, channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 224x224 (use 3x3 conv with padding to get to 224x224)
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, byol_embedding, mask, masked_image):
        '''
        z: random latent vector
        byol_embedding: embedding from BYOL model
        mask: mask of the image
        masked_image: masked image (assumes that the masked region is already removed and is all zeros)
        '''
        # Project BYOL embedding to latent space
        byol_latent = self.byol_proj(byol_embedding)
        
        # Reshape latents for cross-attention
        z = z.unsqueeze(1)  # [B, 1, latent_dim]
        byol_latent = byol_latent.unsqueeze(1)  # [B, 1, latent_dim]
        
        # Apply cross-attention
        attn_output, _ = self.cross_attention(
            query=z,
            key=byol_latent,
            value=byol_latent
        )
        
        # Combine latents
        combined_latent = attn_output.squeeze(1)  # [B, latent_dim]
        
        # Generate image
        x = self.fc(combined_latent)
        x = x.view(-1, 512, 4, 4)  # Reshape to 4x4 feature maps
        x = self.decoder(x)
        
        # Apply mask and combine with masked image
        generated_content = x * mask
        combined_image = generated_content + masked_image
        
        return combined_image

class Discriminator(nn.Module):
    """Discriminator for GAN."""
    def __init__(self, channels=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(channels * 4, channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(channels * 8, channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 3x3
            nn.Conv2d(channels * 8, channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3x3 -> 1x1
            nn.Conv2d(channels * 8, 1, kernel_size=3, stride=1, padding=0)#,
            #nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x).view(-1, 1)

class EmbeddingGuidedGAN(pl.LightningModule):
    """GAN that uses embeddings from BYOL to guide generation."""
    def __init__(
        self,
        byol_model=None,  # Make BYOL model optional
        latent_dim=256,
        channels=64,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        lambda_real=0.18,
        lambda_embedding=0.72,
        lambda_reconstruction=0.36,  # Weight for optional reconstruction loss
        use_reconstruction_loss=True,  # Whether to use reconstruction loss
        weights_dir=config.BYO_GAN_MODEL_WEIGHTS_DIR  # Directory to save weights
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['byol_model'])
        
        # Set automatic optimization to False for multiple optimizers
        self.automatic_optimization = False
        
        # Frozen BYOL model (if provided)
        self.byol_model = byol_model
        if self.byol_model is not None:
            for param in self.byol_model.parameters():
                param.requires_grad = False
            self.byol_model.eval()
        
        # GAN components
        self.generator = Generator(
            latent_dim=latent_dim,
            channels=channels,
            byol_output_dim=512  # BYOL output dimension
        )
        self.discriminator = Discriminator(channels=channels)
        
        # Learning hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Loss weights
        self.lambda_real = lambda_real
        self.lambda_embedding = lambda_embedding
        self.lambda_reconstruction = lambda_reconstruction
        self.use_reconstruction_loss = use_reconstruction_loss
        
        # Sample images for logging
        self.sample_masked_images = None
        self.sample_masks = None
        self.sample_original_images = None
        
        # Directory to save weights
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        
    def forward(self, masked_images, masks):
        """Generate full images from masked images and masks."""
        # Get BYOL embeddings
        with torch.no_grad():
            byol_embeddings = self.byol_model(masked_images)
        
        # Generate random latent vector
        latent = torch.randn(masked_images.size(0), self.hparams.latent_dim, device=self.device)
        
        # Generate full image
        generated_full_images = self.generator(latent, byol_embeddings, masks, masked_images)
        return generated_full_images
    
    def adversarial_loss(self, y_pred, is_real):
        """Adversarial loss function."""
        target = torch.ones_like(y_pred) if is_real else torch.zeros_like(y_pred)
        return F.binary_cross_entropy_with_logits(y_pred, target)
    
    def embedding_loss(self, original_embedding, generated_embedding):
        """Embedding similarity loss."""
        # Normalize embeddings
        original_embedding = F.normalize(original_embedding, dim=1)
        generated_embedding = F.normalize(generated_embedding, dim=1)
        
        # Calculate cosine similarity (should be high)
        similarity = (original_embedding * generated_embedding).sum(dim=1).mean()
        
        # Convert to a loss (lower value for higher similarity)
        return 1 - similarity
    
    def reconstruction_loss(self, original, generated):
        """Reconstruction loss between original and generated images."""
        return F.l1_loss(generated, original)
    
    def training_step(self, batch, batch_idx):
        """Training step for GAN with manual optimization."""
        masked_images, masks, original_images = batch
        
        # Get optimizers
        opt_d, opt_g = self.optimizers()
        
        # Generate full images
        generated_full_images = self(masked_images, masks)
        
        # Train discriminator
        opt_d.zero_grad()
        
        # Real images
        real_validity = self.discriminator(original_images)
        real_loss = self.adversarial_loss(real_validity, True)
        
        # Fake images
        fake_validity = self.discriminator(generated_full_images.detach())
        fake_loss = self.adversarial_loss(fake_validity, False)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        # Manually optimize discriminator
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train generator
        opt_g.zero_grad()
        
        # Adversarial loss
        validity = self.discriminator(generated_full_images)
        g_adv_loss = self.adversarial_loss(validity, True)
        
        # Embedding loss (only if BYOL model is provided)
        g_emb_loss = 0
        if self.byol_model is not None:
            with torch.no_grad():
                masked_embeddings = self.byol_model(masked_images)
            # Assert that masks are binary (contain only 0 and 1)
            assert len(torch.unique(masks[0])) == 2, f"Mask should be binary but contains {len(torch.unique(masks[0]))} unique values"
            generated_inpainted_images = generated_full_images * masks
            generated_embeddings = self.byol_model(generated_inpainted_images)
            g_emb_loss = self.embedding_loss(masked_embeddings, generated_embeddings)
        
        # Reconstruction loss (optional)
        g_rec_loss = 0
        if self.use_reconstruction_loss:
            # Focus on the masked region
            masked_region_original = original_images * masks
            masked_region_generated = generated_full_images * masks
            g_rec_loss = self.reconstruction_loss(masked_region_original, masked_region_generated)
            
        # Total generator loss
        g_loss = self.lambda_real * g_adv_loss + \
                 self.lambda_embedding * g_emb_loss + \
                 self.lambda_reconstruction * g_rec_loss
        
        # Manually optimize generator
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Log metrics
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('g_adv_loss', g_adv_loss)
        if self.byol_model is not None:
            self.log('g_emb_loss', g_emb_loss)
        
        if self.use_reconstruction_loss:
            self.log('g_rec_loss', g_rec_loss)
            
        # Save weights every 5 epochs
        if (self.current_epoch + 1) % 5 == 0 and batch_idx == 0:
            self.save_model_weights(f"epoch_{self.current_epoch + 1}")
            
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step for GAN."""
        masked_images, masks, original_images = batch
        
        # Store sample images for visualization
        if batch_idx == 0 and self.sample_masked_images is None:
            self.sample_masked_images = masked_images[:8].clone()
            self.sample_masks = masks[:8].clone()
            self.sample_original_images = original_images[:8].clone()
            
        # Generate full images
        generated_full_images = self(masked_images, masks)
        
        # Calculate losses
        # Adversarial loss
        validity = self.discriminator(generated_full_images)
        g_adv_loss = self.adversarial_loss(validity, True)
        
        # Embedding loss
        g_emb_loss = 0
        if self.byol_model is not None:
            with torch.no_grad():
                original_embeddings = self.byol_model(original_images)
            generated_embeddings = self.byol_model(generated_full_images)
            g_emb_loss = self.embedding_loss(original_embeddings, generated_embeddings)
        
        # Reconstruction loss (optional)
        g_rec_loss = 0
        if self.use_reconstruction_loss:
            # Focus on the masked region
            masked_region_original = original_images * masks
            masked_region_generated = generated_full_images * masks
            g_rec_loss = self.reconstruction_loss(masked_region_original, masked_region_generated)
            
        # Total generator loss
        g_loss = g_adv_loss + \
                 self.lambda_real * g_emb_loss + \
                 self.lambda_reconstruction * g_rec_loss
        
        # Log metrics
        self.log('val_g_loss', g_loss)
        self.log('val_g_adv_loss', g_adv_loss)
        if self.byol_model is not None:
            self.log('val_g_emb_loss', g_emb_loss)
        
        if self.use_reconstruction_loss:
            self.log('val_g_rec_loss', g_rec_loss)
            
        return g_loss
    
    def on_validation_epoch_end(self):
        """Log sample images at the end of validation epoch."""
        if self.sample_masked_images is not None:
            # Generate full images
            self.eval()
            with torch.no_grad():
                generated_full_images = self(self.sample_masked_images, self.sample_masks)
            self.train()
            
            # Create grid of images
            grid_images = []
            for i in range(min(4, len(self.sample_masked_images))):
                # Unnormalize images
                masked = self.sample_masked_images[i].cpu()
                mask = self.sample_masks[i].cpu()
                original = self.sample_original_images[i].cpu()
                generated = generated_full_images[i].cpu()
                
                # Convert to [0,1] range
                masked = (masked * 0.5) + 0.5
                mask = (mask * 0.5) + 0.5
                original = (original * 0.5) + 0.5
                generated = (generated * 0.5) + 0.5
                
                # Add to grid
                grid_images.extend([masked, mask, original, generated])
            
            # Create grid
            grid = torchvision.utils.make_grid(grid_images, nrow=4)
            
            # Log to tensorboard
            self.logger.experiment.add_image(
                'val_samples', grid, self.current_epoch
            )
    
    def configure_optimizers(self):
        """Configure optimizers for GAN."""
        # Discriminator optimizer
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        # Generator optimizer
        opt_g = torch.optim.Adam(
            list(self.generator.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        return [opt_d, opt_g], []
    
    def save_model_weights(self, name):
        """Save model weights to disk."""
        save_path = os.path.join(self.weights_dir, f"{name}.pt")
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, save_path)
        print(f"Model weights saved to {save_path}")
    
    def load_model_weights(self, weights_path=config.DEFAULT_BYO_GAN_MODEL_WEIGHTS_PATH):
        """Load model weights from disk."""
        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return False
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        print(f"Model weights loaded from {weights_path}")
        return True

def train_embedding_guided_gan(
    _dataset,
    output_dir='./gan_outputs',
    weights_dir=config.BYO_GAN_MODEL_WEIGHTS_DIR,
    byol_model=None,
    max_epochs=100,
    batch_size=16,
    accelerator='gpu',
    devices=1,
    load_weights=False,
    loaded_model_name="final.pt",
    save_weights=True,
    subset=None
):
    """Train the EmbeddingGuidedGAN."""
    # Create model
    model = EmbeddingGuidedGAN(
        byol_model=byol_model,
        latent_dim=256,
        channels=64,
        beta1=0.5,
        beta2=0.999,
        learning_rate=0.0002,
        lambda_real=.18,
        lambda_reconstruction=.82,
        use_reconstruction_loss=True,
        weights_dir=config.BYO_GAN_MODEL_WEIGHTS_DIR
    )


    
    if load_weights:
        weights_path = os.path.join(weights_dir, loaded_model_name)
        print(f"Loading pretrained model from {weights_path}")
        model.load_model_weights(weights_path)
        model.eval()
    else:
        # Create dataloaders
        if subset is not None:
            # Create a new dataset by sampling a subset of the original dataset
            dataset_size = len(_dataset)
            indices = torch.randperm(dataset_size)[:subset].tolist()
            sampled_dataset = torch.utils.data.Subset(_dataset, indices)
            _dataset = sampled_dataset
            print(f"Using subset of {subset} images from a total of {dataset_size} images")
           
        tr_dataset, va_dataset = torch.utils.data.random_split(_dataset, [0.8, 0.2])

        train_loader = DataLoader(
            tr_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            va_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        # Create trainer
        trainer = pl.Trainer(
            default_root_dir=output_dir,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision="16-mixed",
            log_every_n_steps=5,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor='val_g_loss',
                    mode='min',
                    save_top_k=3,
                    save_last=True,
                    filename='{epoch}-{val_g_loss:.4f}'
                ),
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
                pl.callbacks.EarlyStopping(
                    monitor='val_g_loss',
                    patience=10,
                    mode='min'
                )
            ]
        )
    
        # Train model
        trainer.fit(model, train_loader, val_loader)
    
    # Save final weights
    if save_weights:
        model.save_model_weights("final")
    
    # Return the trained model
    return model

def test_embedding_gan(
    gan_model,
    masked_image_tensor,
    mask_tensor,
    original_image_tensor,
    output_dir,
    filename="gan_result.png"
):
    """
    Test the StencilGAN model on a masked image and save the result.
    
    Args:
        gan_model: Trained StencilGAN model
        masked_image_tensor: Tensor containing the masked image
        mask_tensor: Tensor containing the mask
        original_image_tensor: Tensor containing the original image
        output_dir: Directory to save the output image
        filename: Name of the output file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

   # Generate and save sample results
    print("Generating sample results...")
    test_output_dir = output_dir
    os.makedirs(test_output_dir, exist_ok=True)

    masked_image = masked_image_tensor.unsqueeze(0).to(gan_model.device)
    mask = mask_tensor.unsqueeze(0).to(gan_model.device)
        
    # Generate the image
    with torch.no_grad():
        generated_image = gan_model(masked_image, mask)
        
    # Convert tensors to images for display
    generated_image = generated_image.squeeze(0).cpu()
    original_image = original_image_tensor.cpu()
        
    # Create a side-by-side comparison
    comparison = torch.cat([original_image, generated_image], dim=2)  # Concatenate along width
        
    # Save the comparison image
    torchvision.utils.save_image(
        comparison, 
        os.path.join(test_output_dir, filename),
        normalize=True
    )

    torchvision.utils.save_image(
        generated_image, 
        os.path.join(test_output_dir, f"gen_from_{filename}.jpg"),
        normalize=True
    )

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import models.byo_gan.byo_gan_dataset as byo_gan_dataset
    import models.byol.byol as byol
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train EmbeddingGuidedGAN on BYO GAN dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space')
    parser.add_argument('--output_dir', type=str, default='./byo_gan_results', help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--subset', type=int, default=None, help='Subset of dataset to use for training')
    parser.add_argument('--byol_weights', type=str, default=config.BYOL_MODEL_WEIGHTS_PATH, help='Path to pretrained BYOL weights')
    parser.add_argument('--byol_encoder_type', type=str, default='dual', help='BYOL encoder type (default: dual)')
    parser.add_argument('--load', action='store_true', help='Load a pretrained model instead of training')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained model weights (required if --load is True)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and load pretrained BYOL model
    print("Loading pretrained BYOL model...")
    byol_model = byol.ModifiedBYOL(
        encoder_type=args.byol_encoder_type,
        encoder_output_dim=512,
        projector_hidden_dim=1024,
        projector_output_dim=256,
        predictor_hidden_dim=1024,
        ema_decay=0.99,
        learning_rate=3e-4,
        weight_decay=1e-6,
        negative_weight=0.5,
        temperature=0.1
    )
    
    # Load the weights
    byol_model = byol.load_model_weights(byol_model, args.byol_weights)
    byol_model.eval()  # Set to evaluation mode
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = byo_gan_dataset.BYOGANDataset(
        masked_image_dir=config.MASKED_DATASET_PATH,
        mask_dir=config.MASKS_DATASET_PATH,
        original_image_dir=config.ORIGINAL_DATASET_PATH,
        img_size=224,
        subset=args.subset
    )

    # Split the dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    
    # Create EmbeddingGuidedGAN model
    print("Creating EmbeddingGuidedGAN model...")
    model = EmbeddingGuidedGAN(
        byol_model=byol_model,
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        lambda_embedding=0.18,
        lambda_reconstruction=0.82,
        use_reconstruction_loss=True,
        weights_dir=os.path.join(args.output_dir, 'weights')
    )
    
    model = train_embedding_guided_gan(
        train_dataset,
        output_dir=args.output_dir,
        weights_dir=os.path.join(args.output_dir, 'weights'),
        max_epochs=args.epochs,
        load_weights=args.load,
        byol_model=byol_model
    )
    
    # Generate and save sample results
    print("Generating sample results...")
    test_output_dir = os.path.join(args.output_dir, 'test_samples')
    print(f"Creating test output directory at: {test_output_dir}")
    os.makedirs(test_output_dir, exist_ok=True)

    for i in range(5):
        masked_image, mask, original_image = val_dataset[i]
        test_embedding_gan(
            model,
            masked_image,
            mask,
            original_image,
            test_output_dir,
            filename=f"sample_{i}_comparison.jpg"
        )
    
    print(f"Process complete! Results saved to {args.output_dir}")
    print(f"Test samples directory contents: {os.listdir(test_output_dir)}")