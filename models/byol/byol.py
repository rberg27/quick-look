import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
from typing import Tuple, List, Dict, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
from sklearn.manifold import TSNE
import timm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import pandas as pd

import config

import models.byol.contrastive_dataset as dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class EMA(nn.Module):
    """Exponential Moving Average for the target network parameters."""
    def __init__(self, model, decay=0.99):
        super().__init__()
        self.decay = decay
        self.model = model
        self.target_model = None
        self.initialize_target_model()
        
    def initialize_target_model(self):
        """Initialize the target model with the same weights as the online model."""
        self.target_model = type(self.model)(**self.model.get_args())
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False
            
    def update(self):
        """Update target model parameters with EMA."""
        with torch.no_grad():
            for online_param, target_param in zip(self.model.parameters(), 
                                                 self.target_model.parameters()):
                target_param.data = self.decay * target_param.data + \
                                   (1 - self.decay) * online_param.data

class CNNEncoder(nn.Module):
    """CNN-based encoder using ResNet architecture."""
    def __init__(self, output_dim=512):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
        self.fc = nn.Linear(2048, output_dim)
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.fc(features)
    
    def get_args(self):
        return {"output_dim": self.fc.out_features}

class TransformerEncoder(nn.Module):
    """Vision Transformer-based encoder."""
    def __init__(self, output_dim=512):
        super().__init__()
        # Use ViT base model - https://huggingface.co/google/vit-base-patch16-224
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.vit.gradient_checkpointing = True
        self.vit.head = nn.Linear(self.vit.head.in_features, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):
        return self.vit(x)
    
    def get_args(self):
        return {"output_dim": self.output_dim}

class DualEncoder(nn.Module):
    """Dual encoder with both CNN and Transformer paths."""
    def __init__(self, output_dim=512, fusion='concat'):
        super().__init__()
        self.output_dim = output_dim
        self.fusion = fusion
        
        # CNN path
        self.cnn = CNNEncoder(output_dim=output_dim)
        
        # Transformer path
        self.transformer = TransformerEncoder(output_dim=output_dim)
        
        # Fusion layer
        if fusion == 'concat':
            self.fusion_layer = nn.Linear(output_dim * 2, output_dim)
        else:  # average or max
            self.fusion_layer = nn.Identity()
            
    def forward(self, x):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        
        if self.fusion == 'concat':
            combined = torch.cat([cnn_features, transformer_features], dim=1)
            output = self.fusion_layer(combined)
        elif self.fusion == 'average':
            output = (cnn_features + transformer_features) / 2
        elif self.fusion == 'max':
            output = torch.max(cnn_features, transformer_features)
            
        return output
    
    def get_args(self):
        return {"output_dim": self.output_dim, "fusion": self.fusion}

class ProjectionNetwork(nn.Module):
    """MLP projection network."""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.layer2(x)
        return x
    
    def get_args(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim
        }

class PredictionNetwork(nn.Module):
    """MLP prediction network."""
    def __init__(self, input_dim=256, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.layer2(x)
        return x
    
    def get_args(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim
        }

class ModifiedBYOL(pl.LightningModule):
    """Modified BYOL model with negative pair handling."""
    def __init__(
        self,
        encoder_type='dual',  # 'cnn', 'transformer', or 'dual'
        encoder_output_dim=512,
        projector_hidden_dim=1024,
        projector_output_dim=256,
        predictor_hidden_dim=1024,
        ema_decay=0.99,
        learning_rate=3e-4,
        weight_decay=1e-6,
        negative_weight=0.5,  # Weight for negative pairs
        temperature=0.1,      # Temperature for similarity scaling
        log_embedding_dims=2,  # Dimensions for embedding visualization
        log_every_n_steps=100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Select encoder type
        if encoder_type == 'cnn':
            self.online_encoder = CNNEncoder(output_dim=encoder_output_dim)
        elif encoder_type == 'transformer':
            self.online_encoder = TransformerEncoder(output_dim=encoder_output_dim)
        else:  # dual
            self.online_encoder = DualEncoder(output_dim=encoder_output_dim)
            
        # Create online projector and predictor
        self.online_projector = ProjectionNetwork(
            input_dim=encoder_output_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim
        )
        
        self.online_predictor = PredictionNetwork(
            input_dim=projector_output_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_output_dim
        )
        
        # Create target network with EMA
        self.target_ema = EMA(
            model=self.online_encoder,
            decay=ema_decay
        )
        
        self.target_projector = ProjectionNetwork(
            input_dim=encoder_output_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim
        )
        
        # Initialize target projector with online projector weights
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        
        # Freeze target network parameters
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
        # Learning parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negative_weight = negative_weight
        self.temperature = temperature
        
        # Logging parameters
        self.log_embedding_dims = log_embedding_dims
        self.log_every_n_steps = log_every_n_steps
        self.writer = SummaryWriter()
        
    def forward(self, x):
        """Forward pass to get embeddings for inference."""
        return self.online_encoder(x)
    
    def shared_step(self, batch):
        """Shared step for both training and validation."""
        # Batch contains: view1, view2, is_negative_pair
        view1, view2, is_negative_pair = batch
        
        # Online network forward passes
        online_proj1 = self.online_projector(self.online_encoder(view1))
        online_proj2 = self.online_projector(self.online_encoder(view2))
        
        # Predictor outputs
        pred1 = self.online_predictor(online_proj1)
        pred2 = self.online_predictor(online_proj2)
        
        # Target network forward passes (no grad)
        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_ema.target_model(view1))
            target_proj2 = self.target_projector(self.target_ema.target_model(view2))
            
        # Normalize projections and predictions
        pred1 = F.normalize(pred1, dim=-1)
        pred2 = F.normalize(pred2, dim=-1)
        target_proj1 = F.normalize(target_proj1, dim=-1)
        target_proj2 = F.normalize(target_proj2, dim=-1)
        
        # Calculate positive and negative losses
        pos_loss = 0.5 * (
            2 - 2 * (pred1 * target_proj2).sum(dim=-1) +
            2 - 2 * (pred2 * target_proj1).sum(dim=-1)
        )
        
        # Convert similarity to a loss where similar pairs have low loss
        # For negative pairs, we want high similarity to result in high loss
        sim1 = (pred1 * target_proj2).sum(dim=-1) / self.temperature
        sim2 = (pred2 * target_proj1).sum(dim=-1) / self.temperature
        
        # Apply negative weight only to negative pairs
        neg_mask = is_negative_pair.float()
        pos_mask = 1.0 - neg_mask
        
        # Calculate weighted loss
        loss = (pos_mask * pos_loss) - (neg_mask * self.negative_weight * (sim1 + sim2))
        
        # Calculate mean loss
        loss = loss.mean()
        
        return loss, online_proj1, online_proj2, is_negative_pair

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, proj1, proj2, is_negative = self.shared_step(batch)
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log embeddings periodically
        if batch_idx % self.log_every_n_steps == 0:
            self._log_embeddings(proj1, proj2, is_negative, 'train')
            
        # Update target network with EMA
        self.target_ema.update()
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, proj1, proj2, is_negative = self.shared_step(batch)
        
        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log embeddings on first batch
        if batch_idx == 0:
            self._log_embeddings(proj1, proj2, is_negative, 'val')
            
        return loss
    
    def _log_embeddings(self, proj1, proj2, is_negative, prefix):
        """Log embeddings and visualize using dimensionality reduction."""
        # Limit to a smaller subset for visualization
        max_samples = 200
        if len(proj1) > max_samples:
            indices = torch.randperm(len(proj1))[:max_samples]
            proj1 = proj1[indices]
            proj2 = proj2[indices]
            is_negative = is_negative[indices]
        
        # Concatenate embeddings from both views
        all_proj = torch.cat([proj1, proj2], dim=0)
        
        # Map is_negative to the concatenated tensor
        pair_labels = torch.cat([is_negative, is_negative], dim=0)
        
        # Track whether each embedding comes from view1 or view2
        view_labels = torch.cat([
            torch.zeros(len(proj1), dtype=torch.long),
            torch.ones(len(proj2), dtype=torch.long)
        ], dim=0)
        
        # Calculate embedding statistics
        embedding_stats = {
            'mean_norm': torch.norm(all_proj, dim=1).mean().item(),
            'std_norm': torch.norm(all_proj, dim=1).std().item(),
            'mean_cosine_sim': torch.mm(all_proj, all_proj.t()).mean().item(),
            'std_cosine_sim': torch.mm(all_proj, all_proj.t()).std().item(),
            'positive_pair_sim': torch.mm(proj1[~is_negative], proj2[~is_negative].t()).mean().item() if (~is_negative).any() else 0.0,
            'negative_pair_sim': torch.mm(proj1[is_negative], proj2[is_negative].t()).mean().item() if is_negative.any() else 0.0
        }
        
        # Log embedding statistics
        for name, value in embedding_stats.items():
            self.log(f'{prefix}_{name}', value, on_step=True, on_epoch=True)
            
        # Save statistics to CSV for later analysis
        stats_df = pd.DataFrame([embedding_stats])
        stats_path = f"embedding_stats/{prefix}_step_{self.global_step}.csv"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        
        # Log embeddings
        self.logger.experiment.add_embedding(
            all_proj,
            metadata=[
                f"View{'1' if v==0 else '2'}_{'Neg' if n else 'Pos'}"
                for v, n in zip(view_labels.cpu(), pair_labels.cpu())
            ],
            global_step=self.global_step,
            tag=f"{prefix}_embeddings"
        )
        
        # Create embedding visualization using T-SNE
        all_proj_np = all_proj.detach().cpu().numpy()
        tsne = TSNE(n_components=self.log_embedding_dims, perplexity=min(30, all_proj_np.shape[0] // 4))
        reduced_data = tsne.fit_transform(all_proj_np)
        
        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Define color scheme
        colors = np.array([
            'blue' if (v == 0 and n == 0) else
            'green' if (v == 1 and n == 0) else
            'red' if (v == 0 and n == 1) else
            'orange'
            for v, n in zip(view_labels.cpu().numpy(), pair_labels.cpu().numpy())
        ])
        
        # Define marker scheme
        markers = np.array(['o' if v == 0 else '^' for v in view_labels.cpu().numpy()])
        
        # Create scatter plot
        for c, m in set(zip(colors, markers)):
            mask = (colors == c) & (markers == m)
            ax.scatter(
                reduced_data[mask, 0], 
                reduced_data[mask, 1],
                c=c,
                marker=m,
                alpha=0.7,
                label=f"{'View1' if m=='o' else 'View2'}_{'Neg' if c in ['red', 'orange'] else 'Pos'}"
            )
            
        ax.legend()
        ax.set_title(f"{prefix.capitalize()} Embeddings T-SNE (Step {self.global_step})")
        ax.axis('equal')
        
        # Save figure to disk or log to tensorboard
        fig_path = f"embedding_plots/{prefix}_step_{self.global_step}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # Log to tensorboard
        self.logger.experiment.add_figure(
            f"{prefix}_tsne_embeddings",
            fig,
            global_step=self.global_step
        )
        
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=100,  # Number of epochs
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
def save_model_weights(model, output_dir='./model_weights'):
    """Save model weights to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save encoder weights
    encoder_path = os.path.join(output_dir, 'encoder_weights.pth')
    torch.save(model.online_encoder.state_dict(), encoder_path)
    
    # Save projector weights
    projector_path = os.path.join(output_dir, 'projector_weights.pth')
    torch.save(model.online_projector.state_dict(), projector_path)
    
    # Save predictor weights
    predictor_path = os.path.join(output_dir, 'predictor_weights.pth')
    torch.save(model.online_predictor.state_dict(), predictor_path)
    
    # Save hyperparameters
    hyperparams = {
        'encoder_type': model.hparams.encoder_type,
        'encoder_output_dim': model.hparams.encoder_output_dim,
        'projector_hidden_dim': model.hparams.projector_hidden_dim,
        'projector_output_dim': model.hparams.projector_output_dim,
        'predictor_hidden_dim': model.hparams.predictor_hidden_dim,
        'ema_decay': model.hparams.ema_decay,
        'learning_rate': model.hparams.learning_rate,
        'weight_decay': model.hparams.weight_decay,
        'negative_weight': model.hparams.negative_weight,
        'temperature': model.hparams.temperature
    }
    hyperparams_path = os.path.join(output_dir, 'hyperparameters.json')
    import json
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f)
    
    print(f"Model weights and hyperparameters saved to {output_dir}")

def load_model_weights(model, weights_dir=config.BYOL_MODEL_WEIGHTS_PATH):
    """Load model weights from disk."""
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory {weights_dir} does not exist")
    
    # Load encoder weights
    encoder_path = os.path.join(weights_dir, 'encoder_weights.pth')
    if os.path.exists(encoder_path):
        model.online_encoder.load_state_dict(torch.load(encoder_path))
    
    # Load projector weights
    projector_path = os.path.join(weights_dir, 'projector_weights.pth')
    if os.path.exists(projector_path):
        model.online_projector.load_state_dict(torch.load(projector_path))
    
    # Load predictor weights
    predictor_path = os.path.join(weights_dir, 'predictor_weights.pth')
    if os.path.exists(predictor_path):
        model.online_predictor.load_state_dict(torch.load(predictor_path))
    
    print(f"Model weights loaded from {weights_dir}")
    return model

def train_model(
    _image_list,
    output_dir='./model_outputs',
    max_epochs=100,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-6,
    encoder_type='dual',
    negative_weight=0.5,
    temperature=0.1,
    accelerator='gpu',
    devices=1,
    subset=None,
    load_weights=False,
    weights_dir='./model_weights'
):
    """Train the modified BYOL model."""
    # Create model
    model = ModifiedBYOL(
        encoder_type=encoder_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        negative_weight=negative_weight,
        temperature=temperature
    )
    
    # Load weights if specified
    if load_weights:
        model = load_model_weights(model, weights_dir)
        print("Weights loaded successfully. Skipping training.")
        return model
    
    # Create dataloaders
    train_loader, val_loader = dataset.get_dataloaders(
        _image_list,
        batch_size=batch_size,
        subset=subset
    )
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",
        log_every_n_steps=50,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
                filename='{epoch}-{val_loss:.4f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final weights
    save_model_weights(model, weights_dir)
    
    # Return the trained model
    return model

def test_embeddings(
    model,
    _dataset,
    output_dir='./embedding_tests',
    batch_size=16,
    num_samples=None
):
    """Test and visualize the embeddings created by the model using a contrastive dataset.
    
    Args:
        model: The trained BYOL model
        dataset: A contrastive dataset (e.g., SceneMaskDataset)
        output_dir: Directory to save test results
        batch_size: Batch size for processing
        num_samples: Optional number of samples to test (if None, use all data)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create result data structure
    results = []
    embeddings = []
    labels = []
    paths = []
    
    # Create dataloader
    dataloader = DataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Process batches
    for batch_idx, (view1, view2, is_negative) in enumerate(dataloader):
        # Move to device
        device = next(model.parameters()).device
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Get embeddings with no grad
        with torch.no_grad():
            view1_embeddings = model(view1)
            view2_embeddings = model(view2)
            
        # Calculate similarities
        for j in range(len(view1)):
            view1_emb = view1_embeddings[j]
            view2_emb = view2_embeddings[j]
            
            # Normalize embeddings
            view1_emb = F.normalize(view1_emb, dim=0)
            view2_emb = F.normalize(view2_emb, dim=0)
            
            # Add embeddings to list for visualization
            embeddings.extend([view1_emb.cpu().numpy(), view2_emb.cpu().numpy()])
            labels.extend(['view1', 'view2'])
            paths.extend([f'batch_{batch_idx}_sample_{j}_view1', f'batch_{batch_idx}_sample_{j}_view2'])
            
            # Calculate cosine similarity score
            similarity = torch.dot(view1_emb, view2_emb).item()
            
            # Store results
            results.append({
                'batch_idx': batch_idx,
                'sample_idx': j,
                'is_negative': is_negative[j].item(),
                'similarity': similarity
            })
            
        # Break if we've processed enough samples
        if num_samples is not None and len(results) >= num_samples:
            break
    
    # Convert results to DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'similarity_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Calculate statistics
    stats = {
        'mean_similarity': df['similarity'].mean(),
        'std_similarity': df['similarity'].std(),
        'mean_positive_similarity': df[~df['is_negative']]['similarity'].mean(),
        'mean_negative_similarity': df[df['is_negative']]['similarity'].mean(),
        'std_positive_similarity': df[~df['is_negative']]['similarity'].std(),
        'std_negative_similarity': df[df['is_negative']]['similarity'].std(),
    }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_csv_path = os.path.join(output_dir, 'similarity_stats.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    
    # Create histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot overall similarity distribution
    axs[0].hist(df['similarity'], bins=20, alpha=0.7)
    axs[0].set_title('Overall Similarity Distribution')
    axs[0].set_xlabel('Cosine Similarity')
    axs[0].set_ylabel('Count')
    
    # Plot positive vs negative similarity distributions
    axs[1].hist(df[~df['is_negative']]['similarity'], bins=20, alpha=0.7, label='Positive Pairs')
    axs[1].hist(df[df['is_negative']]['similarity'], bins=20, alpha=0.7, label='Negative Pairs')
    axs[1].set_title('Positive vs Negative Pair Similarities')
    axs[1].set_xlabel('Cosine Similarity')
    axs[1].legend()
    
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'similarity_histograms.png')
    plt.savefig(hist_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Create T-SNE visualization of embeddings
    embeddings_array = np.array(embeddings)
    tsne = TSNE(n_components=2, perplexity=min(30, embeddings_array.shape[0] // 4))
    reduced_data = tsne.fit_transform(embeddings_array)
    
    # Plot T-SNE
    plt.figure(figsize=(10, 8))
    colors = {'view1': 'blue', 'view2': 'green'}
    
    for label in colors.keys():
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            reduced_data[indices, 0], 
            reduced_data[indices, 1],
            c=colors[label],
            label=label,
            alpha=0.7
        )
    
    plt.legend()
    plt.title('T-SNE Visualization of Embeddings')
    plt.axis('equal')
    
    tsne_path = os.path.join(output_dir, 'embedding_tsne.png')
    plt.savefig(tsne_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"Embedding test results saved to {output_dir}")
    print(f"Mean Similarity: {stats['mean_similarity']:.3f}")
    print(f"Mean Positive Pair Similarity: {stats['mean_positive_similarity']:.3f}")
    print(f"Mean Negative Pair Similarity: {stats['mean_negative_similarity']:.3f}")
    
    return df, stats

def train_model(
    _image_list,
    output_dir='./model_outputs',
    max_epochs=100,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-6,
    encoder_type='dual',
    negative_weight=0.5,
    temperature=0.1,
    accelerator='gpu',
    devices=1,
    subset=None,
    load_weights=False,
    weights_dir='./model_weights'
):
    """Train the modified BYOL model."""
    # Create model
    model = ModifiedBYOL(
        encoder_type=encoder_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        negative_weight=negative_weight,
        temperature=temperature
    )
    
    # Load weights if specified
    if load_weights:
        model = load_model_weights(model, weights_dir)
        print("Weights loaded successfully. Skipping training.")
        return model
    
    # Create dataloaders
    train_loader, val_loader = dataset.get_dataloaders(
        _image_list,
        batch_size=batch_size,
        subset=subset
    )
    
    # Verify dataloaders have data
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Empty dataset detected. Check if the data paths are correct and files exist.")
    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",
        log_every_n_steps=50,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
                filename='{epoch}-{val_loss:.4f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final weights
    save_model_weights(model, weights_dir)
    
    # Return the trained model
    return model