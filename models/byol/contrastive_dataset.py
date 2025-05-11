# Dataset and data loading
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
import numpy as np


class SceneMaskDataset(Dataset):
    """Dataset for scene-mask pairs with inpainted masks."""
    def __init__(
        self,
        masked_image_dir,
        chair_crops_dir,
        inpainted_masks_dir,
        img_size=224,
        positive_ratio=0.7,
        include_orig_orig=True,
        include_mask_mask=True,
        image_file_names=None
    ):
        """Initialize the dataset.
        
        Args:
            image_dir: Directory with masked images
            mask_dir: Directory with original masks
            inpaint_dir: Directory with inpainted masks
            transform: Optional additional transforms
            img_size: Size to resize images
            positive_ratio: Ratio of positive pairs to total pairs
            include_orig_orig: Whether to include original-original pairs
            include_mask_mask: Whether to include mask-mask pairs
        """
        self.masked_image_dir = masked_image_dir
        self.chair_crops_dir = chair_crops_dir
        self.inpainted_masks_dir = inpainted_masks_dir

        self.image_file_names = image_file_names

        self.img_size = img_size
        self.positive_ratio = positive_ratio
        self.include_orig_orig = include_orig_orig
        self.include_mask_mask = include_mask_mask
        
        # Get file names
        if self.image_file_names is None:
            self.image_files = sorted([f for f in os.listdir(masked_image_dir) if f.endswith(('.jpg', '.png'))])
        else:
            self.image_files = self.image_file_names
        
        # Base transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Generate training pairs
        self._generate_pairs()
        
    def _generate_pairs(self):
        """Generate training pairs from available data."""
        pairs = []
        
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            
            # Paths for each component
            masked_image_path = os.path.join(self.masked_image_dir, img_file)  # Use original filename
            chair_crop_path = os.path.join(self.chair_crops_dir, f"{base_name}.png")
            inpainted_mask_path = os.path.join(self.inpainted_masks_dir, f"{base_name}.png")
            
            # Skip if files don't exist
            if not (os.path.exists(masked_image_path) and os.path.exists(chair_crop_path) and os.path.exists(inpainted_mask_path)):
                print(f"Skipping {img_file} - missing files:")
                if not os.path.exists(masked_image_path):
                    print(f"  Missing masked image: {masked_image_path}")
                if not os.path.exists(chair_crop_path):
                    print(f"  Missing chair crop: {chair_crop_path}")
                if not os.path.exists(inpainted_mask_path):
                    print(f"  Missing inpainted mask: {inpainted_mask_path}")
                continue
                
            # Add positive pairs
            # Original - Mask (main positive pair)
            pairs.append((masked_image_path, chair_crop_path, False))
            # Original - Original (with different augmentations)
            if self.include_orig_orig:
                pairs.append((masked_image_path, masked_image_path, False))
            
            # Mask - Mask (with different augmentations)
            if self.include_mask_mask:
                pairs.append((chair_crop_path, chair_crop_path, False))

            # Add negative pairs
            # Original - Inpainted
            pairs.append((masked_image_path, inpainted_mask_path, True))
            # Chair Crop - Inpainted
            pairs.append((chair_crop_path, inpainted_mask_path, True))

        # Shuffle pairs
        random.shuffle(pairs)
        self.pairs = pairs
        
        print(f"Generated {len(pairs)} pairs from {len(self.image_files)} images")
        if len(pairs) == 0:
            print("WARNING: No pairs were generated! Check if the file paths are correct:")
            print(f"Masked image directory: {self.masked_image_dir}")
            print(f"Chair crop directory: {self.chair_crops_dir}")
            print(f"Inpainted mask directory: {self.inpainted_masks_dir}")
        
    def __len__(self):
        return len(self.pairs)
    
    def _load_and_transform(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
            
        return img
    
    def __getitem__(self, idx):
        view1_path, view2_path, is_negative = self.pairs[idx]
        
        # Load and transform images
        view1 = self._load_and_transform(view1_path)
        view2 = self._load_and_transform(view2_path)
        
        # If it's a same-image pair but not negative, apply different augmentations
        if view1_path == view2_path and not is_negative:
            view1 = self._apply_random_augmentation(view1)
            view2 = self._apply_random_augmentation(view2)
            
        return view1, view2, torch.tensor(is_negative, dtype=torch.bool)
    
    def _apply_random_augmentation(self, img):
        """Apply random augmentation to create a different view."""
        # Convert tensor to PIL image for RandomResizedCrop
        img_pil = transforms.ToPILImage()(img)
        
        # Apply RandomResizedCrop
        random_crop = transforms.RandomResizedCrop(
            size=self.img_size,
            scale=(0.85, 1.0),  # Crop between 70% to 100% of original size
            ratio=(0.95, 1.05)  # Aspect ratio constraints
        )
        
        img_aug = random_crop(img_pil)
        
        # Convert back to tensor and normalize
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalize(to_tensor(img_aug))

import sys
import os
from torch.utils.data import DataLoader, random_split
import random

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def get_dataloaders(
    _image_list,
    train_size=0.8,
    val_size=0.2,
    batch_size=32,
    num_workers=4,
    subset=None,
):
    """Create training and validation dataloaders.
    
    Args:
        _image_list: List of image filenames to use
        train_size: Proportion of data to use for training (default: 0.8)
        val_size: Proportion of data to use for validation (default: 0.2)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        subset: Optional number of data points to use (if None, use all data)
    """
    
    # Define data directories
    masked_image_dir = config.MASKED_DATASET_PATH  
    chair_crops_dir = config.ORIGINAL_MASK_CROPS_DATASET_PATH
    inpainted_masks_dir = config.INPAINTED_MASKS_DATASET_PATH

 
    if subset is not None:
        print(f"Using subset of {subset} images from a total of {len(_image_list)} images")
        indices = random.sample(range(len(_image_list)), subset)
        new_image_list = [_image_list[i] for i in indices]
        _image_list = new_image_list
    
    dataset = SceneMaskDataset(masked_image_dir=masked_image_dir, 
                               chair_crops_dir=chair_crops_dir, 
                               inpainted_masks_dir=inpainted_masks_dir,
                               image_file_names=_image_list)
    
    # Calculate actual lengths for train and validation splits
    total_length = len(dataset)
    train_length = int(total_length * train_size)
    val_length = total_length - train_length  # Use remaining samples for validation
    
    print(f"Total dataset size: {total_length}")
    print(f"Training set size: {train_length}")
    print(f"Validation set size: {val_length}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    return train_loader, val_loader