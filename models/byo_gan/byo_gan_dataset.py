# Dataset and data loading
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
import numpy as np

class BinarizeTransform:
    """
    Transform that converts image pixels to binary (0 or 1) based on a threshold.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, img):
        """
        Args:
            img: Tensor image of size (C, H, W)
        Returns:
            Tensor: Binarized image (0 or 1 values)
        """
        return (img > self.threshold).float()

class BYOGANDataset(Dataset):
    """
        Dataset for training the BYO-GAN model.
        Each entry will have the mask, the masked image, and the original image.

        The generator is provided the mask and the masked image

        The discriminator is provided the generated image and mask for detecting fakes, and then any image and a mask for detecting real images

    """
    def __init__(
        self,
        masked_image_dir,
        mask_dir,
        original_images_dir,
        img_size=224,
        subset=None,
        image_file_names=None
    ):
        """Initialize the dataset.
        
        Args:   
            masked_image_dir: Directory with masked images
            mask_dir: Directory with original masks
            original_images_dir: Directory with original images
            transform: Optional additional transforms
            img_size: Size to resize images
        """
        self.masked_image_dir = masked_image_dir
        self.mask_dir = mask_dir
        self.original_image_dir = original_images_dir
        self.img_size = img_size
        self.subset = subset
        
        # Get file names
        if image_file_names is None:
            self.image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.png'))])
        else:
            self.image_files = image_file_names
        
        if self.subset is not None:
            # Randomly sample subset number of files instead of taking the first ones
            self.image_files = random.sample(self.image_files, self.subset)

        # Base transformations for images
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure tuple of integers
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformations for masks (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            BinarizeTransform(threshold=0.5)
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def _load_and_transform(self, path, is_mask=False):
        #print(f"Loading image from {path}")
        img = Image.open(path).convert('RGB')
        # print(f"Image loaded: {img}")
        # print(f"Type of image: {type(img)}")
        # Apply appropriate transform based on whether it's a mask or not
        if is_mask:
            img = transforms.ToTensor()(img)
            img = self.mask_transform(img)
        else:
            img = self.image_transform(img)
            
            
        return img
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]

        # Load and transform images
        masked_image = self._load_and_transform(os.path.join(self.masked_image_dir, file_name), is_mask=False)
        mask = self._load_and_transform(os.path.join(self.mask_dir, file_name), is_mask=True)
        original_image = self._load_and_transform(os.path.join(self.original_image_dir, file_name), is_mask=False)
            
        return masked_image, mask, original_image
    

def get_dataloaders(
    masked_image_dir,
    mask_dir,
    original_images_dir,
    val_split=0.1,
    batch_size=32,
    num_workers=4,
    img_size=224,
    subset=None
):
    """Create training and validation dataloaders.
    
    Args:
        masked_image_dir: Directory with masked images
        mask_dir: Directory with original masks
        original_images_dir: Directory with original images
        val_split: Fraction of data to use for validation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        img_size: Size to resize images
        subset: Optional number of data points to use (if None, use all data)
    """
    
    # Create dataset
    dataset = BYOGANDataset(
        masked_image_dir=masked_image_dir,
        mask_dir=mask_dir,
        original_images_dir=original_images_dir,
        img_size=img_size,
        subset=subset
    )
    
    # Apply subset if specified
    if subset is not None and subset < len(dataset):
        # Get a random subset of the data
        indices = random.sample(range(len(dataset)), subset)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split into train and validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
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

    train_loader.persistent_workers = True
    val_loader.persistent_workers = True
    
    return train_loader, val_loader