# * Segmentation
#    - In: An Image of a Scene
#    - Out: The Scene Broken Into Segments

import os
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class Segmenter:
    def __init__(self, checkpoint_path=None, model_type="vit_b"):
        """
        Initialize the Segmenter with a SAM model.
        
        Args:
            checkpoint_path (str): Path to the SAM model checkpoint
            model_type (str): Type of SAM model ('vit_h', 'vit_l', 'vit_b')
        """
        if checkpoint_path is None:
            # Default path if none provided
            checkpoint_path = "models/segmentation/SAM/sam_vit_b_01ec64.pth"
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
        
        # Load the model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam.to(self.device)
        
        # Initialize predictor and mask generator
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def segment(self, image, method="automatic"):
        """
        Segment the input image.
        
        Args:
            image (numpy.ndarray): RGB image to segment
            method (str): Segmentation method - 'automatic' or 'point'
            
        Returns:
            list or dict: Segmentation masks and metadata
        """
        if method == "automatic":
            # Generate masks for all objects in the image
            return self.mask_generator.generate(image)
        elif method == "point":
            # For point-based segmentation, we'll use the center point as default
            self.predictor.set_image(image)
            input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
            input_label = np.array([1])  # 1 indicates a foreground point
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            return {"masks": masks, "scores": scores, "logits": logits}
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
