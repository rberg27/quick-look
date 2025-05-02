import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        # Use HSV color space for more distinct colors
        hue = i / n
        # High saturation and value for vibrant colors
        color = plt.cm.hsv(hue)
        # Add some transparency
        color = list(color)
        color[3] = 0.6  # Set alpha for transparency
        colors.append(color)
    return colors

def show_mask(mask, ax, color=None):
    """Helper function to display a mask on a given matplotlib axis."""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Helper function to display points on a given matplotlib axis."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def test_sam_model():
    """
    Test the Segment Anything Model (SAM) checkpoint saved in the segmentation/SAM folder.
    This function loads the model, runs inference on a test image, and visualizes the results.
    """
    # Define paths
    checkpoint_path = "/Users/ryanbergmac/Desktop/Master Plan/AI-Masters/SP25 - Deep Learning With Transformers/code/quick-look/models/segmentation/SAM/sam_vit_b_01ec64.pth"
    model_type = "vit_b"  # Options: vit_h, vit_l, vit_b
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: SAM checkpoint not found at {checkpoint_path}")
        return
    
    # Load the model
    print(f"Loading SAM model from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    
    # Load a test image
    test_image_path = "data/chair-in-living-room-easy.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Using a random image for testing.")
        # Create a random test image if none exists
        random_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_image = random_image
    else:
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # 1. Point-based segmentation
    predictor = SamPredictor(sam)
    predictor.set_image(test_image)
    
    print("Generating point-based mask...")
    # Provide input points for specific segmentation
    input_point = np.array([[test_image.shape[1] // 2, test_image.shape[0] // 2]])
    input_label = np.array([1])  # 1 indicates a foreground point
    
    # Predict masks based on input points
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # Return multiple masks
    )
    
    # 2. Automatic segmentation of all objects
    print("Generating automatic masks for all objects...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks_auto = mask_generator.generate(test_image)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.axis('off')
    plt.title("Original Image")
    
    # Point-based segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(test_image)
    best_mask_idx = np.argmax(scores)
    show_mask(masks[best_mask_idx], plt.gca(), color=[30/255, 144/255, 255/255, 0.6])
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.title(f"Point-based Segmentation\n(Score: {scores[best_mask_idx]:.3f})")
    
    # Automatic segmentation with distinct colors
    plt.subplot(1, 3, 3)
    plt.imshow(test_image)
    colors = generate_distinct_colors(len(masks_auto))
    for mask, color in zip(masks_auto, colors):
        show_mask(mask['segmentation'], plt.gca(), color=color)
    plt.axis('off')
    plt.title(f"Automatic Segmentation\n({len(masks_auto)} objects found)")
    
    plt.tight_layout()
    plt.savefig("sam_results.png", bbox_inches='tight', dpi=150)
    print("SAM model test completed. Results saved as sam_results.png")

if __name__ == "__main__":
    test_sam_model()
