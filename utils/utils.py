import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# Helper to extract mask array from dict or array
def _get_mask_array(mask):
    if isinstance(mask, dict) and 'segmentation' in mask:
        return mask['segmentation']
    return mask

def display_segmentation(image, masks, alpha=0.5, show=True, show_segment_numbers=False):
    """
    Display an image with colored segmentation masks overlaid.
    
    Args:
        image: The original image (numpy array)
        masks: List of mask dicts (SAM format) or binary masks (for backward compatibility)
        alpha: Transparency of the overlay (0-1)
        show: Whether to display the image immediately
        show_segmentation: Whether to display segment numbers next to each segment
        
    Returns:
        The visualization image with colored segments
    """
    # Create a copy of the image to avoid modifying the original
    visualization = image.copy()
    
    # Ensure image has 3 channels (RGB)
    if len(visualization.shape) == 2:
        visualization = np.stack([visualization, visualization, visualization], axis=2)
    elif visualization.shape[2] == 1:
        visualization = np.concatenate([visualization, visualization, visualization], axis=2)
    
    # Generate random colors for each mask
    colors = []
    for _ in range(len(masks)):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    
    # Apply each mask with a different color
    for i, mask in enumerate(masks):
        mask_arr = _get_mask_array(mask)
        color_mask = np.zeros_like(visualization)
        color_mask[mask_arr > 0] = colors[i]
        
        # Blend the color mask with the original image
        visualization = cv2.addWeighted(visualization, 1, color_mask, alpha, 0)
        
        # Add a border around the segment for better visibility
        contours, _ = cv2.findContours(mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, colors[i], 2)
        
        # Add segment number if requested
        if show_segment_numbers:
            # Find centroid of the mask to place the number
            M = cv2.moments(mask_arr.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Put segment number text with white color and black outline for visibility
                cv2.putText(visualization, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(visualization, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display the result if requested
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(visualization)
        plt.axis('off')
        plt.title(f"Segmentation with {len(masks)} segments")
        plt.show()
    
    return visualization

def _get_mask_array(mask):
    """
    Helper function to extract the binary mask array from a mask dict or return the mask if it's already an array.
    
    Args:
        mask: Mask dict (SAM format) or binary mask array
        
    Returns:
        Binary mask array
    """
    if isinstance(mask, dict) and 'segmentation' in mask:
        return mask['segmentation']
    return mask

def _combine_masks(masks):
    """
    Combine multiple mask dicts into a single mask dict.
    
    Args:
        masks: List of mask dicts (SAM format)
        
    Returns:
        A single mask dict with combined segmentation
    """
    if not masks:
        return None
    
    # Use the first mask as a template for the combined mask
    combined_mask = masks[0].copy()
    
    # Start with the first mask's segmentation
    combined_segmentation = _get_mask_array(masks[0]).copy()
    
    # Combine with the rest of the masks using logical OR
    for mask in masks[1:]:
        mask_array = _get_mask_array(mask)
        combined_segmentation = np.logical_or(combined_segmentation, mask_array)
    
    # Update the segmentation in the combined mask
    combined_mask['segmentation'] = combined_segmentation
    
    return combined_mask

def select_segment_by_index(masks, index):
    """
    Select a specific segment or multiple segments by index.
    
    Args:
        masks: List of mask dicts (SAM format)
        index: Index of the segment to select, or list/tuple of indices
        
    Returns:
        The selected mask dict (combined if multiple indices are provided)
    """
    if isinstance(index, (list, tuple)):
        # Handle multiple indices
        selected_masks = []
        for idx in index:
            if idx < 0 or idx >= len(masks):
                raise ValueError(f"Index {idx} out of range. There are {len(masks)} segments.")
            selected_masks.append(masks[idx])
        
        # Combine the selected masks into a single mask
        return _combine_masks(selected_masks)
    else:
        # Handle single index
        if index < 0 or index >= len(masks):
            raise ValueError(f"Index {index} out of range. There are {len(masks)} segments.")
        return masks[index]

def create_dummy_segmentations(image_shape, num_segments=3):
    """
    Create dummy segmentation masks for testing without loading the SAM model.
    
    Args:
        image_shape: Tuple of (height, width) or (height, width, channels)
        num_segments: Number of dummy segments to create
        
    Returns:
        List of mask dicts in SAM format
    """
    height, width = image_shape[:2]
    masks = []
    
    # Create simple geometric shapes as segments
    for i in range(num_segments):
        mask = np.zeros((height, width), dtype=bool)
        
        # Segment 0: Circle in top-left
        if i == 0:
            center_x, center_y = width // 4, height // 4
            radius = min(width, height) // 6
            for y in range(height):
                for x in range(width):
                    if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                        mask[y, x] = True
        
        # Segment 1: Rectangle in bottom-right
        elif i == 1:
            x1, y1 = width // 2, height // 2
            x2, y2 = int(width * 0.9), int(height * 0.9)
            mask[y1:y2, x1:x2] = True
        
        # Segment 2: Triangle in top-right
        elif i == 2:
            points = np.array([
                [int(width * 0.7), int(height * 0.1)],
                [int(width * 0.9), int(height * 0.4)],
                [int(width * 0.5), int(height * 0.4)]
            ])
            cv2.fillPoly(mask.astype(np.uint8), [points], 1)
            mask = mask.astype(bool)
        
        # Additional segments if needed
        else:
            # Random blob
            center_x = np.random.randint(width // 4, 3 * width // 4)
            center_y = np.random.randint(height // 4, 3 * height // 4)
            radius = np.random.randint(min(width, height) // 10, min(width, height) // 5)
            
            for y in range(height):
                for x in range(width):
                    dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                    if dist < radius:
                        mask[y, x] = True
        
        mask_dict = {
            'segmentation': mask,
            'area': int(np.sum(mask)),
            'bbox': [0, 0, width, height],
            'predicted_iou': 1.0,
            'point_coords': [[width // 2, height // 2]],
            'stability_score': 1.0,
            'crop_box': [0, 0, width, height]
        }
        masks.append(mask_dict)
    
    return masks

def test_segmentation_utils():
    """
    Test the segmentation utility functions using dummy data.
    This function can be used to verify functionality without loading the SAM model.
    """
    # Create a dummy image
    height, width = 400, 600
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_image[:, :, 0] = np.linspace(0, 255, width)
    dummy_image[:, :, 1] = np.linspace(0, 255, height)[:, np.newaxis]
    dummy_image[:, :, 2] = 255
    
    # Create dummy segmentation masks
    dummy_masks = create_dummy_segmentations((height, width), num_segments=3)
    
    print(f"Created {len(dummy_masks)} dummy segmentation masks")
    
    # Test display_segmentation
    print("Testing display_segmentation...")
    segmented_image = display_segmentation(dummy_image, dummy_masks, show=False)
    print(f"Segmented image shape: {segmented_image.shape}")
    
    # Test select_segment_by_index
    print("Testing select_segment_by_index...")
    for i in range(len(dummy_masks)):
        selected_mask = select_segment_by_index(dummy_masks, i)
        print(f"Selected mask {i} shape: {selected_mask['segmentation'].shape}")
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(dummy_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(segmented_image)
    plt.title("All Segments")
    plt.axis('off')
    
    for i in range(min(2, len(dummy_masks))):
        plt.subplot(1, 4, i+3)
        selected_mask = select_segment_by_index(dummy_masks, i)
        selected_viz = display_segmentation(dummy_image, [selected_mask], show=False)
        plt.imshow(selected_viz)
        plt.title(f"Segment {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return dummy_image, dummy_masks, segmented_image

if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_segmentation_utils()
