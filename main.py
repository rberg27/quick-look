from models.segmentation.seg import Segmenter
from utils.utils import display_segmentation, select_segment_by_index, create_dummy_segmentations
from config import TEST_IMAGE_PATH

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# The main delivery will display in a grid:
#  1. the original image
#  2. segmented image
#  3. selected segments
#  4. image without that segment
#  5. the edited segment
#  6. the final image

def display_results_grid(original_image, segmented_image, selected_segment, 
                         image_without_segment, edited_segment, final_image,
                         save_path=None):
    """
    Display all processing steps in a 3x2 grid.
    
    Args:
        original_image: The input image
        segmented_image: Image showing all segments
        selected_segment: Image highlighting the selected segment
        image_without_segment: Original image with selected segment removed
        edited_segment: The segment after editing
        final_image: The final composite image
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(segmented_image)
    axes[0, 1].set_title("2. Segmented Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(selected_segment)
    axes[0, 2].set_title("3. Selected Segment")
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(image_without_segment)
    axes[1, 0].set_title("4. Image without Segment")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edited_segment)
    axes[1, 1].set_title("5. Edited Segment")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(final_image)
    axes[1, 2].set_title("6. Final Image")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Results visualization saved to {save_path}")
    
    plt.show()
    
    return fig

# Placeholder functions for unimplemented parts
def remove_segment(image, mask_dict):
    """
    Remove the selected segment from the image.
    Args:
        image: Original image
        mask_dict: Mask dict (SAM format) of the segment to remove
    Returns:
        Image with the segment removed (with a placeholder hole)
    """
    mask = mask_dict['segmentation'] if isinstance(mask_dict, dict) else mask_dict
    result = image.copy()
    if len(result.shape) == 2:
        result = np.stack([result, result, result], axis=2)
    elif result.shape[2] == 1:
        result = np.concatenate([result, result, result], axis=2)
    result[mask > 0] = [128, 128, 128]
    return result

def edit_segment(image, mask_dict):
    """
    Edit the selected segment.
    Args:
        image: Original image
        mask_dict: Mask dict (SAM format) of the segment to edit
    Returns:
        The edited segment as an image
    """
    # Extract the binary mask from the mask dictionary if it's a dictionary,
    # otherwise use the mask directly
    mask = mask_dict['segmentation'] if isinstance(mask_dict, dict) else mask_dict
    
    # Create a copy of the original image to avoid modifying it
    segment = image.copy()
    
    # Ensure the image has 3 channels (RGB)
    # If it's grayscale (2D), convert it to RGB by stacking the same channel 3 times
    if len(segment.shape) == 2:
        segment = np.stack([segment, segment, segment], axis=2)
    # If it has only 1 channel but is 3D, concatenate to make it RGB
    elif segment.shape[2] == 1:
        segment = np.concatenate([segment, segment, segment], axis=2)
    
    # Create another copy to work with for the segment extraction
    segment_only = segment.copy()
    
    # Set all pixels outside the mask (where mask == 0) to black [0, 0, 0]
    segment_only[mask == 0] = [0, 0, 0]
    
    # For pixels inside the mask (where mask > 0), apply a blue tint:
    # - Multiply red channel by 0.5 (reduce red)
    # - Multiply green channel by 0.5 (reduce green)
    # - Multiply blue channel by 1.0 (keep blue unchanged)
    # This effectively turns the segment blue-ish
    # Convert back to uint8 to maintain proper image format
    segment_only[mask > 0] = (segment_only[mask > 0] * [1.0, 0.5, 0.5]).astype(np.uint8)
    
    # Return the edited segment (blue-tinted object with black background)
    return segment_only

def composite_final_image(image_without_segment, edited_segment, mask_dict):
    """
    Composite the edited segment back into the image.
    Args:
        image_without_segment: Image with the segment removed
        edited_segment: The edited segment
        mask_dict: Mask dict (SAM format) of the segment
    Returns:
        Final composite image
    """
    mask = mask_dict['segmentation'] if isinstance(mask_dict, dict) else mask_dict
    result = image_without_segment.copy()
    result[mask > 0] = edited_segment[mask > 0]
    return result


if __name__ == "__main__":
    # Import necessary modules
    import cv2
    from config import TEST_IMAGE_PATH
    from models.segmentation.seg import Segmenter
    from utils.utils import display_segmentation, select_segment_by_index
    
    # Initialize the segmenter
    segmenter = Segmenter()
    
    # Load and segment the image
    image = cv2.imread(TEST_IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Get segmentation masks
    masks = segmenter.segment(image)
    print(f"Masks: {masks}")
    #masks = create_dummy_segmentations(image.shape)
    #dummy_masks = masks
    #print(f"Dummy masks: {dummy_masks}")

    # Display all segmentation masks
    segmented_image = display_segmentation(image, masks, show=False, show_segment_numbers=True)
    
    # Select a specific segment (using the first one for this example)
    segment_index = [21,29,56,46,49]
    selected_mask = select_segment_by_index(masks, segment_index)
    
    # Create an image showing just the selected segment
    selected_segment_image = display_segmentation(image, [selected_mask], show=False)
    
    # Remove the segment from the image
    image_without_segment = remove_segment(image, selected_mask)
    
    # Edit the segment
    edited_segment_image = edit_segment(image, selected_mask)
    
    # Composite the final image
    final_image = composite_final_image(image_without_segment, edited_segment_image, selected_mask)
    

    print(f"Segmented image shape: {segmented_image.shape} type: {type(segmented_image)}")
    print(f"Selected mask type: {type(selected_mask)}, keys: {list(selected_mask.keys()) if isinstance(selected_mask, dict) else 'N/A'}")
    print(f"Selected mask segmentation shape: {selected_mask['segmentation'].shape if isinstance(selected_mask, dict) else selected_mask.shape}")
    print(f"Image without segment shape: {image_without_segment.shape} type: {type(image_without_segment)}")
    print(f"Edited segment shape: {edited_segment_image.shape} type: {type(edited_segment_image)}")
    print(f"Final image shape: {final_image.shape} type: {type(final_image)}")

    # Display the results grid
    display_results_grid(
        original_image=image,
        segmented_image=segmented_image,
        selected_segment=selected_segment_image,
        image_without_segment=image_without_segment,
        edited_segment=edited_segment_image,
        final_image=final_image,
        save_path="segmentation_results.png"
    )
    
