import os

# Define the path to the test image
TEST_IMAGE_PATH = r"C:\Users\rberg\Documents\quick-look\quick-look\data\chair-in-living-room-easy.jpg"

COCO_ANNOTATIONS_PATH = r"C:\Users\rberg\Documents\quick-look\data\annotations_trainval2017\annotations"
COCO_IMAGES_PATH = r"C:\Users\rberg\Documents\quick-look\data\train2017\train2017"

# Define the path to the filtered dataset
FILTERED_DATASET_PATH = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset"
FILTERED_ANNOTATIONS_PATH = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset\chair_annotations.json"
FILTERED_IMAGES_PATH = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset\images"

SAM_CHECKPOINT_PATH = r"C:\Users\rberg\Documents\quick-look\models\segmentation\SAM\sam_vit_h_4b8939.pth"

# Processed File Paths
PROCESSED_DATASET_PATH = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset\processed"
ORIGINAL_DATASET_PATH = PROCESSED_DATASET_PATH + "\originals"
MASKED_DATASET_PATH = PROCESSED_DATASET_PATH + "\masked"
INPAINTED_DATASET_PATH = PROCESSED_DATASET_PATH + "\inpainted"
MASKS_DATASET_PATH = PROCESSED_DATASET_PATH + "\masks"
INPAINTED_MASKS_DATASET_PATH = PROCESSED_DATASET_PATH + "\inpainted_masks"
ORIGINAL_MASK_CROPS_DATASET_PATH = PROCESSED_DATASET_PATH + "\chair_crops"


# BYOL and BYO_GAN Paths
BYOL_MODEL_WEIGHTS_PATH = r"C:\Users\rberg\Documents\quick-look\models\byol\model_weights"
BYOL_DUAL_MODEL_WEIGHTS_PATH = r"C:\Users\rberg\Documents\quick-look\models\byol\model_weights\dual"
BYOL_CNN_MODEL_WEIGHTS_PATH = r"C:\Users\rberg\Documents\quick-look\models\byol\model_weights\cnn"
BYOL_TRANSFORMER_MODEL_WEIGHTS_PATH = r"C:\Users\rberg\Documents\quick-look\models\byol\model_weights\transformer"

STENCIL_GAN_MODEL_WEIGHTS_DIR = r"C:\Users\rberg\Documents\quick-look\models\stencil_gan\model_weights"


BYOL_CHECKPOINT_PATH = r"C:\Users\rberg\Documents\quick-look\models\byol\checkpoints"
BYO_GAN_MODEL_WEIGHTS_DIR = r"C:\Users\rberg\Documents\quick-look\models\byo_gan\gan_weights"
DEFAULT_BYO_GAN_MODEL_WEIGHTS_PATH = BYO_GAN_MODEL_WEIGHTS_DIR + "\final.pt"
BYO_GAN_CHECKPOINT_PATH = r"C:\Users\rberg\Documents\quick-look\models\byo_gan\checkpoints"

STENCIL_GAN_CHECKPOINT_PATH = r"C:\Users\rberg\Documents\quick-look\models\stencil_gan\checkpoints"
STENCIL_GAN_WEIGHTS_PATH = r"C:\Users\rberg\Documents\quick-look\models\stencil_gan\weights"
