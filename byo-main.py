import models.byol.byol as byol
import models.byo_gan.byo_gan as byo_gan
import argparse
import config
import os

from PIL import Image

import torch


def main():
    """Main function to run the training pipeline."""
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high') 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BYOL and GAN models')
    parser.add_argument('--subset', type=int, default=None, help='Number of samples to use for training (default: use all)')
    parser.add_argument('--test_subset', type=int, default=None, help='Number of samples to use for testing (default: use all)')
    parser.add_argument('--load_weights', action='store_true', help='Load BYOL model weights from previous training')
    parser.add_argument('--load_weights_for_gan', action='store_true', help='Load BYO_GAN model weights from previous training')
    parser.add_argument('--weights_dir', type=str, default=config.BYOL_MODEL_WEIGHTS_PATH, help='Directory containing saved model weights')
    parser.add_argument('--byol_encoder_type', type=str, default='dual', help='BYOL encoder type (default: dual)')
    args = parser.parse_args()
    
    # Define data directories
    image_dir = config.MASKED_DATASET_PATH  
    mask_dir = config.ORIGINAL_MASK_CROPS_DATASET_PATH
    inpaint_dir = config.INPAINTED_MASKS_DATASET_PATH
    
    # Step 1: Train the BYOL model
    weights_dir = args.weights_dir
    if args.byol_encoder_type:
        weights_dir = os.path.join(args.weights_dir, args.byol_encoder_type)
    

    print("Training BYOL model...")
    byol_model = byol.train_model(
        image_dir=image_dir,
        mask_dir=mask_dir,
        inpaint_dir=inpaint_dir,
        output_dir=config.BYOL_CHECKPOINT_PATH,
        max_epochs=100,
        batch_size=1024,
        encoder_type=args.byol_encoder_type,  # Use both CNN and Transformer
        negative_weight=0.5,
        temperature=0.1,
        subset=args.subset,
        load_weights=args.load_weights,
        weights_dir=weights_dir
    )
    
    # Step 2: Test the embeddings
    print("Testing embeddings...")
    results, stats = byol.test_embeddings(
        model=byol_model,
        test_image_dir=image_dir,
        test_mask_dir=mask_dir,
        test_inpaint_dir=inpaint_dir,
        output_dir='./embedding_tests',
        num_samples=args.test_subset
    )
    
    # Step 3: Train the GAN model
    print("Training GAN...")
    gan_model = byo_gan.train_gan(
        byol_model=byol_model,
        image_dir=image_dir,
        mask_dir=mask_dir,
        inpaint_dir=inpaint_dir,
        output_dir=config.BYO_GAN_CHECKPOINT_PATH,
        max_epochs=100,
        batch_size=64,
        lambda_embedding=1.0,
        lambda_reconstruction=0.1,
        use_reconstruction_loss=True,
        subset=args.subset,
        load_weights=args.load_weights_for_gan,
        weights_path=config.DEFAULT_BYO_GAN_MODEL_WEIGHTS_PATH,
        weights_dir=config.BYO_GAN_MODEL_WEIGHTS_DIR
    )
    
    print("Training complete!")
    
    # Step 4: Test the GAN model
    print("Testing GAN...")
    masked_image_path = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset\processed\masked\000000001138.png"
    original_image_path = r"C:\Users\rberg\Documents\quick-look\data\filtered_dataset\processed\originals\000000001138.png"

    masked_image = Image.open(masked_image_path)
    original_image = Image.open(original_image_path)
    # Convert images to tensors for test_gan method
    import torchvision.transforms as transforms
    transform = transforms.PILToTensor()
    
    masked_tensor = transform(masked_image)
    original_tensor = transform(original_image)
    byo_gan.test_gan(
        gan_model=gan_model,
        masked_image=masked_image,
        original_image=original_image,
        output_dir="./",
        filename="gan_result.png"
    )
    
    print("Testing complete!")
    
if __name__ == '__main__':
    main()