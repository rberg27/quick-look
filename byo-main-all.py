import models.byol.byol as byol
import models.byo_gan.byo_gan as byo_gan
import models.byo_gan.stencil_gan as stencil_gan
import argparse
import config
import os

from PIL import Image
import torch
import numpy as np

import models.byol.contrastive_dataset as contrastive_dataset
import models.byo_gan.byo_gan_dataset as byo_gan_dataset

def main():
    """Main function to run the training pipeline."""
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high') 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BYOL and GAN models')
    parser.add_argument('--subset', type=int, default=None, help='Number of samples to use for training (default: use all)')
    parser.add_argument('--test_subset', type=int, default=None, help='Number of samples to use for testing (default: use all)')
    parser.add_argument('--byol_epochs', type=int, default=10, help='Number of epochs to train the BYOL model')
    parser.add_argument('--load_weights', action='store_true', help='Load BYOL model weights from previous training')
    parser.add_argument('--load_weights_for_gan', action='store_true', help='Load BYO_GAN model weights from previous training')
    parser.add_argument('--weights_dir', type=str, default=config.BYOL_MODEL_WEIGHTS_PATH, help='Directory containing saved model weights')
    parser.add_argument('--byol_encoder_type', type=str, default='dual', help='BYOL encoder type (default: dual)')
    parser.add_argument('--gan_epochs', type=int, default=10, help='Number of epochs to train the GAN model')
    parser.add_argument('--gan_batch_size', type=int, default=16, help='Batch size for the GAN model')
    parser.add_argument('--gan_model_name', type=str, default='final.pt', help='Name of the GAN model file')
    parser.add_argument('--skip_embedding_test', action='store_true', help='Skip embedding test')
    parser.add_argument('--output_dir', type=str, default='./byo_gan_results', help='Output directory')
    parser.add_argument('--load_weights_for_stencil_gan', action='store_true', help='Load Stencil GAN model weights from previous training')
    parser.add_argument('--loaded_stencil_gan_model_name', type=str, default='final.pt', help='Name of the Stencil GAN model file')

    args = parser.parse_args()
    
    # Define data directories
    masked_image_dir = config.MASKED_DATASET_PATH  
    chair_crops_dir = config.ORIGINAL_MASK_CROPS_DATASET_PATH
    inpainted_masks_dir = config.INPAINTED_MASKS_DATASET_PATH
    masks_dir = config.MASKS_DATASET_PATH
    original_images_dir = config.ORIGINAL_DATASET_PATH
    
    # Create File Name list
    from sklearn.model_selection import train_test_split
    all_names = os.listdir(original_images_dir)
    image_file_names = [name for name in all_names if name.endswith(('.jpg', '.png'))]
    

    train_file_names, test_file_names = train_test_split(image_file_names, test_size=0.2, random_state=42)

    print(f"Using {len(train_file_names)} training images and {len(test_file_names)} test images")
    # Create Contrastive Learning Dataset
    test_byol_dataset = contrastive_dataset.SceneMaskDataset(masked_image_dir=masked_image_dir, 
                                                             chair_crops_dir=chair_crops_dir, 
                                                             inpainted_masks_dir=inpainted_masks_dir, 
                                                             image_file_names=test_file_names)

    print("Number of test_byol_dataset: ", len(test_byol_dataset))

    # Step 1: Train the BYOL model
    weights_dir = args.weights_dir
    if args.byol_encoder_type:
        weights_dir = os.path.join(args.weights_dir, args.byol_encoder_type)
    

    print("Training BYOL model...")
    byol_model = byol.train_model(
        train_file_names,
        output_dir=config.BYOL_CHECKPOINT_PATH,
        max_epochs=args.byol_epochs,
        batch_size=64,
        encoder_type=args.byol_encoder_type,  # Use both CNN and Transformer
        negative_weight=0.5,
        temperature=0.1,
        subset=args.subset,
        load_weights=args.load_weights,
        weights_dir=weights_dir
    )
    
    # Step 2: Test the embeddings
    if args.skip_embedding_test:
        print("Skipping embedding test...")
    else:
        print("Testing embeddings...")
        results, stats = byol.test_embeddings(
            model=byol_model,
            _dataset=test_byol_dataset,
            output_dir='./embedding_tests',
            num_samples=args.test_subset
        )
    
    # Create Embedding Guided GAN Dataset
    train_gan_dataset = byo_gan_dataset.BYOGANDataset(
        masked_image_dir=masked_image_dir,
        mask_dir=masks_dir,
        original_images_dir=original_images_dir,
        image_file_names=train_file_names
    )
    
    test_gan_dataset = byo_gan_dataset.BYOGANDataset(
        masked_image_dir=masked_image_dir,
        mask_dir=masks_dir,
        original_images_dir=original_images_dir,
        image_file_names=test_file_names
    )
    # Step 3: Train the GAN model
    print("Training BYO GAN...")
    gan_model = byo_gan.train_embedding_guided_gan(
        byol_model=byol_model,
        _dataset=train_gan_dataset,
        output_dir=config.BYO_GAN_CHECKPOINT_PATH,
        max_epochs=args.gan_epochs,
        batch_size=args.gan_batch_size,
        subset=args.subset,
        load_weights=args.load_weights_for_gan,
        loaded_model_name=args.gan_model_name,
        save_weights=True
    )

    # Step 3: Train the GAN model
    print("Training Stencil GAN...")
    stencil_gan_model = stencil_gan.train_gan(
        _dataset=train_gan_dataset,
        output_dir="./stencil_gan_results",
        weights_dir=os.path.join(args.output_dir, 'weights'),
        max_epochs=args.gan_epochs,
        lambda_real=0.18,
        lambda_reconstruction=0.82,
        use_reconstruction_loss=True,
        batch_size=args.gan_batch_size,
        load_weights=args.load_weights_for_stencil_gan,
        loaded_model_name=args.loaded_stencil_gan_model_name,
        save_weights=True,
        subset=args.subset
    )
    
    print("Training complete!")
    
    # Step 4: Test the GAN model
    print("Testing GAN...")
    # Generate and save sample results
    print("Generating sample results...")
    test_gan_output_dir = os.path.join("./byo_gan_results", 'test_samples')
    test_stencil_gan_output_dir = os.path.join("./stencil_gan_results", 'test_samples')
    print(f"Creating test output directory at: {test_gan_output_dir}")
    os.makedirs(test_gan_output_dir, exist_ok=True)
    print(f"Creating test output directory at: {test_stencil_gan_output_dir}")
    os.makedirs(test_stencil_gan_output_dir, exist_ok=True)

    for i in range(args.test_subset):
        masked_image, mask, original_image = test_gan_dataset[i]
        byo_gan.test_embedding_gan(
            gan_model,
            masked_image,
            mask,
            original_image,
            test_gan_output_dir,
            filename=f"byo_gan_sample_{i}_comparison.jpg"
        )
        stencil_gan.test_gan(
            stencil_gan_model,
            masked_image,
            mask,
            original_image,
            test_stencil_gan_output_dir,
            filename=f"stencil_gan_sample_{i}_comparison.jpg"
        )
    print(f"Test file names: {test_file_names}")
    print(f"Process complete! Results saved to {args.output_dir}")
    print(f"Test samples directory contents: {os.listdir(test_gan_output_dir)}")
    print(f"Test stencil samples directory contents: {os.listdir(test_stencil_gan_output_dir)}")
    
   
    print("Testing complete!")
    
if __name__ == '__main__':
    main()