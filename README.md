

# Research Question

Q: What is the best method for repainting an image in context? Is there an advantage for using a transformer to encode orientation opposed to CNNs?

Specific Q: What type of projector used during contrastive learning can be used to generate scene specific embeddings to create more stylistically accurate scenes?

How to run the code:

python byo-main-all.py --subset 4000 --test_subset 25 --byol_epochs 10 --gan_epochs 10  --byol_encoder_type "dual"

Review the byo-main-all.py for all the available flags to run the code. Use subset if your machine is not able to train on the full dataset.

Full dataset is available at: https://www.kaggle.com/datasets/ryanbergamini/inpainted-chairs-from-coco-2017/


This code was developed in Cursor - an AI integrated IDE environment. I used the claude-sonnet-3.7 model to co-develop the code. I also used the IOPaint repository from github (www.github.com/Sanster/IOPaint) to use the LMD model to generate painted infills to masked images.
