

# Research Question

Q: What is the best method for repainting an image in context? Is there an advantage for using a transformer to encode orientation opposed to CNNs?


# Project Deliverable

In: An Image of a Scene, An Image of an Object

Out: That Object replaced into the scene


## Sub-Modules
* Segmentation
    - In: An Image of a Scene
    - Out: The Scene Broken Into Segments

* Segment Selection:
    - In: Image with Segments
    - Out: Selection of a segment that will be edited

* Segment Edit:
    - In: Image, Selection Image, New Image
    - Out: Updated Image



# TODO
* Implementation
1. Complete seg.py to take in a photo and then output segmentations - DONE
    - Draft paper to identify what models to substitute and test
2. Collect user input to select a segmentation model
3. Remove the selection from the image
4. Fill in the space
    * GANPaint code is not available
    - Model: GAN Model
        - IN: Image with a masked area
        - OUT: A complete Image
    - Data: P
    - Training Process:
        * Loss Function: Similarity with the rest of the Image + If discriminated or not
        * 
5. Edit the image
    a. Turn the image red - DONE
    b. Swap the pattern of the object
    c. Remove the object
6. Add the object back to the image
7. Any final realism processing


# Problems and Candidate Solutions
- Problem: When you replace the chair, how do you deal with the shading being different?
    * A: Train a GAN to take in an image, using attention, to fix the improper shading and overlap

# quick-look
