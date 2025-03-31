""" HOW TO USE:
from online_augmentation import augment_image
from PIL import Image

# Load an image
image = Image.open('path/to/your/image.jpg').convert('RGB')

# Generate augmented images (default is 5)
augmented_images = augment_image(image, num_examples=5, final_resolution=(224, 224))
"""

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def set_seed():
    """Set the random seed to a fixed value (42) for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_augmentation_pipeline(final_resolution=(224, 224)):
    """
    Create and return an augmentation pipeline.
    
    The pipeline performs:
    - Random resized crop to the final resolution
    - Random rotation between -20° and 20°
    - Conversion to a tensor
    """
    pipeline = transforms.Compose([
        transforms.RandomResizedCrop(final_resolution, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.ToTensor(),
    ])
    return pipeline

def augment_image(image: Image.Image, num_examples: int = 5, final_resolution: tuple = (224, 224)):
    """
    Apply online augmentation to a single PIL image using a fixed random seed.
    
    Args:
        image (PIL.Image.Image): The input image.
        num_examples (int): The number of augmented images to generate.
        final_resolution (tuple): Desired final resolution (width, height) for output images.
        
    Returns:
        List[torch.Tensor]: A list of augmented image tensors.
    """
    # Set the random seed for reproducibility (fixed to 42)
    set_seed()
    pipeline = get_augmentation_pipeline(final_resolution)
    return [pipeline(image) for _ in range(num_examples)]

# Example usage when running this file directly:
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Update the path to an example image
    example_image_path = "aligned/7153718@N04/landmark_aligned_face.2282.11597935265_29bcdfa4a5_o.jpg"
    image = Image.open(example_image_path).convert("RGB")
    
    # Generate 5 augmented versions of the image
    augmented_images = augment_image(image, num_examples=5, final_resolution=(224, 224))
    
    # Visualize the augmented images
    fig, axes = plt.subplots(1, len(augmented_images), figsize=(6 * len(augmented_images), 6))
    for ax, aug_img in zip(axes, augmented_images):
        ax.imshow(aug_img.permute(1, 2, 0))  # Convert tensor from C x H x W to H x W x C for display
        ax.axis('off')
    plt.tight_layout()
    plt.show()