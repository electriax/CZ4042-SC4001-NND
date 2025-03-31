import os
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

class ImageAugmentation:
    def __init__(self,
                #  horizontal_flip_prob=0.5,
                #  color_jitter_prob=0.5,
                #  zoom_prob=0.5,
                #  shift_prob=0.5,
                #  zoom_scale=(0.8, 1.2),
                #  shift_range=(-0.1, 0.1),
                 resize_resolution=(112, 92)):  # Add resize resolution
        # self.horizontal_flip_prob = horizontal_flip_prob
        # self.color_jitter_prob = color_jitter_prob
        # self.zoom_prob = zoom_prob
        # self.shift_prob = shift_prob
        # self.zoom_scale = zoom_scale
        # self.shift_range = shift_range
        self.resize_resolution = resize_resolution  # Store resize resolution

    def __call__(self, image):
        # Apply horizontal flip
        # if random.random() < self.horizontal_flip_prob:
        #     image = F.hflip(image)

        # # Apply color jitter
        # if random.random() < self.color_jitter_prob:
        #     color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        #     image = color_jitter(image)

        # # Apply zoom using affine transform
        # if random.random() < self.zoom_prob:
        #     zoom_factor = random.uniform(self.zoom_scale[0], self.zoom_scale[1])
        #     image = F.affine(image, angle=0, translate=(0, 0), scale=zoom_factor, shear=0)

        # # Apply shift using affine transform
        # if random.random() < self.shift_prob:
        #     width, height = image.size
        #     shift_x = random.uniform(self.shift_range[0], self.shift_range[1]) * width
        #     shift_y = random.uniform(self.shift_range[0], self.shift_range[1]) * height
        #     image = F.affine(image, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)

        # Resize the image to the specified resolution
        image = F.resize(image, self.resize_resolution)

        return image

def augment_and_save_images(src_dir, dst_dir, num_augmentations=5):
    """Traverse src_dir, augment each image, and save to dst_dir preserving folder structure."""
    augmentation = ImageAugmentation()
    
    for root, dirs, files in os.walk(src_dir):
        # Determine the relative path of the current folder to the source directory
        rel_path = os.path.relpath(root, src_dir)
        # Create corresponding destination directory
        dst_folder = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_folder, exist_ok=True)
        
        for file in files:
            # Process common image file extensions only
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                try:
                    # Open the image and convert to RGB
                    image = Image.open(src_path).convert("RGB")
                except Exception as e:
                    print(f"Error opening {src_path}: {e}")
                    continue
                
                base_filename, ext = os.path.splitext(file)
                # Generate and save each augmented version
                for i in range(1, num_augmentations + 1):
                    augmented_image = augmentation(image)
                    output_filename = f"{base_filename}_aug_{i}{ext}"
                    output_path = os.path.join(dst_folder, output_filename)
                    augmented_image.save(output_path)
                print(f"Processed {src_path}: saved {num_augmentations} augmentations in {dst_folder}")

if __name__ == '__main__':
    # Set the source directory (your original dataset) and the destination directory
    src_folder = "aligned_augmented"  # folder with original images and subfolders
    dst_folder = "aligned_augmented_resized"  # new folder for augmented images
    num_augmentations = 5  # adjust the number of augmentations per image as needed

    augment_and_save_images(src_folder, dst_folder, num_augmentations)