import os
import shutil
import random

# === CONFIG ===
main_folder = 'aligned'   
output_folder = 'augmented_dataset' 
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.1
seed = 42

# === SETUP ===
random.seed(seed)
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# === GET FOLDERS ===
person_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
random.shuffle(person_folders)

total = len(person_folders)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

split_people = {
    'train': person_folders[:train_end],
    'val': person_folders[train_end:val_end],
    'test': person_folders[val_end:]
}

# === COPY FOLDERS TO EACH SPLIT ===
for split, people in split_people.items():
    for person in people:
        src_folder = os.path.join(main_folder, person)
        dst_folder = os.path.join(output_folder, split, person)
        shutil.copytree(src_folder, dst_folder)

for split in splits:
    total_images = 0
    split_path = os.path.join(output_folder, split)
    for person_folder in os.listdir(split_path):
        person_path = os.path.join(split_path, person_folder)
        if not os.path.isdir(person_path):
            continue
        total_images += len([f for f in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, f))])
    print(f"{split.upper()}: {total_images} images")

print("Dataset split by person completed!")
