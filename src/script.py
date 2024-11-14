import os
import random
import shutil

# Paths
dataset_dir = "."
train_dir = "images/train"
val_dir = "images/val"
test_dir = "images/test"

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png'))]

# Shuffle the dataset for randomness
random.shuffle(image_files)

# Split sizes
train_split = int(0.7 * len(image_files))
val_split = int(0.2 * len(image_files))

# Splitting
train_files = image_files[:train_split]
val_files = image_files[train_split:train_split + val_split]
test_files = image_files[train_split + val_split:]

# Function to move files
def move_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# Move files
move_files(train_files, dataset_dir, train_dir)
move_files(val_files, dataset_dir, val_dir)
move_files(test_files, dataset_dir, test_dir)

print("Dataset split completed!")
