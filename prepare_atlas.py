import os
import shutil
import random

# Define the source directory and the target directories
source_dir = "C:/Users/mats/Desktop/Pytorch/datasets/ATLAS/Data_Training_655"
prepare_dir = 'C:/Users/mats/Desktop/Pytorch/datasets/ATLAS/Atlas_prepared'
train_volumes_dir = os.path.join(prepare_dir, 'TrainVolumes')
train_masks_dir = os.path.join(prepare_dir, 'TrainSegmentation')
test_volumes_dir = os.path.join(prepare_dir, 'TestVolumes')
test_masks_dir = os.path.join(prepare_dir, 'TestSegmentation')

os.makedirs(train_volumes_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(test_volumes_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

nii_files = []

for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))


# Iterate over all files in the source directory
for filename in nii_files:
    if filename.endswith('T1w.nii.gz'):
        # Copy to train_volumes directory
        shutil.copy(os.path.join(source_dir, filename), train_volumes_dir)
    elif filename.endswith('mask.nii.gz'):
        # Copy to train_masks directory
        shutil.copy(os.path.join(source_dir, filename), train_masks_dir)



# List all the train volumes
train_volumes = [f for f in os.listdir(train_volumes_dir)]

# Calculate 20% of the total train volumes
num_test_volumes = int(len(train_volumes) * 0.1)

# Select a random 20% of train volumes
test_volumes = random.sample(train_volumes, num_test_volumes)

# Copy the selected train volumes and their corresponding masks to test directories
for volume in test_volumes:

    # move volume
    shutil.move(os.path.join(train_volumes_dir, volume), test_volumes_dir)
    
    # Derive the corresponding mask filename
    mask_filename = volume.replace('_T1w.nii.gz', '_space-orig_label-L_desc-T1lesion_mask.nii.gz')
    
    # Copy mask if it exists
    mask_path = os.path.join(train_masks_dir, mask_filename)
    if os.path.exists(mask_path):
        shutil.move(mask_path, test_masks_dir)