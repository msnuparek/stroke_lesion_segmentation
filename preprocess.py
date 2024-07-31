import os
from glob import glob
#import shutil
#from tqdm import tqdm
#import dicom2nifti
#import numpy as np
#import nibabel as nib
from monai.transforms import *
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism


def prepare(prepare_dir, spatial_size=[156, 156, 156]):

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(prepare_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(prepare_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(prepare_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(prepare_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    train_transforms = Compose(
                        [
                            LoadImaged(keys=["vol", "seg"]),
                            EnsureChannelFirstd(keys=["vol", "seg"]),
                            Orientationd(keys=["vol", "seg"], axcodes='RAI'),
                            RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=0),
                            RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=1),
                            RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=2),
                            NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
                            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
                            ToTensord(keys=["vol", "seg"])
                        ]
                    )
    
    test_transforms = Compose(
        [
                            LoadImaged(keys=["vol", "seg"]),
                            EnsureChannelFirstd(keys=["vol", "seg"]),
                            Orientationd(keys=["vol", "seg"], axcodes='RAI'),
                            NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
                            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
                            ToTensord(keys=["vol", "seg"])

        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1)

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader