import os
import argparse
import numpy as np
import torch
from monai.networks.nets import SegResNetDS
from monai.transforms import Resize, Compose, LoadImage, EnsureChannelFirst, Orientation, NormalizeIntensity, ToTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
import ants
from antspynet.utilities import brain_extraction
import nibabel as nib
from scipy.ndimage import zoom

def load_model(model_weights_path, device):
    model = SegResNetDS(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    return model

def perform_brain_extraction(file_path):
    try:
        raw_img_ants = ants.image_read(file_path, reorient='RAI')
        prob_brain_mask = brain_extraction(raw_img_ants, modality='t1', verbose=True)
        brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)
        masked = ants.mask_image(raw_img_ants, brain_mask)
        return masked
    except Exception as e:
        print(f"Error during brain extraction: {e}")
        return None

def generate_segmentation(nifti_file_path, model, device):
    try:
        # Perform brain extraction
        masked = perform_brain_extraction(nifti_file_path)
        if masked is None:
            return None, None

        # Save the brain-extracted image temporarily
        masked_brain = ants.to_nibabel(masked)
        temp_file_path = './extracted.nii.gz'
        nib.save(masked_brain, temp_file_path)

        # Define the transforms
        test_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            Resize(spatial_size=[156, 156, 156]),
            ToTensor()
        ])

        # Apply the transforms
        image = test_transforms(temp_file_path)
        sw_batch_size = 2
        roi_size = (152, 152, 152)

        with torch.no_grad():
            t_volume = image.unsqueeze(0).to(device)
            test_outputs = sliding_window_inference(t_volume, roi_size, sw_batch_size, model)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.53
            segmentation_mask = test_outputs.cpu().numpy()[0, 0]

            segmentation_mask_inverted = np.logical_not(segmentation_mask).astype(np.uint8)
            return segmentation_mask_inverted, masked_brain.shape
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None, None

def resample_mask_to_original_size(segmentation_mask, original_shape):
    try:
        current_shape = segmentation_mask.shape
        zoom_factors = np.array(original_shape) / np.array(current_shape)
        resampled_mask = zoom(segmentation_mask, zoom_factors, order=0)  # Nearest-neighbor interpolation
        return resampled_mask
    except Exception as e:
        print(f"Error during resampling: {e}")
        return None
    
  
def adjust_orientation_to_inferior(orientation):
    """
    Adjust the given orientation to ensure the last axis is 'inferior'.
    """
    if len(orientation) != 3:
        raise ValueError("Orientation must be a 3-character string")
    
    return orientation[:2] + 'I'

def save_segmentation(segmentation_mask, save_folder, original_file_path, affine, original_shape, original_orientation):
    try:
        # Resample the segmentation mask to the original image size
        resampled_mask = resample_mask_to_original_size(segmentation_mask, original_shape)

        if resampled_mask is None:
            return

        # Convert the mask to ANTs image
        ants_mask = ants.from_numpy(resampled_mask, origin=None, spacing=None, direction=None)

        # Reorient the mask to the original orientation with 'inferior' as the last axis
        ants_mask_original = ants_mask.reorient_image2(original_orientation)

        # Convert back to numpy array
        resampled_mask_original = ants_mask_original.numpy()

        # Construct the output file name
        base_name = os.path.basename(original_file_path).replace('.nii', '').replace('.gz', '')
        output_file_name = f"{base_name}_seg_mask.nii.gz"
        output_file_path = os.path.join(save_folder, output_file_name)

        # Save the resampled and reoriented segmentation mask
        segmentation_img = nib.Nifti1Image(resampled_mask_original, affine=affine)
        nib.save(segmentation_img, output_file_path)
        print(f"Segmentation mask saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving segmentation mask: {e}")

def main(nifti_file_path, save_folder, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_weights_path, device)

    # Load the original image to get the affine matrix, shape, and orientation
    original_image = nib.load(nifti_file_path)
    affine = original_image.affine
    original_shape = original_image.shape
    
    # Extract orientation from the original image and ensure inferior
    original_ants_image = ants.image_read(nifti_file_path)
    original_orientation = original_ants_image.orientation
    adjusted_orientation = adjust_orientation_to_inferior(original_orientation)
    print(f"Original image orientation: {original_orientation}")
    print(f"Adjusted orientation: {adjusted_orientation}")

    # Generate segmentation
    segmentation_mask, masked_shape = generate_segmentation(nifti_file_path, model, device)

    if segmentation_mask is not None and masked_shape is not None:
        save_segmentation(segmentation_mask, save_folder, nifti_file_path, affine, original_shape, adjusted_orientation)
    else:
        print("Segmentation failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIfTI Segmentation Predictor")
    parser.add_argument('-i', '--input', required=True, help="Path to the input NIfTI file")
    parser.add_argument('-o', '--output_folder', required=True, help="Path to the folder to save the segmentation mask")
    parser.add_argument('-w', '--weights', required=True, help="Path to the pretrained model weights")

    args = parser.parse_args()

    main(args.input, args.output_folder, args.weights)
