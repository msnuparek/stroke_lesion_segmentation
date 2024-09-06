import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from monai.networks.nets import SegResNetDS
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation, Resize, NormalizeIntensity, ToTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
from matplotlib.colors import ListedColormap
import ants
from antspynet.utilities import brain_extraction
import nibabel as nib
import argparse  


parser = argparse.ArgumentParser(description="NIfTI Viewer with Segmentation")
parser.add_argument("-m", "--model_path", required=True, help="Path to the pre-trained model")
args = parser.parse_args()

class NiftiViewerApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("NIfTI Viewer")
        
        # Create a frame for the buttons to align them horizontally
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Load NIfTI button
        self.load_button = tk.Button(button_frame, text="Load NIfTI File", command=self.load_nifti)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Generate segmentation button (next to the Load NIfTI button)
        self.segment_button = tk.Button(button_frame, text="Generate Segmentation", command=self.generate_segmentation)
        self.segment_button.pack(side=tk.LEFT, padx=5)

        # Axial slider
        self.slider_axial = tk.Scale(root, from_=0, to=155, orient=tk.HORIZONTAL, label="Axial Slice", command=self.update_slices)
        self.slider_axial.pack()

        # Sagittal slider
        self.slider_sagittal = tk.Scale(root, from_=0, to=155, orient=tk.HORIZONTAL, label="Coronal Slice", command=self.update_slices)
        self.slider_sagittal.pack()

        # Create a 1x2 plot grid for axial and sagittal views
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # Load the model using the provided model path
        self.model, self.device = self.load_model(model_path)
        
    def load_nifti(self):
        global file_path
        file_path = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii *.nii.gz")])
        if not file_path:
            return

        try:
            # Define the transforms
            test_transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes='RAI'),
                Resize(spatial_size=[156, 156, 156])
            ])

            # Apply the transforms directly to the file path
            self.resized_img_data = test_transforms(file_path)

            # Initialize segmentation mask as None
            self.segmentation_mask = None

            # Update the slider range and display the first slice for each view
            max_slices = self.resized_img_data.shape[2] - 1
            self.slider_axial.config(to=max_slices)
            self.slider_sagittal.config(to=self.resized_img_data.shape[1] - 1)
            
            self.update_slices(0)
        except Exception as e:
            print(f"Error loading NIfTI file: {e}")

    def update_slices(self, slice_index):
        axial_slice = int(self.slider_axial.get())
        sagittal_slice = int(self.slider_sagittal.get())

        # Clear Axial view
        self.ax1.clear()
        self.ax1.imshow(self.resized_img_data[0, :, :, axial_slice], cmap="gray")
        self.ax1.set_title(f"Axial View (Slice {axial_slice})")

        # Clear Sagittal view and flip left to right (to show anterior to posterior)
        self.ax2.clear()
        sagittal_img = np.transpose(self.resized_img_data[0, :, sagittal_slice, :])  
        sagittal_img = np.fliplr(sagittal_img)  
        self.ax2.imshow(sagittal_img, cmap="gray")
        self.ax2.set_title(f"Coronal View (Slice {sagittal_slice})")

        # Update segmentation mask if available
        if self.segmentation_mask is not None:
            cmap = ListedColormap(['none', 'red'])

            # Axial view with segmentation
            self.ax1.imshow(self.segmentation_mask[:, :, axial_slice], cmap=cmap, alpha=0.5)

            # Sagittal view with segmentation, flip left to right for correct orientation
            sagittal_mask = np.transpose(self.segmentation_mask[:, sagittal_slice, :])
            sagittal_mask = np.fliplr(sagittal_mask)
            self.ax2.imshow(sagittal_mask, cmap=cmap, alpha=0.5)

        self.canvas.draw()

    def load_model(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SegResNetDS(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        model.load_state_dict(torch.load(model_path))  # Load model from the path provided via CLI
        model.eval()
        return model, device

    def perform_brain_extraction(self, file_path):
        try:
            # Read the image
            raw_img_ants = ants.image_read(file_path, reorient='RAI')

            # Perform brain extraction
            prob_brain_mask = brain_extraction(raw_img_ants, modality='t1', verbose=True)

            # Create the brain mask
            brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)

            # Mask the image
            masked = ants.mask_image(raw_img_ants, brain_mask)

            return masked
        except Exception as e:
            print(f"Error during brain extraction: {e}")
            return None

    def generate_segmentation(self):
        global file_path
        try:
            # Perform brain extraction
            masked = self.perform_brain_extraction(file_path)
            if masked is None:
                return

            # Convert to Nibabel format (required by MONAI)
            masked_brain = ants.to_nibabel(masked)
            nifti_file_path = './extracted.nii.gz'
            nib.save(masked_brain, nifti_file_path)

            # Define the transforms
            test_transforms = Compose([
                LoadImage(),
                EnsureChannelFirst(),
                Orientation(axcodes='RAI'),
                NormalizeIntensity(nonzero=True, channel_wise=True),
                Resize(spatial_size=[156, 156, 156]),
                ToTensor()
            ])

            # Apply the transforms directly to the resized image data
            image = test_transforms(nifti_file_path)
            
            sw_batch_size = 2
            roi_size = (152, 152, 152)

            # Perform inference
            with torch.no_grad():
                t_volume = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                test_outputs = sliding_window_inference(t_volume, roi_size, sw_batch_size, self.model)
                sigmoid_activation = Activations(sigmoid=True)
                test_outputs = sigmoid_activation(test_outputs)
                test_outputs = test_outputs > 0.53
                segmentation_mask = test_outputs.cpu().numpy()[0, 0]

                # Invert the mask
                self.segmentation_mask = np.logical_not(segmentation_mask).astype(np.uint8)

            # Update the plot with the segmentation mask
            self.update_slices(self.slider_axial.get())
        except Exception as e:
            print(f"Error during segmentation: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NiftiViewerApp(root, model_path=args.model_path)
    root.mainloop()
