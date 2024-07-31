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

class NiftiViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NIfTI Viewer")
        
        self.load_button = tk.Button(root, text="Load NIfTI File", command=self.load_nifti)
        self.load_button.pack()

        self.slider = tk.Scale(root, from_=0, to=155, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slider.pack()

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.segment_button = tk.Button(root, text="Generate Segmentation", command=self.generate_segmentation)
        self.segment_button.pack()

        self.model, self.device = self.load_model()
        
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

            # Update the slider range and display the first slice
            self.slider.config(to=self.resized_img_data.shape[2] - 1)
            self.update_slice(0)
        except Exception as e:
            print(f"Error loading NIfTI file: {e}")

    def update_slice(self, slice_index):
        slice_index = int(slice_index)
        self.ax1.clear()
        self.ax1.imshow(self.resized_img_data[0, :, :, slice_index], cmap="gray")
        self.ax1.set_title(f"Original Slice {slice_index}")

        self.ax2.clear()
        if self.segmentation_mask is not None:
            cmap = ListedColormap(['red', 'none'])
            self.ax2.imshow(self.resized_img_data[0, :, :, slice_index], cmap="gray")
            self.ax2.imshow(self.segmentation_mask[:, :, slice_index], cmap=cmap, alpha=0.5)
            self.ax2.set_title(f"Segmented Slice {slice_index}")
        else:
            self.ax2.imshow(self.resized_img_data[0, :, :, slice_index], cmap="gray")
            self.ax2.set_title("Segmented Slice (No Mask)")

        self.canvas.draw()

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SegResNetDS(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        model.load_state_dict(torch.load("/mnt/d/results_SegResNet_200ep_280724_dice7818/best_metric_model.pth"))
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
            
            # Check the shape of the image after transforms
            print("Shape of image after transforms:", image.shape)
            
            sw_batch_size = 2
            roi_size = (152, 152, 152)

            # Perform inference
            with torch.no_grad():
                t_volume = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                test_outputs = sliding_window_inference(t_volume, roi_size, sw_batch_size, self.model)
                sigmoid_activation = Activations(sigmoid=True)
                test_outputs = sigmoid_activation(test_outputs)
                test_outputs = test_outputs > 0.53
                self.segmentation_mask = test_outputs.cpu().numpy()[0, 0]

            # Update the plot with the segmentation mask
            self.update_slice(self.slider.get())
        except Exception as e:
            print(f"Error during segmentation: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NiftiViewerApp(root)
    root.mainloop()
