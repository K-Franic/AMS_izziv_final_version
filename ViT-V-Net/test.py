import os
import torch
import numpy as np
import nibabel as nib
import models
from utils import register_model
from models import CONFIGS as CONFIGS_ViT_seg
import device_helper

def load_nifti_image(file_path):
    """Loads a NIfTI image and converts it to a PyTorch tensor."""
    image = nib.load(file_path).get_fdata(dtype=np.float32)
    if len(image.shape) == 3: 
        image = image[np.newaxis, ...]
    return torch.tensor(image, dtype=torch.float32)

def save_nifti_image(tensor, reference_nifti, output_path):
    """Saves a PyTorch tensor as a NIfTI file using a reference image."""
    tensor = tensor.squeeze().cpu().numpy()
    affine = nib.load(reference_nifti).affine
    nib.save(nib.Nifti1Image(tensor, affine), output_path)

def generate_deformation_fields(model_path, pairs, data_dir, output_dir, img_size=(256, 192, 192)):
    """
    Generates deformation fields for specified pairs of fixed and moving images.
    
    Args:
        model_path: Path to the saved model (.pth file).
        pairs: List of (case_id, fixed_suffix, moving_suffix) tuples.
        data_dir: Directory containing the images.
        output_dir: Directory to save deformation fields.
        img_size: Tuple representing the input image size.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    model = models.ViTVNet(config_vit, img_size=(256, 192, 192)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Registration model
    reg_model = register_model(img_size, mode='nearest').to(device).eval()

    # Process each pair of images
    for case_id, fixed_suffix, moving_suffix in pairs:
        fixed_path = os.path.join(data_dir, f"ThoraxCBCT_{case_id}_{fixed_suffix}.nii.gz")
        moving_path = os.path.join(data_dir, f"ThoraxCBCT_{case_id}_{moving_suffix}.nii.gz")
        
        # Load images
        fixed = load_nifti_image(fixed_path).to(device)
        moving = load_nifti_image(moving_path).to(device)
        # Concatenate images along channel dimension
        x_in = torch.cat((fixed.unsqueeze(0), moving.unsqueeze(0)), dim=1) 

        # Forward pass through the model
        with torch.no_grad():
            output = model(x_in)
            deformation_field = output[1]
        # Save deformation field
        output_filename = f"disp_{case_id}_{fixed_suffix}_{case_id}_{moving_suffix}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        save_nifti_image(deformation_field.permute(2,3,4,0,1), fixed_path, output_path)

        print(f"Saved deformation field: {output_path}")

if __name__ == "__main__":
    # Paths to the saved model and image directory
    model_path = "ViT-V-Net/Pretrained_models/Vitvnet500metric_0.486.pth.tar"
    data_directory = "Release_06_12_23/imagesTr"
    output_directory = "evaluation_def_fields"

    # Validation pairs
    validation_pairs = [
        ("0011", "0001", "0000"),
        ("0012", "0001", "0000"),
        ("0013", "0001", "0000"),
        ("0011", "0002", "0000"),
        ("0012", "0002", "0000"),
        ("0013", "0002", "0000"),
    ]

    # Generate deformation fields
    generate_deformation_fields(model_path, validation_pairs, data_directory, output_directory)
