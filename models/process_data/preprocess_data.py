# Preprocess all data to a common spacing and size, normalize intensities, and binarize labels
# import lib
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.io import loadmat
import os
import glob
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Dict, Optional, Union, Any
import pickle

# import modulus
from ..augmentation.noise_augmenter import NoiseAugmenter, NoiseProfileManager
from ..augmentation.augmentation_pipeline import MREAugmentation
from ..augmentation.basic_augment import SpatialAugmenter, IntensityAugmenter



class Preprocessor:
    """
    Simple preprocessor for medical images.
    """
    
    def __init__(self, target_spacing=(1.5, 1.5, 1.5), target_size=(128, 128, 64), is_2d=False):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.is_2d = is_2d
    
    def load_nifti(self, filepath):
        """Load NIfTI file"""
        nii = nib.load(filepath)
        data = nii.get_fdata().astype(np.float32)
        spacing = nii.header.get_zooms()[:3]
        return data, spacing
    
    def resample(self, volume, original_spacing, target_spacing, order=1):
        """Resample to target spacing"""
        zoom_factors = [orig/target for orig, target in zip(original_spacing, target_spacing)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
    
    def resize(self, volume, target_size, order=1):
        """Resize to target size"""
        zoom_factors = [target/current for target, current in zip(target_size, volume.shape)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
    
    def normalize(self, volume):
        """Z-score normalization"""
        valid_voxels = volume[volume > 0]
        if valid_voxels.size == 0:
            return volume
        mean = np.mean(valid_voxels)
        std = np.std(valid_voxels)
        return (volume - mean) / (std + 1e-8)
    
    def process_pair(self, image_path, label_path):
        """Process image-label pair"""
        # Load
        image, img_spacing = self.load_nifti(image_path)
        label, lbl_spacing = self.load_nifti(label_path)
        
        original_shape = image.shape
        
        # Resample to target spacing
        image = self.resample(image, img_spacing, self.target_spacing, order=1)
        label = self.resample(label, lbl_spacing, self.target_spacing, order=0)
        
        # Match label size to image
        if label.shape != image.shape:
            label = self.resize(label, image.shape, order=0)
        
        # Resize to target size
        if self.target_size:
            image = self.resize(image, self.target_size, order=1)
            label = self.resize(label, self.target_size, order=0)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize label
        label = (label > 0.5).astype(np.float32)
        
        return image, label, original_shape



class MedicalDataset(Dataset):
    """Dataset with metadata"""
    
    def __init__(self, images, labels, subject_names, original_shapes):
        self.images = images
        self.labels = labels
        self.subject_names = subject_names
        self.original_shapes = [tuple(s) if isinstance(s, (list, np.ndarray)) else s 
                                for s in original_shapes]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'subject_name': self.subject_names[idx],
            'original_shape': self.original_shapes[idx]
        }
        


class AugmentedDataset(Dataset):
    """Dataset wrapper for augmentation"""
    
    def __init__(self, base_dataset, augmentation=None, is_training=True, is_2d=False):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.is_training = is_training
        self.is_2d = is_2d
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        if self.is_training and self.augmentation:
            # Convert to numpy, remove channel dimension
            image_np = sample['image'].numpy()[0] if torch.is_tensor(sample['image']) else sample['image'][0]
            label_np = sample['label'].numpy()[0] if torch.is_tensor(sample['label']) else sample['label'][0]
            
            # Apply augmentation
            image_aug, label_aug = self.augmentation(
                image_np, label_np, sample['subject_name'], is_2d=self.is_2d
            )
            
            # Convert back and add channel dimension
            sample['image'] = torch.from_numpy(image_aug).float().unsqueeze(0)
            sample['label'] = torch.from_numpy(label_aug).float().unsqueeze(0)
        
        return sample
    
    
def custom_collate_fn(batch):
    """Collate function"""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'subject_name': [item['subject_name'] for item in batch],
        'original_shape': [item['original_shape'] for item in batch]
    }
    
    
def mat_to_nii(mat_path, ref_path):
    """Convert .mat to temporary .nii.gz"""
    ref = nib.load(ref_path)
    data_dict = loadmat(mat_path)
    
    # Find 3D array
    arr = None
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and not key.startswith('__'):
            arr = value
            break
    
    if arr is None:
        raise ValueError(f"No 3D array in {mat_path}")
    
    # Save temporary
    temp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(temp_dir, os.path.basename(mat_path).replace(".mat", ".nii.gz"))
    nii_img = nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header)
    nib.save(nii_img, tmp_path)
    
    return tmp_path