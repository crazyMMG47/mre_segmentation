import json
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
from typing import Tuple, List, Dict, Optional, Union
import pickle

# import module
from .preprocess_data import *


class NoiseProfileManager:
    """
    Manages loading and matching of scanner-specific noise profiles.
    Each subject has their own noise profile.
    """
    
    def __init__(self, noise_dir: Union[str, Path]):
        """
        Initialize noise profile manager.
        
        Args:
            noise_dir: Directory containing all noise profiles (*_noise.mat)
        """
        self.noise_dir = Path(noise_dir)
        self.noise_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        self.available_subjects: List[str] = []
        
        self._load_all_profiles()
    
    def _load_all_profiles(self):
        """Load all noise profiles from directory."""
        if not self.noise_dir.exists():
            raise ValueError(f"Noise directory does not exist: {self.noise_dir}")
        
        noise_files = sorted(self.noise_dir.glob("*_noise.mat"))
        
        if not noise_files:
            print(f"Warning: No *_noise.mat files found in {self.noise_dir}")
            return
        
        for noise_file in noise_files:
            try:
                # Extract subject ID (e.g., S001_noise.mat -> S001)
                subject_id = noise_file.stem.replace('_noise', '')
                
                # Load noise profile
                noise_data = loadmat(str(noise_file))
                
                # Check for 'noise' first, then 'noise_scaled'
                if 'noise' in noise_data:
                    noise_array = noise_data['noise']
                elif 'noise_scaled' in noise_data:
                    noise_array = noise_data['noise_scaled']
                else:
                    print(f"Warning: No 'noise' or 'noise_scaled' field in {noise_file.name}")
                    continue
                
                noise_array = np.asarray(noise_array).squeeze()
                
                # Store profile
                self.noise_profiles[subject_id] = {
                    'noise': noise_array,
                    'shape': noise_array.shape
                }
                
                self.available_subjects.append(subject_id)
                
            except Exception as e:
                print(f"Failed to load {noise_file.name}: {e}")
    
    def get_profile(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Get noise profile for a specific subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S001', 'G028')
        
        Returns:
            Noise array or None if not found
        """
        # Direct match
        if subject_id in self.noise_profiles:
            return self.noise_profiles[subject_id]['noise']
        
        # Fuzzy match: subject_id starts with profile_id
        for profile_id in self.available_subjects:
            if subject_id.startswith(profile_id):
                return self.noise_profiles[profile_id]['noise']
        
        # Partial match: profile_id is in subject_id
        for profile_id in self.available_subjects:
            if profile_id in subject_id:
                return self.noise_profiles[profile_id]['noise']
        
        return None
    
    def has_profile(self, subject_id: str) -> bool:
        """Check if noise profile exists for subject."""
        return self.get_profile(subject_id) is not None


def create_and_save_pairs(
    t2_dir: str,
    mask_dir: str,
    train_txt: str,
    val_txt: str,
    test_txt: str,
    noise_dir: Optional[str] = None,
    output_file: str = 'pairs_mapping.json'
):
    """
    Create pairs once and save to JSON file for reuse.
    Includes noise profile information if available.
    Run this ONCE before preprocessing.
    
    Args:
        t2_dir: Directory containing T2 images
        mask_dir: Directory containing segmentation masks
        train_txt: Path to training split file
        val_txt: Path to validation split file
        test_txt: Path to test split file
        noise_dir: Optional directory containing noise profiles
        output_file: Output JSON file path
    """
    print("\n" + "="*70)
    print("CREATING AND SAVING PAIRS MAPPING")
    print("="*70)
    
    # Initialize noise manager if directory provided
    noise_manager = None
    noise_dir_path = None
    if noise_dir and os.path.exists(noise_dir):
        try:
            noise_dir_path = Path(noise_dir)
            noise_manager = NoiseProfileManager(noise_dir)
            print(f"✓ Loaded {len(noise_manager.available_subjects)} noise profiles")
        except Exception as e:
            print(f"⚠️  Could not load noise profiles: {e}")
    
    # Read split files
    def read_ids(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    splits = {
        'train': read_ids(train_txt),
        'val': read_ids(val_txt),
        'test': read_ids(test_txt)
    }
    
    # Get all available files
    t2_files = {Path(f).stem.split('.')[0]: f for f in glob.glob(os.path.join(t2_dir, "*.nii*"))}
    
    mask_files = {}
    for f in glob.glob(os.path.join(mask_dir, "*.nii*")) + glob.glob(os.path.join(mask_dir, "*.mat")):
        subject_id = Path(f).stem.split('.')[0]
        mask_files[subject_id] = f
    
    # Create pairs for all splits
    all_pairs = {}
    
    for split_name, split_ids in splits.items():
        print(f"\n--- Finding pairs for {split_name} ---")
        
        pairs = []
        for subject_id in split_ids:
            # Find T2 image
            t2_path = None
            for t2_id, t2_file in t2_files.items():
                if subject_id in t2_id or t2_id in subject_id:
                    t2_path = t2_file
                    break
            
            # Find mask
            mask_path = None
            for mask_id, mask_file in mask_files.items():
                if subject_id in mask_id or mask_id in subject_id:
                    mask_path = mask_file
                    break
            
            # Check for noise profile and get the file path
            noise_file_path = None
            if noise_manager and noise_manager.has_profile(subject_id):
                # Construct the noise file path
                noise_file_path = str(noise_dir_path / f"{subject_id}_noise.mat")
                # Verify it exists
                if not os.path.exists(noise_file_path):
                    noise_file_path = None
            
            if t2_path and mask_path:
                pair_info = {
                    'image': t2_path,
                    'label': mask_path,
                    'subject_id': subject_id,
                    'noise_path': noise_file_path
                }
                pairs.append(pair_info)
                
                noise_status = "✓ with noise" if noise_file_path else "○ no noise"
                print(f"  ✓ {subject_id} {noise_status}")
            else:
                missing = []
                if not t2_path:
                    missing.append('T2')
                if not mask_path:
                    missing.append('mask')
                print(f"  ✗ {subject_id} - missing {', '.join(missing)}")
        
        all_pairs[split_name] = pairs
        noise_count = sum(1 for p in pairs if p['noise_path'] is not None)
        print(f"Found {len(pairs)} pairs for {split_name} ({noise_count} with noise profiles)")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"\n✓ Saved pairs mapping to {output_file}")
    print("="*70 + "\n")
    
    return all_pairs


def load_pairs_from_file(pairs_file: str = 'pairs_mapping.json') -> Dict:
    """Load pre-computed pairs from JSON file"""
    with open(pairs_file, 'r') as f:
        return json.load(f)


def preprocess_and_save_as_dict(
    pairs_file: str,
    output_dir: str,
    noise_dir: Optional[str] = None,
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    target_size: Tuple[int, int, int] = (128, 128, 64),
    is_2d: bool = False
):
    """
    Preprocess using pre-saved pairs mapping and save as dictionaries.
    Each sample is a dict with: image, label, subject_id, noise (optional).
    
    Args:
        pairs_file: Path to JSON file with pairs mapping
        output_dir: Output directory for preprocessed data
        noise_dir: Optional directory containing noise profiles (not used, kept for compatibility)
        target_spacing: Target voxel spacing
        target_size: Target image size
        is_2d: Whether to process as 2D data
    """
    
    print("\n" + "="*70)
    print("PREPROCESSING WITH DICTIONARY FORMAT")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pairs from file
    all_pairs = load_pairs_from_file(pairs_file)
    
    # Note: noise_dir parameter is kept for backward compatibility but not used
    # Noise paths are now stored directly in the pairs_file JSON
    
    # Create preprocessor
    preprocessor = Preprocessor(target_spacing, target_size, is_2d)
    
    # Process each split
    for split_name, pairs in all_pairs.items():
        print(f"\n--- Processing {split_name} split ({len(pairs)} subjects) ---")
        
        if not pairs:
            print(f"⚠️  No pairs for {split_name} split!")
            continue
        
        # Process each pair into a dictionary
        dataset_list = []
        
        for pair in pairs:
            try:
                label_path = pair['label']
                
                # Convert .mat to .nii if needed
                if label_path.endswith('.mat'):
                    label_path = mat_to_nii(label_path, pair['image'])
                
                # Process image and label
                img, lbl, orig_shape = preprocessor.process_pair(pair['image'], label_path)
                
                # Load noise profile if path is available
                noise_profile = None
                if pair.get('noise_path') and os.path.exists(pair['noise_path']):
                    try:
                        noise_data = loadmat(pair['noise_path'])
                        if 'noise' in noise_data:
                            noise_profile = np.asarray(noise_data['noise']).squeeze()
                        elif 'noise_scaled' in noise_data:
                            noise_profile = np.asarray(noise_data['noise_scaled']).squeeze()
                    except Exception as e:
                        print(f"    Warning: Could not load noise from {pair['noise_path']}: {e}")
                
                # Create sample dictionary
                sample_dict = {
                    'image': img,  # numpy array (D, H, W) or (H, W)
                    'label': lbl,  # numpy array (D, H, W) or (H, W)
                    'subject_id': pair['subject_id'],
                    'original_shape': orig_shape,
                    'noise': noise_profile  # None if not available
                }
                
                dataset_list.append(sample_dict)
                
                noise_status = "✓ with noise" if noise_profile is not None else "○"
                print(f"  {noise_status} {pair['subject_id']}")
                
            except Exception as e:
                print(f"  ✗ Error processing {pair['subject_id']}: {e}")
                continue
        
        if not dataset_list:
            print(f"⚠️  No successful processing for {split_name} split!")
            continue
        
        
        noise_count = sum(1 for s in dataset_list if s['noise'] is not None)
        print(f"✓ Saved {split_name}: {len(dataset_list)} samples ({noise_count} with noise) to {output_path}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70 + "\n")


def load_preprocessed_dataset(dataset_path: str) -> Dict:
    """
    Load preprocessed dataset from pickle file.
    
    Args:
        dataset_path: Path to dataset.pkl file
        
    Returns:
        Dictionary with 'samples' (list of dicts) and 'metadata'
    """
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


class DictDataset(Dataset):
    """
    PyTorch Dataset that works with dictionary format.
    Each item returns a dict with: image, label, subject_id, noise (optional)
    """
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Args:
            dataset_path: Path to dataset.pkl file
            transform: Optional transform to apply
        """
        self.data = load_preprocessed_dataset(dataset_path)
        self.samples = self.data['samples']
        self.metadata = self.data['metadata']
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        image = torch.from_numpy(sample['image']).unsqueeze(0).float()  # Add channel dim
        label = torch.from_numpy(sample['label']).unsqueeze(0).float()  # Add channel dim
        
        # Prepare output dict
        output = {
            'image': image,
            'label': label,
            'subject_id': sample['subject_id'],
            'original_shape': sample['original_shape']
        }
        
        # Add noise if available
        if sample['noise'] is not None:
            noise = torch.from_numpy(sample['noise']).float()
            output['noise'] = noise
        
        # Apply transforms if provided
        if self.transform:
            output = self.transform(output)
        
        return output
    
    def has_noise(self):
        """Check if this dataset includes noise profiles"""
        return self.metadata.get('has_noise', False)
    
    