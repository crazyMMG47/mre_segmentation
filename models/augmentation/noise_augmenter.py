# noise_augmenter.py
# NoiseProfileManager (just basic profile matching): load profiles, match to subject IDs 
# NoiseAugmenter: apply relative noise to images using matched profile

import numpy as np
import scipy.io as sio
import random
from scipy.ndimage import zoom
from typing import Tuple
from utils.noise_profile_manager import NoiseProfileManager


import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, Optional, List, Union
import random


class NoiseProfileManager:
    """
    Each subject has their own noise profile.
    Manages loading and matching of scanner-specific noise profiles."""
    
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
            raise ValueError(f"No *_noise.mat files found in {self.noise_dir}")
        
        for noise_file in noise_files:
            try:
                # Extract subject ID (e.g., S001_noise.mat -> S001)
                subject_id = noise_file.stem.replace('_noise', '')
                
                # Load noise profile
                noise_data = sio.loadmat(str(noise_file))
                noise_array = noise_data.get('noise') or noise_data.get('noise_scaled')
                
                if noise_array is None:
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
    
    def get_profile(self, subject_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get noise profile for a specific subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S001', 'G028')
        
        Returns:
            Dictionary containing noise array or None if not found
        """
        # Direct match
        if subject_id in self.noise_profiles:
            return self.noise_profiles[subject_id]
        
        # Fuzzy match: subject_id starts with profile_id
        for profile_id in self.available_subjects:
            if subject_id.startswith(profile_id):
                return self.noise_profiles[profile_id]
        
        # Partial match: profile_id is in subject_id
        for profile_id in self.available_subjects:
            if profile_id in subject_id:
                return self.noise_profiles[profile_id]
        
        return None
    
    def has_profile(self, subject_id: str) -> bool:
        """Check if noise profile exists for subject."""
        return self.get_profile(subject_id) is not None
    
    
class NoiseAugmenter:
    """Applies realistic MRE noise using relative noise injection."""
    
    def __init__(self, 
                 noise_manager: NoiseProfileManager,
                 noise_strength_range: Tuple[float, float] = (0.05, 0.15),
                 apply_prob: float = 0.75):
        """
        Initialize noise augmenter.
        
        Args:
            noise_manager: NoiseProfileManager instance
            noise_strength_range: Noise strength as fraction of image std (e.g., 0.1 = 10%)
            apply_prob: Probability of applying noise
        """
        self.noise_manager = noise_manager
        self.noise_strength_range = noise_strength_range
        self.apply_prob = apply_prob
    
    def __call__(self, 
                 image: np.ndarray, 
                 subject_id: str,
                 is_2d: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Apply relative noise to image.
        
        Args:
            image: Input image (2D or 3D numpy array)
            subject_id: Subject identifier
            is_2d: Whether image is 2D slice
        
        Returns:
            Tuple of (noisy_image, was_noise_applied)
        """
        # Decide whether to apply noise
        if random.random() > self.apply_prob:
            return image, False
        
        # Get noise profile
        profile = self.noise_manager.get_profile(subject_id)
        
        # For subjects without profile, simply return original image.
        if profile is None:
            return image, False
        
        # Match noise dimensions to image
        noise = self._match_dimensions(image, profile['noise'], is_2d)
        
        # Apply relative noise
        noisy_image = self._apply_relative_noise(image, noise)
        
        return noisy_image, True
    
    def _match_dimensions(self, 
                          image: np.ndarray, 
                          noise_profile: np.ndarray,
                          is_2d: bool) -> np.ndarray:
        """Match noise profile dimensions to image."""
        
        if is_2d:
            # Extract 2D slice from 3D noise if needed
            if noise_profile.ndim == 3:
                mid_slice = noise_profile.shape[2] // 2
                noise = noise_profile[:, :, mid_slice]
            else:
                noise = noise_profile
        else:
            noise = noise_profile
        
        # Resize if shapes don't match
        if noise.shape != image.shape:
            zoom_factors = tuple(img_dim / noise_dim 
                                for img_dim, noise_dim in zip(image.shape, noise.shape))
            noise = zoom(noise, zoom_factors, order=1)
        
        return noise
    
    def _apply_relative_noise(self, image: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Apply relative noise: Scale noise to percentage of image std.
        Scale-invariant and predictable.
        """
        # Normalize noise to unit variance
        valid_noise = noise[np.isfinite(noise)]
        if valid_noise.size == 0:
            return image
        
        noise_std = np.std(valid_noise)
        if noise_std == 0:
            return image
        
        normalized_noise = (noise - np.mean(valid_noise)) / noise_std
        
        # Calculate image std (only non-zero voxels)
        valid_image = image[image > 0]
        if valid_image.size == 0:
            return image
        
        image_std = np.std(valid_image)
        
        # Random noise strength
        noise_strength = random.uniform(*self.noise_strength_range)
        
        # Scale and add noise
        scaled_noise = normalized_noise * image_std * noise_strength
        noisy_image = image + scaled_noise
        
        # Preserve non-negative values
        if image.min() >= 0:
            noisy_image = np.maximum(noisy_image, 0)
        
        return noisy_image