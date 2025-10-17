# Augmentation pipeline that combines the basic augmentation with mre subject's noise profile 
# to perform data augmentation on medical images

# Order matters when we augment our data!
# Augmentation order:
# 1. spatially augment (image, label) -> record transformation 
# 2. Apply the same spatial transformation to the noise profile of a subject
# 3. Add the transformed noise of the subject to the spatially augmented image.


from typing import Dict, Optional, Tuple, List, Union, Any
from .basic_augment import SpatialAugmenter, IntensityAugmenter
from .noise_augmenter import NoiseAugmenter
import random
import numpy as np


class MREAugmentation:
    
    """
    Complete augmentation pipeline for MRE segmentation.
    Order: 
    1. Spatial (image+label+noise)
    2. Intensity is kept as optional now (disabled here)
    3. Add noise (image only)
    """

    def __init__(self,
                 noise_augmenter: Optional['NoiseAugmenter'] = None,
                 spatial_augmenter: Optional['SpatialAugmenter'] = None,
                #  intensity_augmenter: Optional['IntensityAugmenter'] = None,
                 apply_prob: float = 0.80):
        self.noise_augmenter = noise_augmenter
        self.spatial_augmenter = spatial_augmenter
        # self.intensity_augmenter = intensity_augmenter
        self.apply_prob = apply_prob

    def __call__(self,
                 image: np.ndarray,
                 label: np.ndarray,
                 subject_id: str,
                 is_2d: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image
            label: Input label
            subject_id: Subject identifier for noise matching
            is_2d: Whether data is 2D
        
        Returns:
            Augmented (image, label) pair
        """
        
        # Skip augmentation with probability
        if random.random() > self.apply_prob:
            return image, label

        # Apply spatial augmentation to image, label (i.e. binary mask), and noise field
        # spatial_param: dict of the spatial transformation applied, initialize as None
        spatial_params = None
        if self.spatial_augmenter is not None:
            image, label, spatial_params = self.spatial_augmenter(
                image, label, return_params=True
            )

        # Intensity augmentation (image only) - Add code if you want intensity augmentation prior to the noise injection
        # if self.intensity_augmenter is not None:
        #     image = self.intensity_augmenter(image)

        # Noise augmentation applied to image (i.e. t2stack) only
        if self.noise_augmenter is not None:
            image = self._apply_noise_with_spatial_warp(
                image, subject_id, is_2d, spatial_params
            )

        return image, label

    def _apply_noise_with_spatial_warp(self,
                                       image: np.ndarray,
                                       subject_id: str,
                                       is_2d: bool,
                                       spatial_params: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Load noise field, warp it with same spatial transform, then add to image.
        
        Args:
            image: Input image
            subject_id: Subject identifier
            is_2d: Whether image is 2D
            spatial_params: Spatial transformation parameters
        
        Returns:
            Image with noise added
        """
        # Load noise field for this subject
        noise_field = self.noise_augmenter.load_field(subject_id, is_2d=is_2d)
        
        # If no noise profile found, return image unchanged
        # some subjects don't have noise profiles
        if noise_field is None:
            return image
        
        # Warp noise field with same spatial transform as image
        if spatial_params is not None and self.spatial_augmenter is not None:
            noise_field = self.spatial_augmenter.apply_to(
                noise_field, spatial_params, is_label=False
            )
        
        # Add warped noise to image
        image = self.noise_augmenter.add(image, noise_field=noise_field)
        
        return image