# 2 basic augmentation classes for medical images 
# spatial augmenter (rotation and scaling only)
# intensity augmenter (brightness, contrast only)
# let's keep the intensity augmenter for now
# since we have the noise augmenter, we can skip intensity augmentation now

import numpy as np
import random
from scipy.ndimage import zoom, rotate
from typing import Tuple, Optional, Dict, Any


class SpatialAugmenter:
    """Spatial augmentations for medical images with parameter tracking."""
    
    def __init__(self,
                 rotation_range: Tuple[float, float] = (-10, 10),
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 is_2d: bool = False):
        """
        Initialize spatial augmenter.
        
        Args:
            rotation_range: Rotation angle range in degrees
            scale_range: Scaling factor range
            is_2d: Whether working with 2D or 3D data
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.is_2d = is_2d
    
    def __call__(self, 
                 image: np.ndarray, 
                 label: np.ndarray,
                 return_params: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """
        Apply spatial augmentations to image-label pair.
        
        Args:
            image: Input image
            label: Input label
            return_params: If True, return transformation parameters
        
        Returns:
            If return_params=False: (image, label)
            If return_params=True: (image, label, params_dict)
        """
        params = {}
        
        # Random rotation
        if self.rotation_range != (0, 0):
            angle = random.uniform(*self.rotation_range)
            axes = random.choice([(1, 2), (0, 2), (0, 1)]) if not self.is_2d else None
            image, label = self._rotate(image, label, angle, axes)
            params['rotation'] = {'angle': angle, 'axes': axes}
        else:
            params['rotation'] = None
        
        # Random scaling
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            image, label = self._scale(image, label, scale)
            params['scale'] = {'factor': scale}
        else:
            params['scale'] = None
        
        if return_params:
            return image, label, params
        else:
            return image, label
    
    def apply_to(self, 
                 array: np.ndarray, 
                 params: Dict[str, Any], 
                 is_label: bool = False) -> np.ndarray:
        """
        Apply same transformation to another array (e.g., noise field). 
        This is used in the augmentation pipeline wrapper.
        
        Returns: 
            Transformed array
        """
        order = 0 if is_label else 1
        
        # Apply rotation
        if params.get('rotation') is not None:
            rot_params = params['rotation']
            angle = rot_params['angle']
            axes = rot_params['axes']
            
            if self.is_2d:
                array = rotate(array, angle, reshape=False, order=order, mode='constant', cval=0)
            else:
                array = rotate(array, angle, axes=axes, reshape=False, order=order, mode='constant', cval=0)
        
        # Apply scaling
        if params.get('scale') is not None:
            scale_factor = params['scale']['factor']
            original_shape = array.shape
            zoom_factors = [scale_factor] * array.ndim
            
            array = zoom(array, zoom_factors, order=order, mode='constant', cval=0)
            array = self._resize_to_shape(array, original_shape)
        
        return array
    
    def _rotate(self, 
                image: np.ndarray, 
                label: np.ndarray, 
                angle: float,
                axes: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotation."""
        if self.is_2d:
            image_rot = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
            label_rot = rotate(label, angle, reshape=False, order=0, mode='constant', cval=0)
        else:
            image_rot = rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)
            label_rot = rotate(label, angle, axes=axes, reshape=False, order=0, mode='constant', cval=0)
        
        return image_rot, label_rot
    
    def _scale(self, image: np.ndarray, label: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply scaling."""
        original_shape = image.shape
        zoom_factors = [scale] * image.ndim
        
        image_scaled = zoom(image, zoom_factors, order=1, mode='constant', cval=0)
        label_scaled = zoom(label, zoom_factors, order=0, mode='constant', cval=0)
        
        # Crop or pad to original size
        image_scaled = self._resize_to_shape(image_scaled, original_shape)
        label_scaled = self._resize_to_shape(label_scaled, original_shape)
        
        return image_scaled, label_scaled
    
    def _resize_to_shape(self, array: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize array to target shape by cropping or padding."""
        if array.shape == target_shape:
            return array
        
        result = np.zeros(target_shape, dtype=array.dtype)
        
        # Calculate slices for cropping/copying
        slices_src = []
        slices_dst = []
        
        for current, target in zip(array.shape, target_shape):
            if current > target:
                # Need to crop
                start = (current - target) // 2
                slices_src.append(slice(start, start + target))
                slices_dst.append(slice(None))
            else:
                # Need to pad
                start = (target - current) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + current))
        
        result[tuple(slices_dst)] = array[tuple(slices_src)]
        
        return result
    

class IntensityAugmenter:
    """Intensity augmentations for medical images."""
    
    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.95, 1.05),
                 contrast_range: Tuple[float, float] = (0.95, 1.05),
                 apply_prob: float = 0.7):
        """
        Initialize intensity augmenter.
        
        Args:
            brightness_range: Brightness multiplication range
            contrast_range: Contrast multiplication range
            gamma_range: Gamma correction range
            apply_prob: Probability of applying each augmentation
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.apply_prob = apply_prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply intensity augmentations to image."""
        
        # Brightness
        if random.random() < self.apply_prob and self.brightness_range != (1.0, 1.0):
            factor = random.uniform(*self.brightness_range)
            image = image * factor
        
        # Contrast
        if random.random() < self.apply_prob and self.contrast_range != (1.0, 1.0):
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        return image