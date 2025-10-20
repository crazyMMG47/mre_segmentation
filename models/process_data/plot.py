import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def _plot_noise_on_axis(ax, sample: Dict, slice_idx: int):
    """Helper function to plot noise with its shape in the title and a scale bar."""
    noise = sample.get('noise')
    
    if noise is None:
        ax.text(0.5, 0.5, 'No Noise Available', ha='center', va='center', fontsize=12, color='gray')
        ax.set_title("No Noise", fontsize=10)
        ax.axis('off')
        return

    is_image_plot = False

    if noise.ndim == 3: # 3D noise
        noise_depth = noise.shape[2]
        image_depth = sample['image'].shape[2]
        noise_slice_idx = min(int(slice_idx * noise_depth / image_depth), noise_depth - 1)
        
        ax.imshow(noise[:, :, noise_slice_idx], cmap='viridis', aspect='equal')
        ax.set_title(f"Noise Slice {noise_slice_idx + 1}/{noise_depth}\nShape: {noise.shape}", fontsize=10)
        is_image_plot = True
        
    elif noise.ndim == 2: # 2D noise
        ax.imshow(noise, cmap='viridis', aspect='equal')
        ax.set_title(f"Noise Profile (2D)\nShape: {noise.shape}", fontsize=10)
        is_image_plot = True
        
    elif noise.ndim == 1: # 1D noise
        ax.plot(noise)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Noise Profile (1D)\nShape: {noise.shape}", fontsize=10)
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    

def plot_sample_comparison(samples_to_plot: List[Dict], 
                           slices_to_show: Optional[List[int]] = None):
    """
    Plots a comparison of axial image slices and their corresponding noise profiles for a list of samples.

    Args:
        samples_to_plot: A list of sample dictionaries to plot.
        slices_to_show: A list of integer slice indices to display. If None, defaults to first, middle, and last.
        save_path: Optional path to save the figure.
    """
    num_samples = len(samples_to_plot)
    
    # If no slices are specified, automatically select the first, middle, and last
    if slices_to_show is None:
        num_axial_slices = samples_to_plot[0]['image'].shape[2]
        slices_to_show = [0, num_axial_slices // 2, num_axial_slices - 1]
    
    num_slices = len(slices_to_show)
    
    fig, axes = plt.subplots(num_samples * 2, num_slices, figsize=(5 * num_slices, 4.5 * num_samples))
    fig.suptitle('Axial Slices and Noise Profiles', fontsize=16, fontweight='bold')

    for i, sample in enumerate(samples_to_plot):
        for j, slice_idx in enumerate(slices_to_show):
            # Plot Image (top row for this sample)
            img_ax = axes[i * 2, j]
            img_ax.imshow(sample['image'][:, :, slice_idx], cmap='gray')
            img_ax.set_title(f"{sample['subject_id']} | Slice {slice_idx + 1}", fontsize=11)
            img_ax.axis('off')

            # Plot Noise (bottom row for this sample)
            noise_ax = axes[i * 2 + 1, j]
            _plot_noise_on_axis(noise_ax, sample, slice_idx)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle

        
    plt.show()

def print_sample_stats(sample: Dict):
    """Prints detailed statistics for a single sample dictionary."""
    print("-" * 40)
    print(f" Statistics for Subject: {sample['subject_id']}")
    print(f"  Image Shape: {sample['image'].shape}")
    
    noise = sample.get('noise')
    if noise is not None:
        print(f"  Noise Shape: {noise.shape}")
        print(f"  Noise Range: [{noise.min():.4f}, {noise.max():.4f}]")
        print(f"  Noise Mean:  {noise.mean():.4f}")
        print(f"  Noise Std:   {noise.std():.4f}")
    else:
        print("  Noise: None")
    print("-" * 40)

