# Helper functions to extract unet blocks
# We will be directly using the MONAI UNet implementation 
from monai.networks.nets import UNet
from typing import Tuple, List, Optional
import torch 
import torch.nn as nn
from monai.netwprks.layers.simplelayers import SkipConnection

def extract_unet_encoder_blocks(unet: UNet) -> List[nn.Module]:
    """
    Recursively traverses a MONAI UNet to extract the encoder (downsampling) blocks
    in top-down order (from shallowest to deepest).
    """
    blocks = []
    current_level = unet.model
    # The UNet model is a recursive structure of Sequential modules.
    # Each level contains an encoder block, a SkipConnection, and a decoder block.
    while isinstance(current_level, nn.Sequential) and isinstance(current_level[1], SkipConnection):
        blocks.append(current_level[0])
        current_level = current_level[1].submodule
    # The final level is the bottleneck block.
    blocks.append(current_level)
    return blocks


def extract_unet_decoder_blocks(unet: UNet) -> List[nn.Module]:
    """
    Recursively traverses a MONAI UNet to extract the decoder (upsampling) blocks
    in bottom-up order (from deepest to shallowest).
    """
    blocks = []
    def _traverse(module):
        # A MONAI UNet level is a Sequential of [encoder_block, SkipConnection, decoder_block]
        if isinstance(module, nn.Sequential) and isinstance(module[1], SkipConnection):
            # Recurse to the deepest part of the network first.
            _traverse(module[1].submodule)
            # Add the decoder block (up_layer) of the current level on the way back up.
            blocks.append(module[2])
    
    _traverse(unet.model)
    return blocks