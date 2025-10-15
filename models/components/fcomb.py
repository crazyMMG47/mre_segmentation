from use_monai import extract_unet_decoder_blocks, extract_unet_encoder_blocks
from monai.networks.nets import UNet
from typing import Tuple, List, Optional
import torch 
import torch.nn as nn
from monai.netwprks.layers.simplelayers import SkipConnection


class Fcomb(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int,
                 seg_out_channels: int, spatial_dims: int = 3,
                 hidden_ch: int = 64,     # you can reuse feature_channels[0]
                 n_layers: int = 3,
                 drop_p: float = 0.4, # ← dropout prob
                 inject_latent: bool = True):   # add a flag here 
        """
        note:
        hidden_ch: number of intermediate channelsin the 1*1 convolution layers inside Fcomb. 
        
        
        """
        super().__init__()
        self.inject_latent = inject_latent
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
    
        Drop = nn.Dropout3d if spatial_dims == 3 else nn.Dropout2d

        if inject_latent:
            in_feats = in_ch + latent_dim  # after concat
            layers = []
            for _ in range(n_layers - 1):
                layers += [
                    Conv(in_feats, hidden_ch, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    Drop(p=drop_p)                 # TODO: mofify the dropout rate ← dropout after each 1×1 conv
                ]
                in_feats = hidden_ch

            layers += [Conv(hidden_ch, seg_out_channels, kernel_size=1, bias=True)]
            # the fcomb contains:
            # 1. a series of 1x1 convolutions with ReLU activation and dropout
            # 2. a final 1x1 convolution that maps to the output channels (seg_out_channels)
            #    which is the number of classes in the segmentation task
            # The final output will have the shape (B, seg_out_channels, H, W, D) or (B, seg_out_channels, H, W)
            # depending on the spatial dimensions.
            self.fcomb = nn.Sequential(*layers)
            
        else:
            # deterministic 1x1 conv, no z used
            self.fcomb = Conv(in_ch, seg_out_channels, kernel_size=1, bias=True)
            
    def forward(self, feat, z):
        """
        Args:
        -- feat: tensor of shape (B, C, H, W, D) or (B, C, H, W)
        -- z: tensor of shape (B, latent_dim) or (B, latent_dim, 1, 1, 1) or (B, latent_dim, 1, 1)
        """
        # TODO: uncomment the print statements for debugging
        # print(f"[Fcomb] Input feat shape: {feat.shape}")
        # Input feat shape: torch.Size([8, 32, 128, 128, 64])
        # print(f"[Fcomb] Latent z shape before reshape: {z.shape}")
        #  Latent z shape before reshape: torch.Size([8, 32])
        if not self.inject_latent:
            return self.fcomb(feat)  # if not injecting latent, just return the feature through 1x1 conv
        
        # tile z to match satial dim 
        # tile z to H×W (or D×H×W) and concat
        while z.dim() < feat.dim():
            # z shape : (B, latent_dim)
            # below expands z to have the same shape as the feature (B, latent_dim, 1, 1, 1) 
            z = z.unsqueeze(-1)
        
        # now they have the same shape 
        # we need to expand across the spatial dimension
        # for example: initial feat = (2,32,4,4,4) but z = (2,8) which 8 being the latent dimension
        # we need to broadcast z across spatial dimensions  and it will become (2,8,4,4,4)
        z = z.expand_as(feat[:, :z.size(1)])    # broadcast over spatial dims
        # x.shape → (2, 40, 4, 4, 4)   # 32 from feat + 8 from z
        x = torch.cat([feat, z], dim=1)
        return self.fcomb(x)
    
    

class SliceWiseFcomb(nn.Module):
    """
    Modified Fcomb that injects different latent vectors per slice.
    """
    def __init__(self,
                 in_ch: int,
                 latent_dim: int,
                 seg_out_channels: int = 1,
                 spatial_dims: int = 3,
                 inject_latent: bool = True):
        super().__init__()
        self.inject_latent = inject_latent
        self.spatial_dims = spatial_dims
        
        if inject_latent:
            # 2D conv for per-slice injection
            self.latent_injection = nn.Conv2d(
                in_ch + latent_dim, 
                in_ch, 
                kernel_size=1
            )
        
        # Final segmentation layers (can be 2D or 3D)
        if spatial_dims == 3:
            self.seg_layer = nn.Sequential(
                nn.Conv3d(in_ch, in_ch // 2, kernel_size=3, padding=1),
                nn.InstanceNorm3d(in_ch // 2),
                nn.PReLU(),
                nn.Conv3d(in_ch // 2, seg_out_channels, kernel_size=1),
            )
        else:
            self.seg_layer = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(in_ch // 2),
                nn.PReLU(),
                nn.Conv2d(in_ch // 2, seg_out_channels, kernel_size=1),
            )
    
    def forward(self, 
                feat: torch.Tensor, 
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat: [B, C, D, H, W] from UNet
            z: [B, D, Z] slice-specific latents (or None for deterministic)
        Returns:
            logits: [B, seg_out_channels, D, H, W]
        """
        if not self.inject_latent or z is None:
            # Deterministic mode
            return self.seg_layer(feat)
        
        # Inject latent per slice
        B, C, D, H, W = feat.shape
        _, _, Z = z.shape
        
        # add sanity checks
        if z.shape[0] != B or z.shape[1] != D:
            raise ValueError(
                f"[Fcomb] z shape {tuple(z.shape)} must be [B,D,Z] matching feat [B,*,D,*,*]={feat.shape}."
            )
            
        # Process each slice
        feat_injected = feat.clone()
        for d in range(D):
            # Broadcast z to spatial dimensions [B, Z, H, W]
            z_spatial = z[:, d, :, None, None].expand(B, Z, H, W)
            
            # Concatenate and inject
            feat_with_z = torch.cat([feat[:, :, d], z_spatial], dim=1)  # [B, C+Z, H, W]
            feat_injected[:, :, d] = self.latent_injection(feat_with_z)  # [B, C, H, W]
        
        # Final segmentation
        logits = self.seg_layer(feat_injected)
        return logits
    
    

    
