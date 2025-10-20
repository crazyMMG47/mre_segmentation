# posterior net 
from use_monai import extract_unet_decoder_blocks, extract_unet_encoder_blocks
from monai.networks.nets import UNet
from typing import Tuple, List, Optional
import torch 
import torch.nn as nn
from monai.netwprks.layers.simplelayers import SkipConnection
from models.components.noise_encoder import NoiseEncoder

# 3d posterior net 
class PosteriorNet(nn.Module):
    """
    Posterior network q(z|x, y) reusing MONAI's UNet encoder.
    """

    def __init__(
        self,
        image_channels: int,
        mask_channels: int,
        latent_dim: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256),
        num_res_units: int = 2,
        act="PRELU",
        norm="INSTANCE",
        dropout: float = 0.2, # TODO: adjust this dropout rate if needed 
    ):
        
        super().__init__()
        # this is different from the prior net
        # the posterior net takes in both the image and the mask as input
        # image_channels: number of channels in the input image (e.g., 1 for grayscale images)
        # mask_channels: number of channels in the input mask (e.g., 1 for binary masks)
        in_ch = image_channels + mask_channels
        # add one more last channel to the feature channels because the last layer is the bottleneck layer (similar to the prior net)
        channels = tuple(feature_channels) + (feature_channels[-1],)
        strides = tuple([2] * len(feature_channels))
        temp_unet = UNet(
            spatial_dims=spatial_dims, in_channels=in_ch, out_channels=1,
            channels=channels, strides=strides, num_res_units=num_res_units,
            act=act, norm=norm, dropout=dropout,
        )
        self.encoder = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        ConvNd = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.latent_head = ConvNd(feature_channels[-1], 2 * latent_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.latent_head.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, mask], dim=1)
        feats = inp
        for blk in self.encoder:
            feats = blk(feats)
        stats = self.latent_head(self.global_pool(feats)).flatten(1)
        # return two tensors:
        # 1. (B, mu)
        # 2. (B, latet_dim)
        # torch.chunk will split the pices into two
        return torch.chunk(stats, 2, dim=1)
    
   
# 2d posterior net 
class SliceWisePosteriorNet(nn.Module):
    """
    2D Posterior network that produces slice-specific latent distributions.
    Uses both features and mask to generate [B, D, Z] latents.
    """
    def __init__(self,
                 feature_channels: int,
                 mask_channels: int,
                 latent_dim: int,
                 spatial_dims: int = 3,
                 debug_checks: bool = True):
        super().__init__()
        assert spatial_dims == 3, "SliceWisePosteriorNet requires 3D input"
        
        # 2D encoder for slice features + mask
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(feature_channels + mask_channels, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.debug_checks = debug_checks
        
    def forward(self, 
                features: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, D, H, W] from UNet encoder
            mask: [B, M, D, H, W] ground truth segmentation
        Returns:
            mu: [B, D, Z]
            logvar: [B, D, Z]
        """
        B, C, D, H, W = features.shape
        
            
        # Concatenate features and mask
        x = torch.cat([features, mask], dim=1)  # [B, C+M, D, H, W]
        
        # Reshape to process all slices in parallel: [B*D, C+M, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, -1, H, W)

            
        # Encode all slices
        h = self.slice_encoder(x)  # [B*D, 256]
        
        # Generate latent parameters
        mu = self.fc_mu(h)         # [B*D, Z]
        logvar = self.fc_logvar(h) # [B*D, Z]
        
        # Reshape back to [B, D, Z]
        mu = mu.view(B, D, -1)
        logvar = logvar.view(B, D, -1)
        
        return mu, logvar
    
    
class SliceWiseNoisyPosterior(nn.Module):
    """
    2D Posterior network conditioned on image features, mask, AND noise profiles.
    Produces slice-specific latent distributions informed by all three sources.
    """
    def __init__(
        self,
        feature_channels: int,
        mask_channels: int,
        latent_dim: int,
        noise_feature_dim: int = 128,
        spatial_dims: int = 3,
        debug_checks: bool = True,
        use_noise: bool = True
    ):
        super().__init__()
        assert spatial_dims == 3, "SliceWiseNoisyPosterior requires 3D input"
        
        self.use_noise = use_noise
        self.noise_feature_dim = noise_feature_dim
        self.debug_checks = debug_checks
        
        # Shared noise encoder (should match the one in Prior)
        if self.use_noise:
            self.noise_encoder = NoiseEncoder(
                input_channels=1,
                base_channels=16,
                noise_feature_dim=noise_feature_dim
            )
        
        # 2D encoder for slice features + mask
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(feature_channels + mask_channels, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Fusion layer: combine image+mask features with noise features
        fusion_input_dim = 256 + (noise_feature_dim if self.use_noise else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.PReLU()
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, D, H, W] from UNet encoder
            mask: [B, M, D, H, W] ground truth segmentation
            noise: [B, 1, D, H, W] noise profiles (optional, if use_noise=True)
        Returns:
            mu: [B, D, Z] - mean of latent distribution per slice
            logvar: [B, D, Z] - log variance of latent distribution per slice
        """
        B, C, D, H, W = features.shape
        
        # Check noise input if required
        if self.use_noise:
            assert noise is not None, "Noise input required when use_noise=True"
            assert noise.shape == (B, 1, D, H, W), \
                f"Expected noise shape [B, 1, D, H, W], got {noise.shape}"
        
        # Concatenate features and mask
        x = torch.cat([features, mask], dim=1)  # [B, C+M, D, H, W]
        
        # Reshape to process all slices in parallel: [B*D, C+M, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, -1, H, W)
        
        # Encode image+mask slices
        image_mask_features = self.slice_encoder(x)  # [B*D, 256]
        
        # Process noise if available
        if self.use_noise and noise is not None:
            # Reshape noise: [B*D, 1, H, W]
            noise_slices = noise.permute(0, 2, 1, 3, 4).contiguous()
            noise_slices = noise_slices.view(B * D, 1, H, W)
            
            # Encode noise
            noise_features = self.noise_encoder(noise_slices)  # [B*D, noise_feature_dim]
            
            # Concatenate all features
            combined = torch.cat([image_mask_features, noise_features], dim=1)  # [B*D, 256 + noise_feature_dim]
        else:
            combined = image_mask_features
        
        # Fuse features
        fused = self.fusion(combined)  # [B*D, 256]
        
        # Generate latent distribution parameters
        mu = self.fc_mu(fused)         # [B*D, Z]
        logvar = self.fc_logvar(fused) # [B*D, Z]
        
        # Reshape back to [B, D, Z]
        mu = mu.view(B, D, -1)
        logvar = logvar.view(B, D, -1)
        
        if self.debug_checks:
            print(f"[NoisyPosteriorNet] Output shapes - mu: {mu.shape}, logvar: {logvar.shape}")
            if self.use_noise:
                print(f"[NoisyPosteriorNet] Noise conditioning: ENABLED")
            else:
                print(f"[NoisyPosteriorNet] Noise conditioning: DISABLED")
        
        return mu, logvar

