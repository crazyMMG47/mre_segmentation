# import libraries
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from monai.networks.nets import UNet
from monai.networks.layers.simplelayers import SkipConnection
from torch.cuda.amp import autocast,GradScaler
from monai.metrics import DiceMetric
import torch.nn.functional as F

# import building blocks
from prior import SliceWisePriorNet
from posterior import SliceWisePosteriorNet
from fcomb import SliceWiseFcomb


class SliceWiseProbUNet(nn.Module):
    """
    Slice-wise Probabilistic U-Net where latent vectors are generated and injected
    independently per slice to enable slice-specific uncertainty modeling.
    
    Uses 3D UNet backbone for feature extraction (preserves inter-slice context)
    but generates and injects 2D slice-specific latents.
    
    Put inject_latent to be False when you want the unet in deterministic mode. 
    """
    def __init__(self,
                 image_channels: int,
                 mask_channels: int,
                 latent_dim: int,
                 feature_channels: Tuple[int, ...],
                 num_res_units: int,
                 act: str = "PRELU",
                 norm: str = "INSTANCE",
                 dropout: float = 0.2,
                 spatial_dims: int = 3, # volume input
                 seg_out_channels: int = 1,
                 inject_latent: bool = True):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.inject_latent = inject_latent
        self.spatial_dims = spatial_dims
        self._shape_logged = False  # for one-time logging of tensor shapes
        
        assert spatial_dims == 3, "SliceWise model requires 3D input"

        # 3D UNet backbone (preserves inter-slice context)
        # This is DEPTH-AWARE UNET since it is 3D! 
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=image_channels,
            out_channels=feature_channels[0],
            channels=feature_channels,
            strides=tuple([2] * len(feature_channels)),
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        
        # Slice-wise Fcomb
        self.fcomb = SliceWiseFcomb(
            in_ch=feature_channels[0],
            latent_dim=latent_dim,
            seg_out_channels=seg_out_channels,
            spatial_dims=spatial_dims,
            inject_latent=inject_latent,
        )

        # Slice-wise Prior and Posterior networks
        if inject_latent:
            print("Slice-wise Probabilistic U-Net enabled. Per-slice latent injection active.")
            
            self.prior_net = SliceWisePriorNet(
                feature_channels=feature_channels[0],  # UNet output channels
                latent_dim=latent_dim,
                spatial_dims=spatial_dims
            )
            
            self.post_net = SliceWisePosteriorNet(
                feature_channels=feature_channels[0],
                mask_channels=mask_channels,
                latent_dim=latent_dim,
                spatial_dims=spatial_dims
            )

    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                sample_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] input images
            mask: [B, M, D, H, W] ground truth (training only)
            sample_z: [B, D, Z] optional pre-sampled latents
        Returns:
            Training: (logits, (mu_p, logvar_p), (mu_q, logvar_q))
            Inference: logits
        """
        # DEBUG
        # print(f"\n=== SliceWiseProbUNet Forward ===")
        # print(f"Input x shape: {x.shape}")
        # if mask is not None:
        #     print(f"Input mask shape: {mask.shape}")
    
        # Extract features with 3D UNet (preserves inter-slice context)
        # Depth-aware UNet Backbone
        feat = self.unet(x)  # [B, C, D, H, W]
        # print(f"After UNet feat shape: {feat.shape}")
        
        # TODO: Debug dimension match
        if not self._shape_logged:
            # print("feat shape:", tuple(feat.shape))  # should be [B,C,D,H,W]
            self._shape_logged = True
            
        # Deterministic mode
        if not self.inject_latent:
            logits = self.fcomb(feat, None)
            return logits
        
        # Probabilistic mode
        if self.training:
            assert mask is not None, "Mask must be provided for training."
            # print("Helen's second check, mask shape before ensure:", mask.shape)
            # print("Helen's second check, feat shape before ensure:", feat.shape)
            # Get slice-wise prior and posterior distributions
            mu_p, logvar_p = self.prior_net(feat)      # [B, D, Z]
            mu_q, logvar_q = self.post_net(feat, mask) # [B, D, Z]
            
            # Reparameterization trick for q(z|x,y)
            std_q = torch.exp(0.5 * logvar_q)
            # sample latent z from a Gaussian while keeping the graph different 
            z = mu_q + std_q * torch.randn_like(std_q)  # [B, D, Z]
            
        else:
            # Inference: sample from prior unless z is provided
            mu_p, logvar_p = self.prior_net(feat)  # [B, D, Z]
            
            if sample_z is None:
                std_p = torch.exp(0.5 * logvar_p)
                z = mu_p + std_p * torch.randn_like(std_p)  # [B, D, Z]
            else:
                z = sample_z
            
            mu_q = logvar_q = None  # No posterior at test time
        
        # Inject slice-specific latents and decode
        # TODO: Ensure the accuracy of this 
        logits = self.fcomb(feat, z)  # [B, seg_out_channels, D, H, W], z(d) only conditions feat[:, d]
        
        if self.training:
            return logits, (mu_p, logvar_p), (mu_q, logvar_q)
        else:
            return logits
        