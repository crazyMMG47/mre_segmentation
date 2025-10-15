# prior net 
from use_monai import extract_unet_decoder_blocks, extract_unet_encoder_blocks
from monai.networks.nets import UNet
from typing import Tuple, List, Optional
import torch 
import torch.nn as nn
from monai.netwprks.layers.simplelayers import SkipConnection


# global 3D prior net 
class PriorNet(nn.Module):
    """
    Prior network p(z|x) reusing MONAI's UNet encoder
    
    The prior net is part of a variational model and it will estimate prior distribution p(z|x) over latent variable z. 
    """
    # initialize the PriorNet
    # input_channels: number of input channels (e.g., 1 for grayscale images)
    # latent_dim: dimension of the latent space
    # spatial_dims: number of spatial dimensions (2 or 3)
    # feature_channels: tuple of feature channels for each level in the UNet
    # num_res_units: number of residual units in each block
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        spatial_dims: int = 3, # we are using "3" because we have 3D data
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256), # unless specified, these will be used 
        num_res_units: int = 2, 
        act="PRELU", # parametric ReLU activation function
        norm="INSTANCE",
        dropout: float = 0.2, # TODO: adjust this accordingly! 
        # set for regularization purposes, add stochasticity to the model
        # this dropout only happens in training, not in inference
    ):
        
        super().__init__()
        self.latent_dim = latent_dim
        
        # Appending the last feature channel to the channels tuple
        # e.g. if the original feature channels are: feature_channels = (32, 64, 128, 256)
        # then the channels will be: channels = (32, 64, 128, 256, 256)
        # Adding an extra depth without increasing the no. channels BECAUSE the bottom layer is the bottleneck layer 
        channels = tuple(feature_channels) + (feature_channels[-1],)
        
        # we will not apply the stride to the bottleneck layer
        # And we are using stride of 2 for all other layers --> halving the dimension at each level 
        strides = tuple([2] * len(feature_channels))
        
        # initialize the MONAI's unet with the specified parameters 
        temp_unet = UNet(
            spatial_dims=spatial_dims, in_channels=input_channels, out_channels=1,
            channels=channels, strides=strides, num_res_units=num_res_units,
            act=act, norm=norm, dropout=dropout,
        )
        
        # UNET encoder
        self.encoder = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        
        # nn.Conv3d
        ConvNd = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        # global average pooling (i.e. (B, 256, 10, 10, 5) → (B, 256, 1, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        
        # 1*1*1 convolution, maps pooled features to a vector of size 2 * latent_dim
        # double the size of the tensor shape because we needs to fit both mu and log var in one forward pass
        self.latent_head = ConvNd(feature_channels[-1], 2 * latent_dim, kernel_size=1, bias=True)
        # initialize the bias term of the final conv layer to zero 
        # we are predicting the mean µ and log-var 
        # zeroing out gives µ = 0 and log var = 0 --> sigma = 1 
        # which means the prior distribution is z ~ N (0, 1)
        # This give the NN a neutral starting point 
        nn.init.zeros_(self.latent_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method returns the mean and log-variance of a Gaussian latent distribution inferred from input x.
        
        x will take in the shape (B, C, H, W, D) which is a 3d image batch. 
        
        
        """
        feats = x
        # apply all encoder blocks to the input "x" --> extract encoder block from temp unet 
        for blk in self.encoder:
            feats = blk(feats)
        # apply global average pooling to the features 
        # latent head (1*1 convolution) + fltusiatten 
        stats = self.latent_head(self.global_pool(feats)).flatten(1)
        # return two tensors:
        # 1. (B, mu)
        # 2. (B, latet_dim)
        # torch.chunk will split the pices into two 
        return torch.chunk(stats, 2, dim=1)
    

# per-slice 2D prior net 
class SliceWisePriorNet(nn.Module):
    """
    2D Prior network that produces slice-specific latent distributions.
    Processes features from 3D UNet encoder to generate [B, D, Z] latents.
    """
    def __init__(self, 
                 feature_channels: int,
                 latent_dim: int,
                 spatial_dims: int = 3,
                 debug_checks: bool = True):
        super().__init__()
        assert spatial_dims == 3, "SliceWisePriorNet requires 3D input"
        
        # 2D encoder for slice features
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, 256, kernel_size=3, padding=1),
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
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, D, H, W] from UNet encoder
        Returns:
            mu: [B, D, Z]
            logvar: [B, D, Z]
        """
        B, C, D, H, W = features.shape
        
        if self.debug_checks:
            _assert_slice_order_preserved("PriorNet", features)
            
        # Reshape to process all slices in parallel: [B*D, C, H, W]
        x = features.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, C, H, W)
        
        if self.debug_checks:
            _check_BD_match("PriorNet", B, D, x)
            
        # Encode all slices
        h = self.slice_encoder(x)  # [B*D, 256]
        
        # Generate latent parameters
        mu = self.fc_mu(h)         # [B*D, Z]
        logvar = self.fc_logvar(h) # [B*D, Z]
        
        # Reshape back to [B, D, Z]
        mu = mu.view(B, D, -1)
        logvar = logvar.view(B, D, -1)
        
        return mu, logvar
    
    
# Helper function 
@torch.no_grad()
def _assert_slice_order_preserved(where: str, feats: torch.Tensor):
    # Run in full precision for the check
    with torch.cuda.amp.autocast(enabled=False):
        f = feats.float()
        B, C, D, H, W = f.shape  # assumes already [B,C,D,H,W]
        orig = f.mean(dim=(1,3,4))  # [B,D]
        tmp  = f.permute(0,2,1,3,4).contiguous().view(B*D, C, H, W)
        rec  = tmp.view(B, D, C, H, W).mean(dim=(2,3,4))  # [B,D]
        if not torch.allclose(orig, rec, rtol=1e-3, atol=1e-4):
            raise RuntimeError(
                f"[{where}] Slice order check failed: stats changed after permute/view roundtrip. "
                "Likely wrong spatial order (HWD vs DHW)."
            )
            

def _check_BD_match(where: str, B: int, D: int, x2d: torch.Tensor):
    if x2d.size(0) != B * D:
        raise RuntimeError(f"[{where}] Flattened N={x2d.size(0)} != B*D={B*D}. "
                           "Depth batching misaligned (wrong spatial order?).")
        _check_BD_match
