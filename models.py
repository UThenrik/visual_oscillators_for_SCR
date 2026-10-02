import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from torch.nn.modules.loss import _Loss

# Constants for harmonic oscillator
EPS = 1e-8

def softplus_pos(x, min_val=1e-6):
    """Softplus function with minimum value to ensure positivity"""
    return F.softplus(x) + min_val

def create_encoder(input_channels=1, latent_dim=16, is_vae=True, image_size=32):
    """
    Simple encoder architecture with 4x4 kernels and stride=2.
    Supports 32x32, 64x64, 96x96, 128x128 inputs by adding more conv layers.
    """
    # Determine number of conv layers based on image size
    if image_size <= 32:
        num_conv_layers = 3  # 32->16->8->4
    elif image_size <= 64:
        num_conv_layers = 4  # 64->32->16->8->4
    elif image_size <= 96:
        num_conv_layers = 4  # 96->48->24->12->6, but we'll use adaptive pooling to get to 4x4
    else:  # 128x128
        num_conv_layers = 5  # 128->64->32->16->8->4
    
    layers = []
    in_channels = input_channels
    
    # Conv layers: 4x4 kernel, stride=2
    for i in range(num_conv_layers):
        out_channels = 32 * (2 ** i)  # 32, 64, 128, 256, 512
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])
        in_channels = out_channels
    
    # For 96x96, we need adaptive pooling to get to 4x4
    if image_size == 96:
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
    
    # Flatten and dense layers
    layers.extend([
        nn.Flatten(),
        nn.Linear(in_channels * 4 * 4, latent_dim * (2 if is_vae else 1))
    ])
    
    return nn.Sequential(*layers)


def create_decoder(latent_dim=16, output_channels=1, image_size=32):
    """
    Simple decoder architecture with 4x4 kernels and stride=2.
    Supports 32x32, 64x64, 96x96, 128x128 inputs by adding more deconv layers.
    """
    # Determine number of deconv layers based on image size
    if image_size <= 32:
        num_deconv_layers = 3  # 4->8->16->32
    elif image_size <= 64:
        num_deconv_layers = 4  # 4->8->16->32->64
    elif image_size <= 96:
        num_deconv_layers = 4  # 4->8->16->32->64, then adaptive upsampling to 96x96
    else:  # 128x128
        num_deconv_layers = 5  # 4->8->16->32->64->128
    
    layers = []
    
    # Dense layer to get to 4x4 feature map
    # Determine the number of channels from the encoder
    if image_size <= 32:
        encoder_channels = 128  # 3 conv layers: 32->64->128
    elif image_size <= 64:
        encoder_channels = 256  # 4 conv layers: 32->64->128->256
    elif image_size <= 96:
        encoder_channels = 256  # 4 conv layers: 32->64->128->256 (same as 64x64)
    else:  # 128x128
        encoder_channels = 512  # 5 conv layers: 32->64->128->256->512
    
    layers.extend([
        nn.Linear(latent_dim, encoder_channels * 4 * 4),
        nn.Unflatten(1, (encoder_channels, 4, 4))
    ])
    
    # Deconv layers: 4x4 kernel, stride=2
    in_channels = encoder_channels
    for i in range(num_deconv_layers):
        if i == num_deconv_layers - 1:  # Last layer
            out_channels = output_channels
            activation = nn.Sigmoid()  # Sigmoid for final output
        else:
            out_channels = in_channels // 2  # 512->256->128->64->32
            activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        layers.extend([
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            activation
        ])
        in_channels = out_channels
    
    # For 96x96, we need adaptive upsampling to get to the correct size
    if image_size == 96:
        layers.append(nn.AdaptiveAvgPool2d((96, 96)))
    
    return nn.Sequential(*layers)


def create_broadcast_decoder(latent_dim=16, output_channels=1, image_size=32, feature_dim=16, background_token_value=0.0, attention_downsample_factor=1, dynamics_type="harmonic1d", gumbel_noise_strength=0.0):
    """
    Broadcast decoder that broadcasts latent dimensions to image dimensions.
    
    Args:
        latent_dim: Dimension of latent space
        output_channels: Number of output channels
        image_size: Size of the output image (assumed square)
        feature_dim: Dimension of feature vectors for each latent dimension
        background_token_value: Value for background attention tokens (default 0.0)
        attention_downsample_factor: Downsampling factor for attention computation
        dynamics_type: Type of dynamics ("harmonic" or "2dharmonic")
        gumbel_noise_strength: Strength of Gumbel noise for sharper attention (0.0 = disabled, 0.1-0.5 = typical range)
    
    Returns:
        AttentionBroadcastDecoder: Decoder with attention mechanism
    """
    return AttentionBroadcastDecoder(latent_dim, output_channels, image_size, feature_dim, background_token_value, attention_downsample_factor, dynamics_type, gumbel_noise_strength)


def attention_com_positions(attention_weights):
    """
    Compute center-of-mass positions for attention maps.

    Args:
        attention_weights: [batch, n_nodes, H, W] - attention maps

    Returns:
        com_positions: [batch, n_nodes, 2] - (y, x) center of mass coordinates
    """
    batch, n_nodes, H, W = attention_weights.shape
    device = attention_weights.device

    attn_sq = attention_weights ** 2  # emphasize peaks

    # Coordinate grids in [-1,1]
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H,W]
    y_grid = y_grid.view(1, 1, H, W)
    x_grid = x_grid.view(1, 1, H, W)

    attn_sum = attn_sq.sum(dim=(2,3), keepdim=True) + 1e-8

    com_y = (attn_sq * y_grid).sum(dim=(2,3), keepdim=True) / attn_sum
    com_x = (attn_sq * x_grid).sum(dim=(2,3), keepdim=True) / attn_sum

    com_positions = torch.cat([com_y, com_x], dim=-1).squeeze(-2).squeeze(-2)
    return com_positions


def attention_com_velocities(attention_weights, attention_dot):
    """
    Compute center-of-mass velocities for attention maps using the quotient rule.

    Args:
        attention_weights: [batch, n_nodes, H, W] - attention maps
        attention_dot: [batch, n_nodes, H, W] - JVP / velocity of attention maps

    Returns:
        com_velocities: [batch, n_nodes, 2] - (y, x) velocities of COM
    """
    batch, n_nodes, H, W = attention_weights.shape
    device = attention_weights.device

    attn_sq = attention_weights ** 2
    attn_dot_sq = 2 * attention_weights * attention_dot

    # Coordinate grids in [-1,1]
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    y_grid = y_grid.view(1, 1, H, W)
    x_grid = x_grid.view(1, 1, H, W)

    sum_attn = attn_sq.sum(dim=(2,3), keepdim=True) + 1e-8
    sum_attn_dot = attn_dot_sq.sum(dim=(2,3), keepdim=True)

    com_dot_y = ((attn_dot_sq * y_grid).sum(dim=(2,3), keepdim=True) * sum_attn -
                 (attn_sq * y_grid).sum(dim=(2,3), keepdim=True) * sum_attn_dot) / (sum_attn ** 2)

    com_dot_x = ((attn_dot_sq * x_grid).sum(dim=(2,3), keepdim=True) * sum_attn -
                 (attn_sq * x_grid).sum(dim=(2,3), keepdim=True) * sum_attn_dot) / (sum_attn ** 2)

    com_velocities = torch.cat([com_dot_y, com_dot_x], dim=-1).squeeze(-2).squeeze(-2)
    return com_velocities


def com_jacobian_per_node(vae, z, *, device=None):
    """Per-node Jacobian J[i] = d COM_i / d z_i for image-space force arrows (RA-L method).

    COM is (y, x) in [-1, 1]. Latent oscillator i uses z[2i:2i+2] when dim_per_attention==2.
    Off-diagonal blocks d COM_i / d z_j (j!=i) are discarded: we want the local map for node i.

    Returns:
        J: ndarray [n_nodes, 2, 2]
    """
    decoder = vae.decoder
    if device is None:
        device = z.device if torch.is_tensor(z) else next(decoder.parameters()).device
    if torch.is_tensor(z):
        z_flat = z.detach().reshape(-1).to(device=device, dtype=torch.float32)
    else:
        z_flat = torch.as_tensor(np.asarray(z), device=device, dtype=torch.float32).reshape(-1)
    n_nodes = decoder.latent_dim // decoder.dim_per_attention
    dpa = decoder.dim_per_attention
    was_training = decoder.training
    decoder.eval()

    def com_fn(z_in):
        attn = decoder.get_attention_weights(
            z_in.unsqueeze(0), return_background_weights=False, return_peak_location=False, upsample=True
        )
        if isinstance(attn, tuple):
            attn = attn[0]
        return attention_com_positions(attn)[0]  # [n_nodes, 2] (y, x)

    try:
        try:
            J_full = torch.autograd.functional.jacobian(
                com_fn, z_flat, create_graph=False, vectorize=True
            )
        except Exception:
            J_full = torch.autograd.functional.jacobian(
                com_fn, z_flat, create_graph=False, vectorize=False
            )
        # J_full: [n_nodes, 2, latent_dim]
        J = torch.zeros(n_nodes, 2, dpa, device=device, dtype=z_flat.dtype)
        for i in range(n_nodes):
            J[i] = J_full[i, :, i * dpa:(i + 1) * dpa]
        return J.detach().cpu().numpy()
    finally:
        decoder.train(was_training)


def latent_forces_to_image_jacobian(forces_lat, J):
    """Map per-node latent forces to image COM via the RA-L pushforward F_img = J @ F.

    This is the same map as the attention-coupling velocity relation ṗ = J q̇
    (RA-L: J_l = ∂p_l/∂q_l through squared-attention COM). It is a vector
    pushforward, not the cotangent map J^{-T} used for generalized forces.

    Args:
        forces_lat: [n_nodes, 2]  q_l = (z[2l], z[2l+1])
        J: [n_nodes, 2, 2]        rows = (∂com_y, ∂com_x)
    Returns:
        F_img: [n_nodes, 2] in (com_y, com_x), both in [-1, 1]
    """
    forces_lat = np.asarray(forces_lat)
    J = np.asarray(J)
    return np.einsum("nij,nj->ni", J, forces_lat)


def image_force_arrow_xy(F_img, img_h, img_w):
    """COM-space force (d_com_y, d_com_x) → imshow arrow (dx, dy) in pixels.

    attention_com_positions uses y = linspace(-1, 1) over rows (top → bottom)
    and x = linspace(-1, 1) over columns (left → right). imshow origin='upper'
    uses the same convention: +dx right, +dy down. Scale by (extent)/2 so a
    unit COM displacement spans the image.
    """
    F_img = np.asarray(F_img, dtype=np.float64)
    sx = (float(img_w) - 1.0) / 2.0
    sy = (float(img_h) - 1.0) / 2.0
    dx = F_img[..., 1] * sx
    dy = F_img[..., 0] * sy
    return dx, dy


def tip_mid_base_oscillator_indices(vae, z, *, device=None):
    """Return (tip, mid, base) oscillator indices from attention COM y.

    Camera frames hang the robot from the fixture at the *top* of the image:
    COM is (y, x) in [-1, 1] with smaller y = top of the image = base (mount)
    and larger y = bottom of the image = tip (free end). Mid is the median
    oscillator along that vertical order.
    """
    decoder = vae.decoder
    if device is None:
        device = z.device if torch.is_tensor(z) else next(decoder.parameters()).device
    if torch.is_tensor(z):
        z_t = z.detach().reshape(-1).to(device=device, dtype=torch.float32)
    else:
        z_t = torch.as_tensor(np.asarray(z), device=device, dtype=torch.float32).reshape(-1)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    n_nodes = decoder.latent_dim // decoder.dim_per_attention
    with torch.no_grad():
        attn_out = decoder.get_attention_weights(
            z_t, return_background_weights=True, return_peak_location=True
        )
        peak = attn_out[2] if isinstance(attn_out, tuple) and len(attn_out) == 3 else attn_out[-1]
    com = peak[0, :n_nodes].detach().cpu().numpy()
    order = np.argsort(com[:, 0])  # image top -> bottom
    base = int(order[0])           # top of image = mount
    mid = int(order[len(order) // 2])
    tip = int(order[-1])           # bottom of image = free end
    return tip, mid, base


class AttentionBroadcastDecoder(nn.Module):
    """
    Broadcast decoder with attention mechanism that learns spatial attention weights
    for each latent dimension at each pixel location.
    """
    def __init__(self, latent_dim=16, output_channels=1, image_size=32, feature_dim=16, background_token_value=0.0, attention_downsample_factor=1, dynamics_type="harmonic1d", gumbel_noise_strength=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.feature_dim = feature_dim
        self.background_token_value = background_token_value
        self.attention_downsample_factor = attention_downsample_factor
        self.gumbel_noise_strength = gumbel_noise_strength  # Manual hyperparameter for attention sharpness

        if "harmonic2d" in dynamics_type:
            self.dim_per_attention = 2
        else:
            self.dim_per_attention = 1 

        if self.dim_per_attention == 2:

            #self.coord_transform = nn.Parameter(torch.eye(2))  # shape [2,2]
            self.coord_scale = nn.Parameter(torch.ones(2))
            self.coord_rotation = nn.Parameter(torch.tensor(0.0))  # rotation angle in radians
            self.coord_bias = nn.Parameter(torch.zeros(1, 1, 2))  # broadcast over batch and nodes

        
        self.attention_size = image_size // attention_downsample_factor
        # Attention mechanism with coordinate concatenation
        # Input dimension is latent + coordinates (no background tokens in attention computation)
        # Output dimension includes background tokens if enabled

        output_dim = latent_dim // self.dim_per_attention

        self.attention_net = nn.Sequential(


            # First 1x1 conv: (latent_dim + 2) -> 128 (latent + coords, no background tokens)
            nn.Conv2d(latent_dim + 2, 128, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Third 1x1 conv: 128 -> 64
            nn.Conv2d(128, 64, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Fourth 1x1 conv: 64 -> 64
            nn.Conv2d(64, 64, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(64, output_dim, kernel_size=1, padding='same')
        )
        
        # Learnable background tokens with spatial dimensions (not used for attention weights, only for features)
        # Always use 1 background token
        self.background_features = nn.Parameter(torch.randn(1, feature_dim*self.dim_per_attention, self.attention_size, self.attention_size))
        
        # Individual expanders for each latent dimension
        self.latent_expanders = nn.ModuleList([
            nn.Linear(1, feature_dim) for _ in range(latent_dim)
        ])
        # No background expanders needed - background features are predicted directly
        
        # Main decoder: 1x1 conv layers after attention
        # Input channels include feature dimension + 2 coordinate channels
        decoder_input_channels = feature_dim*self.dim_per_attention + 2  # feature_dim + 2 for coordinate channels
        
            # First conv: feature_dim (+ 2 coords) -> 64 channels, maintains spatial dimensions

            # Always use 3 layers: if no downsampling, 3x 1x1 conv; if 2x or 4x downsampling, use 1 or 2 convtranspose2d at the end
        if self.attention_downsample_factor == 1:
            # No upsampling, just 3x 1x1 conv
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_input_channels, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        elif self.attention_downsample_factor == 2:
            # 2x upsampling: 2x 1x1 conv, then 1x convtranspose2d
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_input_channels, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        elif self.attention_downsample_factor == 4:
            # 4x upsampling: 1x 1x1 conv, then 2x convtranspose2d
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_input_channels, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            raise ValueError("Only attention_downsample_factor of 1, 2, or 4 is supported.")

        self.decoder.append(nn.Conv2d(64, output_channels, kernel_size=1, padding='same'))
        self.decoder.append(nn.Sigmoid())

        
        # Initialize all parameters with Xavier uniform
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Kaiming (He) initialization with Leaky ReLU support."""
        negative_slope = 0.2  # adjust this value as needed for your Leaky ReLU

        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=negative_slope)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=negative_slope)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=negative_slope)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_attention_weights(self, z, return_background_weights=False, return_peak_location=False, upsample=True):
        """
        Get attention weights for a given batch of latent states.
        
        Args:
            z: Latent tensor of shape [batch, latent_dim]
            return_background_weights: If True, also return background attention weights
            return_peak_location: If True, also return peak location coordinates for each latent dimension
        
        Returns:
            If return_background_weights=False and return_peak_location=False: attention_weights [batch, latent_dim, H, W]
            If return_background_weights=True and return_peak_location=False: (attention_weights, background_weights)
            If return_peak_location=True: (attention_weights, peak_location) where peak_location is [batch, latent_dim, 2] (y, x coordinates)
            If both flags are True: (attention_weights, background_weights, peak_location)
        """
        batch_size = z.shape[0]
        
        # Create coordinate grid for attention computation
        y_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        x_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)  # [1, 2, H, W]
        coords = coords.expand(batch_size, -1, -1, -1)  # [batch, 2, H, W]
        
        # Broadcast latent state to match spatial dimensions
        z_broadcast = z.view(batch_size, self.latent_dim, 1, 1).expand(-1, -1, self.attention_size, self.attention_size)
        
        # Concatenate latent state with coordinates
        latent_with_coords = torch.cat([z_broadcast, coords], dim=1)  # [batch, latent_dim + 2, H, W]
        
        # Compute attention logits (background tokens are NOT included in attention computation)
        # The attention network computes weights based only on latent state + coordinates
        
        attention_logits = self.attention_net(latent_with_coords)  # [batch, latent_dim, H, W]

        # Extend the attention logits by concatenating zeros for background tokens along the channel (latent) dimension
        # Always use 1 background token
        zeros_bg = self.background_token_value*torch.ones(
            attention_logits.shape[0],  # batch size
            1,  # Always 1 background token
            attention_logits.shape[2],  # H
            attention_logits.shape[3],  # W
            device=attention_logits.device,
            dtype=attention_logits.dtype
        )

        attention_logits = torch.cat([attention_logits, zeros_bg], dim=1) # experimental
        # latent_logits = self.attention_net(latent_with_coords)[:, :self.latent_dim, :, :]
        # bg_logits = self.attention_net(latent_with_coords.detach())[:, self.latent_dim:, :, :]

        # # combine
        # attention_logits = torch.cat([latent_logits, bg_logits], dim=1)

        # Add adaptive Gumbel noise during training for sharper, more competitive attention
        if self.training and self.gumbel_noise_strength > 0:
            logit_std = attention_logits.std(dim=1, keepdim=True).detach().clamp_min(1e-3)
            u = torch.rand_like(attention_logits)
            gumbel_noise = -torch.log(-torch.log(u.clamp(1e-6, 1 - 1e-6)))
            gumbel_noise = gumbel_noise.clamp(-5, 5)

            attention_logits = attention_logits + self.gumbel_noise_strength * logit_std * gumbel_noise

        # Apply softmax to get attention weights
        attention_weights_full = F.softmax(attention_logits, dim=1)  # [batch, latent_dim+1, H, W]
        
        # Always split into latent and background attention weights (1 background token)
        attention_weights = attention_weights_full[:, :self.latent_dim // self.dim_per_attention, :, :]  # [batch, latent_dim, H, W]
        background_weights = attention_weights_full[:, self.latent_dim // self.dim_per_attention:, :, :]  # [batch, 1, H, W]
    
        if upsample:
            attention_weights = F.interpolate(attention_weights, size=(self.image_size, self.image_size), mode='bilinear')
            background_weights = F.interpolate(background_weights, size=(self.image_size, self.image_size), mode='bilinear')

        # Calculate peak location if requested
        peak_location = None
        if return_peak_location:
            
            peak_location = attention_com_positions(attention_weights)

        
        # Return based on requested outputs
        if return_background_weights and return_peak_location:
            return attention_weights, background_weights, peak_location
        elif return_background_weights:
            return attention_weights, background_weights
        elif return_peak_location:
            return attention_weights, peak_location
        else:
            return attention_weights


    def coord_transform(self, z):
        # Create rotation matrix
        cos_r = torch.cos(self.coord_rotation)
        sin_r = torch.sin(self.coord_rotation)
        
        # Create scaling matrix
        scale_matrix = torch.diag(self.coord_scale).to(z.device)
        
        # Create rotation matrix using tensor operations (preserves gradients)
        rotation_matrix = torch.stack([
            torch.stack([cos_r, -sin_r]),
            torch.stack([sin_r, cos_r])
        ]).to(z.device)
        
        # Combine scaling and rotation: R = Scale * Rotation
        R = scale_matrix @ rotation_matrix
        
        z_transformed = z @ R + self.coord_bias
        return z_transformed


    def project_oscillator_position_and_velocity(self, z, z_dot=None):
        """
        Compute image-space oscillator positions and optionally velocities.

        Args:
            z: [batch, latent_dim] - latent positions
            z_dot: [batch, latent_dim] - latent velocities (optional)

        Returns:
            positions: [batch, n_nodes, 2] - image-space positions
            velocities: [batch, n_nodes, 2] - image-space velocities (if z_dot is provided)
        """
        batch = z.shape[0]
        n_nodes = self.latent_dim // self.dim_per_attention

        # Reshape latent positions
        latent_positions = z.view(batch, n_nodes, self.dim_per_attention)
        #positions = positions + self.position_offsets  # add per-node offsets
        positions = self.coord_transform(latent_positions) # affine transform

        if z_dot is None:
            return positions
        else:
            velocities = self.coord_transform(z_dot.view(batch, n_nodes, self.dim_per_attention))

            return positions, velocities


    
    def forward(self, z):
        """
        Forward pass with attention mechanism.
        
        Args:
            z: Latent tensor of shape [batch, latent_dim]
        
        Returns:
            Reconstructed image of shape [batch, output_channels, image_size, image_size]
        """
        batch_size = z.shape[0]
        
        # Step 1: Get attention weights (including background weights)
        # Always use 1 background token
        attention_weights, background_weights = self.get_attention_weights(z, return_background_weights=True, upsample=False)

        # Combine latent and background attention weights
        attention_weights_full = torch.cat([attention_weights, background_weights], dim=1)  # [batch, latent_dim + 1, H, W]
        
        # Step 2: Reshape latent to [batch, latent_dim, 1, 1] and upsample
        z_reshaped = z.view(batch_size, self.latent_dim, 1, 1)
        z_upsampled = F.interpolate(z_reshaped, size=(self.attention_size, self.attention_size), 
                                   mode='nearest')  # [batch, latent_dim, H, W]
        
        # Step 3: Expand each latent dimension to feature vectors
        z_expanded_list = []
        for i, expander in enumerate(self.latent_expanders):
            z_i = z_upsampled[:, i:i+1, :, :].unsqueeze(-1)  # [batch, 1, H, W, 1]
            z_i_expanded = expander(z_i)  # [batch, 1, H, W, feature_dim]
            z_expanded_list.append(z_i_expanded)
        
        
        if self.dim_per_attention == 1:
            # Standard case: [batch, latent_dim, H, W, feature_dim]
            z_expanded = torch.cat(z_expanded_list, dim=1)
        else:
            # Efficient grouping and concatenation without Python loop
            # z_expanded_list: list of [batch, 1, H, W, feature_dim], length latent_dim
            z_expanded_full = torch.cat(z_expanded_list, dim=1)  # [batch, latent_dim, H, W, feature_dim]
            batch, latent_dim, H, W, feature_dim = z_expanded_full.shape
            num_groups = latent_dim // self.dim_per_attention
            # Reshape to [batch, num_groups, dim_per_attention, H, W, feature_dim]
            z_grouped = z_expanded_full.view(batch, num_groups, self.dim_per_attention, H, W, feature_dim)
            # Move dim_per_attention to last feature dimension and merge
            z_expanded = z_grouped.permute(0, 1, 3, 4, 2, 5).reshape(batch, num_groups, H, W, self.dim_per_attention * feature_dim)
        


        # Step 4: Use learnable background tokens with spatial dimensions
        # Always use 1 background token
        background_features = self.background_features.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )  # [batch, 1, feature_dim, H, W]
        
        # Permute to match the expected format: [batch, 1, H, W, feature_dim]
        background_features_expanded = background_features.permute(0, 1, 3, 4, 2)  # [batch, 1, H, W, feature_dim]
        
        # Concatenate normal and background features
        concatenated_features = torch.cat([z_expanded, background_features_expanded], dim=1)  # [batch, latent_dim + 1, H, W, feature_dim]
        
        # Apply attention weights and sum along latent dimension
        z_attended = attention_weights_full.unsqueeze(-1) * concatenated_features  # [batch, latent_dim + 1, H, W, feature_dim]
        z_attended = torch.sum(z_attended, dim=1)  # [batch, H, W, feature_dim]
        
        # Permute dimensions to match Conv2d input format: [batch, H, W, feature_dim] -> [batch, feature_dim, H, W]
        z_attended = z_attended.permute(0, 3, 1, 2)  # [batch, feature_dim, H, W]
        
        # Step 5: Concatenate coordinates for spatial context in decoder
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        x_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)  # [1, 2, H, W]
        coords = coords.expand(batch_size, -1, -1, -1)  # [batch, 2, H, W]
        
        # Concatenate coordinates to attended features
        z_with_coords = torch.cat([z_attended, coords], dim=1)  # [batch, feature_dim + 2, H, W]
        
        # Step 6: Pass through main decoder
        output = self.decoder(z_with_coords)  # [batch, output_channels, H, W]
        
        return output

def selective_attention_consistency_loss(obs_curr, obs_next, attn_curr, attn_next):
    """
    Vectorized version with L2 norm across all channels, weighted by average activation
    """
    # Compute observation differences
    obs_diff = torch.abs(obs_next - obs_curr)  # [batch, C, H, W]
    
    # L2 norm across channels: sqrt(sum of squares across channel dimension)
    obs_diff_norm = torch.norm(obs_diff, dim=1, keepdim=True)  # [batch, 1, H, W]
    
    # Normalize to [0, 1] range
    obs_diff_norm = obs_diff_norm / (obs_diff_norm.max() + 1e-8)
    
    # Compute attention differences
    attn_diff = torch.abs(attn_next - attn_curr)  # [batch, latent_dim, H, W]
    
    # Compute average activation for each attention map
    attn_avg = (attn_curr + attn_next) / 2  # [batch, latent_dim, H, W]
    attn_avg_per_map = attn_avg.mean(dim=(2, 3), keepdim=True)  # [batch, latent_dim, 1, 1]
    
    # Weight attention differences by average activation
    weighted_attn_diff = attn_diff / (attn_avg_per_map + 1e-8)

    # Vectorized computation with broadcasting
    consistency_loss = (weighted_attn_diff * (1 - obs_diff_norm)).mean()
    
    return consistency_loss

def attention_velocity_loss(oscillator_position, oscillator_velocity, attention_weights, attention_dot, delta_t=1.0, eps=1e-6):
    batch, n_nodes_double = oscillator_velocity.shape  # [batch, n_nodes]
    n_nodes = n_nodes_double // 2
    if n_nodes < 2:
        return torch.tensor(0.0, device=oscillator_velocity.device, dtype=oscillator_velocity.dtype)

    attention_weights = (attention_weights * 0.1 + attention_weights.detach() * 0.9)
    attention_dot = (attention_dot * 0.1 + attention_dot.detach() * 0.9)
    
    com_pos = attention_com_positions(attention_weights)  # [batch, n_nodes, 2]
    com_vel = attention_com_velocities(attention_weights, attention_dot)  # [batch, n_nodes, 2]


    oscillator_position = oscillator_position.view(batch, n_nodes, 2)
    oscillator_velocity = oscillator_velocity.view(batch, n_nodes, 2)

    osc_pos_i = oscillator_position.unsqueeze(2)  # [batch, n_nodes, 1, 2]
    osc_pos_j = oscillator_position.unsqueeze(1)  # [batch, 1, n_nodes, 2]
    osc_rel_pos = osc_pos_i - osc_pos_j  # [batch, n_nodes, n_nodes, 2]
    rel_dist_osc = osc_rel_pos.norm(dim=-1).clamp(min=eps)  # [batch, n_nodes, n_nodes]

    osc_vel_i = oscillator_velocity.unsqueeze(2)  # [batch, n_nodes, 1, 2]
    osc_vel_j = oscillator_velocity.unsqueeze(1)  # [batch, 1, n_nodes, 2]
    osc_rel_vel_vec = osc_vel_i - osc_vel_j  # [batch, n_nodes, n_nodes, 2]
    osc_rel_dir = osc_rel_pos / rel_dist_osc.unsqueeze(-1)  # [batch, n_nodes, n_nodes, 2]
    osc_rel_vel_signed = (osc_rel_vel_vec * osc_rel_dir).sum(dim=-1)  # [batch, n_nodes, n_nodes]
    osc_disp = osc_rel_vel_signed * delta_t  # [batch, n_nodes, n_nodes]
    
    com_pos_i = com_pos.unsqueeze(2)  # [batch, n_nodes, 1, 2]
    com_pos_j = com_pos.unsqueeze(1)  # [batch, 1, n_nodes, 2]
    com_rel_pos = com_pos_i - com_pos_j  # [batch, n_nodes, n_nodes, 2]
    rel_dist_com = com_rel_pos.norm(dim=-1).clamp(min=eps)  # [batch, n_nodes, n_nodes]

    com_vel_i = com_vel.unsqueeze(2)  # [batch, n_nodes, 1, 2]
    com_vel_j = com_vel.unsqueeze(1)  # [batch, 1, n_nodes, 2]
    com_rel_vel_vec = com_vel_i - com_vel_j  # [batch, n_nodes, n_nodes, 2]
    com_rel_dir = com_rel_pos / rel_dist_com.unsqueeze(-1)  # [batch, n_nodes, n_nodes, 2]
    com_rel_vel_signed = (com_rel_vel_vec * com_rel_dir).sum(dim=-1)  # [batch, n_nodes, n_nodes]
    com_disp = com_rel_vel_signed * delta_t  # [batch, n_nodes, n_nodes]

    rel_mean_vel_osc = delta_t*(torch.abs(osc_vel_i.norm(dim=-1))+torch.abs(osc_vel_j.norm(dim=-1)))/2
    rel_mean_vel_com = delta_t*(torch.abs(com_vel_i.norm(dim=-1))+torch.abs(com_vel_j.norm(dim=-1)))/2

    # osc_scaled = osc_disp / rel_dist_osc  # [batch, n_nodes, n_nodes]
    # com_scaled = com_disp / rel_dist_com  # [batch, n_nodes, n_nodes]

    osc_scaled = osc_disp / rel_mean_vel_osc  # [batch, n_nodes, n_nodes]
    com_scaled = com_disp / rel_mean_vel_com  # [batch, n_nodes, n_nodes]

    mask = ~torch.eye(n_nodes, dtype=torch.bool, device=oscillator_velocity.device)  # [n_nodes, n_nodes]
    osc_masked = osc_scaled[:, mask]  # [batch, n_nodes*(n_nodes-1)]
    com_masked = com_scaled[:, mask]  # [batch, n_nodes*(n_nodes-1)]

    loss = F.mse_loss(osc_masked.clamp(min=-1, max=1), com_masked.clamp(min=-1, max=1))

    return loss


def attention_velocity_from_observation_velocity(vae, o_curr, o_dot):
    """
    Maps observation-space velocity to attention velocity via the attention function's Jacobian.
    Also returns the background_weights dot (background_dot).
    """
    def attention_function(obs):
        # Get latent state from observations
        if vae.is_vae:
            mu, _ = vae.encode(obs)
        else:
            mu = vae.encode(obs)
        # Get attention weights and background weights
        attn, background = vae.decoder.get_attention_weights(mu, return_background_weights=True, upsample=True)
        return attn, background

    # JVP: returns ((attn, background), (attn_dot, background_dot))
    _, (attn_dot, background_dot) = torch.autograd.functional.jvp(
        attention_function, (o_curr,), (o_dot,), create_graph=True, strict=False
    )
    return attn_dot, background_dot


def attention_consistency_loss_via_velocity(obs_dot, attn_dot, background_dot):
    """
    Loss based on attention velocity consistency - computed per latent dimension
    """
    # Compute observation differences
    obs_diff = torch.abs(obs_dot)  # [batch, C, H, W]
    obs_diff_norm = torch.norm(obs_diff, dim=1, keepdim=True)  # [batch, 1, H, W]
    obs_diff_norm = obs_diff_norm / (obs_diff_norm.max() + 1e-8)
    
    # Compute attention velocity magnitude for each latent dimension
    attn_velocity_mag = torch.abs(attn_dot)  # [batch, latent_dim, H, W]
    background_velocity_mag = torch.abs(background_dot)  # [batch, 1, H, W]
    
    # Loss: penalize attention velocity in regions with low observation velocity
    # Vectorized computation with broadcasting
    consistency_loss = attn_velocity_mag * (1 - obs_diff_norm)  # [batch, latent_dim, H, W]
    # background_loss = background_velocity_mag * (obs_diff_norm)  # [batch, 1, H, W]

    # Average over spatial dimensions and latent dimensions
    return consistency_loss.mean()


class VAE(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=32, is_vae=True, use_attention_decoder=True, attention_feature_dim=16, attention_downsample_factor=2, background_token_value=0.0, image_size=32, dynamics_type="harmonic1d", gumbel_noise_strength=0.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.is_vae = is_vae
        self.image_size = image_size
        # Create encoder and decoder based on lightweight flag
        self.encoder = create_encoder(input_channels, latent_dim,  is_vae, image_size)
        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.decoder = create_broadcast_decoder(latent_dim, input_channels, image_size=image_size, feature_dim=attention_feature_dim, background_token_value=background_token_value, attention_downsample_factor=attention_downsample_factor, dynamics_type=dynamics_type, gumbel_noise_strength=gumbel_noise_strength)
        else:
            self.decoder = create_decoder(latent_dim, input_channels,  image_size)
        
        
        
        # Initialize all parameters with Xavier uniform
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Xavier uniform initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        if self.is_vae:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar
        else:
            # For AE, just return the encoded representation
            mu = h
            return mu
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        if self.is_vae:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), z, mu, logvar
        else:
            mu = self.encode(x)
            return self.decode(mu), mu 


    def latent_velocity_from_observation_velocity(self, o_curr: torch.Tensor, o_dot: torch.Tensor) -> torch.Tensor:
        """
        Maps observation-space velocity to latent velocity via the encoder's Jacobian using forward-mode AD:
            z_dot(t_k) = (∂Φ/∂o)(o(t_k)) · o_dot(t_k)

        The mapping Φ returns the latent mean µ (not the sampled z).
        """
        def encoder_mu(inp: torch.Tensor) -> torch.Tensor:
            h = self.encoder(inp)
            if self.is_vae:
                mu, _ = h.chunk(2, dim=1)
            else:
                mu = h
            return mu

        # JVP: returns (encoder_mu(o_curr), J·o_dot). We keep only the tangent part.
        _, z_dot = torch.autograd.functional.jvp(encoder_mu, (o_curr,), (o_dot,), create_graph=False, strict=False)
        return z_dot


def vae_loss(recon_x, x, mu, logvar, x0=None):
    if x0 is not None:
        mu = mu - x0.detach()  # KL shapes encoder only; x0 updated by dynamics/steady-state

    # recon per sample (sum over pixels, mean over batch)
    recon_loss = nn.functional.mse_loss(recon_x, x)

    # KL per sample
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())



    return recon_loss, kld

def ae_loss(recon_x, x):

    # Reconstruction loss: average per pixel per sample (MSE)
    recon_loss = nn.functional.mse_loss(
        recon_x, x)

    return recon_loss


class KoopmanModel(nn.Module):

	def __init__(self, config: dict, init_K: str = "identity"):
		super().__init__()
		self.latent_dim = config["latent_dim"]
		self.state_dim = 2 * self.latent_dim
		self.actuation_dim = config["actuation_dim"]
		self.use_linear_control = config.get("use_linear_control", False)
		self.control_velocity_only = config.get("control_velocity_only", False)
		self.use_control_bias = config.get("use_control_bias", False)
		self.use_affine_drift = config.get("use_affine_drift", False)
		if self.use_affine_drift:
			self.c = nn.Parameter(torch.zeros(self.state_dim))

		# Learnable state transition matrix K for concatenated dynamics internally
		self.K = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
		if init_K == "identity":
			nn.init.eye_(self.K)
		elif init_K == "zeros":
			nn.init.zeros_(self.K)
		else:
			# Xavier uniform for a well-scaled random initialization
			nn.init.xavier_uniform_(self.K)

		# Control mapping: either linear B or MLP
		# Handle case where actuation_dim = 0 (no actuation)
		if self.actuation_dim > 0:
			if self.use_linear_control:
				if self.control_velocity_only:
					self.B_vel = nn.Parameter(torch.empty(self.latent_dim, self.actuation_dim))
					nn.init.xavier_uniform_(self.B_vel)
					if self.use_control_bias:
						self.b_vel = nn.Parameter(torch.zeros(self.latent_dim))
				else:
					self.B = nn.Parameter(torch.empty(self.state_dim, self.actuation_dim))
					nn.init.xavier_uniform_(self.B)
					if self.use_control_bias:
						self.b_u = nn.Parameter(torch.zeros(self.state_dim))
				self.control_to_state = None
			else:
				self.control_to_state = nn.Sequential(
					nn.Linear(self.actuation_dim, 16),
					nn.LeakyReLU(),
					nn.Linear(16, self.state_dim),
				)
		else:
			# No actuation: create a dummy module that returns zeros
			self.control_to_state = None

	def forward(
		self,
		z_t: torch.Tensor,
		z_dot_t: torch.Tensor,
		u_t: torch.Tensor,
		dt: float,
	) -> tuple[torch.Tensor, torch.Tensor]:

		# concatenate state
		x_t = torch.cat([z_t, z_dot_t], dim=1)

		# linear Koopman dynamics
		kx = x_t @ self.K.T

		# control contribution
		if u_t is not None and self.actuation_dim > 0:
			if self.use_linear_control:
				if self.control_velocity_only:
					bu = torch.zeros_like(kx)
					bu[..., self.latent_dim:] = u_t @ self.B_vel.T
					if self.use_control_bias:
						bu[..., self.latent_dim:] += self.b_vel
				else:
					bu = u_t @ self.B.T
					if self.use_control_bias:
						bu += self.b_u
			elif self.control_to_state is not None:
				bu = self.control_to_state(u_t)
			else:
				bu = torch.zeros_like(kx)
		else:
			bu = torch.zeros_like(kx)

		# affine drift (state bias)
		if self.use_affine_drift:
			kx = kx + self.c

		x_tp1 = kx + bu

		z_tp1 = x_tp1[..., :self.latent_dim]
		z_dot_tp1 = x_tp1[..., self.latent_dim:]

		return z_tp1, z_dot_tp1


	def ldm_loss(self, o, z_t, z_dot_t, u_t, VAE, delta_t):

		z_tp1_pred, z_dot_tp1_pred = self.forward(z_t, z_dot_t, u_t, delta_t)

		# consistency in latent space
		consistency_loss1 = nn.functional.mse_loss(
			z_tp1_pred[:-1], z_t[1:]
		)

		kinematic_loss = nn.functional.mse_loss(
			z_tp1_pred[:-1] - z_t[1:],
			z_dot_t[:-1] * delta_t
		)

		# consistency_loss2 = nn.functional.mse_loss(
		# 	z_dot_tp1_pred[:-1] * delta_t, z_dot_t[1:] * delta_t
		# )

		consistency_loss = consistency_loss1 + kinematic_loss #+ consistency_loss2

		# recon in observation space
		recon_o_tp1_pred = VAE.decode(z_tp1_pred)
		recon_loss = nn.functional.mse_loss(
			recon_o_tp1_pred[:-1], o[1:]
		)

		return recon_loss, consistency_loss
class KoopmanModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.latent_dim = config["latent_dim"]
        self.state_dim  = 2 * self.latent_dim
        self.actuation_dim = config["actuation_dim"]
        self.use_linear_control = config.get("use_linear_control", False)

        # Full Koopman dynamics matrix: x_next = K @ x
        # No kinematic constraints - let the model learn everything
        self.K = nn.Parameter(
            1e-2 * torch.randn(self.state_dim, self.state_dim)
        )
        # Initialize near identity for stability
        with torch.no_grad():
            self.K += torch.eye(self.state_dim) * 0.9

        # Control mapping: either linear B or MLP
        if self.actuation_dim > 0:
            if self.use_linear_control:
                # Full control matrix: B @ u affects entire state
                self.B = nn.Parameter(
                    1e-2 * torch.randn(self.state_dim, self.actuation_dim)
                )
                self.control_to_state = None
            else:
                # MLP mapping control input to state
                self.control_to_state = nn.Sequential(
                    nn.Linear(self.actuation_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 16),
                    nn.GELU(),
                    nn.Linear(16, self.state_dim),
                )
                self.B = None
        else:
            self.B = None
            self.control_to_state = None

    def forward(self, z_t, z_dot_t, u_t, dt):
        # Concatenate state: x = [z, z_dot]
        # Treat as general latent state (no kinematic constraints)
        x_t = torch.cat([z_t, z_dot_t], dim=1)  # [batch, state_dim]
        
        # Linear dynamics: x_next = K @ x
        x_tp1 = x_t @ self.K.T
        
        # Add control input: x_next = K @ x + B @ u (or MLP(u))
        # Control can affect entire state (both position and velocity components)
        if u_t is not None and self.actuation_dim > 0:
            if self.use_linear_control and self.B is not None:
                bu = u_t @ self.B.T
            elif self.control_to_state is not None:
                bu = self.control_to_state(u_t)
            else:
                bu = torch.zeros_like(x_tp1)
            x_tp1 = x_tp1 + bu
        
        # Split back into position and velocity
        z_tp1 = x_tp1[:, :self.latent_dim]
        v_tp1 = x_tp1[:, self.latent_dim:]
        
        return z_tp1, v_tp1

    def get_K(self, dt, device=None):
        """
        Get the Koopman matrix for control analysis.
        Returns the full K matrix (no kinematic constraints).
        """
        if device is None:
            device = next(self.parameters()).device
        return self.K.to(device)

    def ldm_loss(self, o, z_t, z_dot_t, u_t, VAE, dt):

        z_tp1_pred, v_tp1_pred = self.forward(
            z_t[:-1], z_dot_t[:-1], u_t[:-1], dt
        )

        # state consistency
        z_loss = nn.functional.mse_loss(z_tp1_pred, z_t[1:])
        v_loss = nn.functional.mse_loss(v_tp1_pred*dt, z_dot_t[1:]*dt)

        # reconstruction
        recon = VAE.decode(z_tp1_pred)
        recon_loss = nn.functional.mse_loss(recon, o[1:])

        return recon_loss, z_loss + v_loss

    def ldm_loss_multistep(self, o, z_t, z_dot_t, u_t, VAE, delta_t, n_steps=2):
        """
        Multi-step loss for BOTH latent consistency and observation reconstruction.
        """
        batch_size = z_t.shape[0] - n_steps
        
        # Initialize with ground truth
        z_curr = z_t[:batch_size]
        z_dot_curr = z_dot_t[:batch_size]
        
        total_dyn_loss = 0.0
        total_recon_loss = 0.0
        
        # Autoregressive rollout
        for step in range(n_steps):
            u_curr = u_t[step:batch_size+step]
            
            # Predict next state (uses previous prediction, not ground truth)
            z_next, z_dot_next = self.forward(z_curr, z_dot_curr, u_curr, delta_t)
            
            # Ground truth targets
            z_target = z_t[step+1:batch_size+step+1]
            z_dot_target = z_dot_t[step+1:batch_size+step+1]
            o_target = o[step+1:batch_size+step+1]
            
            # BOTH LOSSES computed at each horizon
            # 1. Latent dynamics consistency
            total_dyn_loss += F.mse_loss(z_next, z_target)
            total_dyn_loss += F.mse_loss(z_dot_next * delta_t, z_dot_target * delta_t)
            
            # 2. Observation reconstruction
            recon = VAE.decode(z_next)
            total_recon_loss += F.mse_loss(recon, o_target)
            
            # Use prediction for next iteration (autoregressive)
            z_curr = z_next
            z_dot_curr = z_dot_next
        
        # Average over steps
        avg_dyn_loss = total_dyn_loss / n_steps
        avg_recon_loss = total_recon_loss / n_steps
        
        return avg_recon_loss, avg_dyn_loss


class MLPModel(nn.Module):
    """
    MLP-based dynamics model for comparison with harmonic oscillator.
    Predicts next velocity directly, then enforces kinematic constraint using symplectic Euler.
    This gives the MLP flexibility while ensuring z_dot is the derivative of z.
    
    Uses symplectic Euler integration (same as harmonic oscillator):
    - Predict next velocity: z_dot_tp1 = MLP([z_t, z_dot_t]) + control
    - Update position using predicted velocity: z_tp1 = z_t + dt * z_dot_tp1
    
    The key is using the FUTURE velocity (z_dot_tp1) to update position, which makes
    this symplectic Euler and ensures better energy conservation properties.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.latent_dim = config["latent_dim"]
        self.state_dim  = 2 * self.latent_dim
        self.actuation_dim = config["actuation_dim"]
        self.use_linear_control = config.get("use_linear_control", False)
        self.n_neurons = config.get("mlp_neurons", 32)

        # MLP dynamics: predicts next velocity z_dot_tp1
        # Input: [z_t, z_dot_t] (concatenated state)
        # Output: z_dot_tp1 (next velocity)
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.n_neurons),
            nn.GELU(),
            nn.Linear(self.n_neurons, self.n_neurons),
            nn.GELU(),
            nn.Linear(self.n_neurons, self.latent_dim),  # Predict next velocity
        )
        # Standard initialization (no special identity initialization)
        # PyTorch default initialization is used

        # Control mapping: either linear B or MLP
        # Control affects velocity (like forces affect acceleration in harmonic oscillator)
        if self.actuation_dim > 0:
            if self.use_linear_control:
                # Linear control: B @ u affects velocity
                self.B = nn.Parameter(
                    1e-2 * torch.randn(self.latent_dim, self.actuation_dim)
                )
                self.control_to_state = None
            else:
                # MLP mapping control input to velocity change
                self.control_to_state = nn.Sequential(
                    nn.Linear(self.actuation_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 16),
                    nn.GELU(),
                    nn.Linear(16, self.latent_dim),  # Maps to velocity
                )
                self.B = None
        else:
            self.B = None
            self.control_to_state = None

    def forward(self, z_t, z_dot_t, u_t, dt):
        # Concatenate state: x = [z, z_dot]
        x_t = torch.cat([z_t, z_dot_t], dim=1)  # [batch, state_dim]
        
        # MLP predicts next velocity: z_dot_tp1 = MLP([z_t, z_dot_t])
        z_dot_tp1 = self.mlp(x_t)
        
        # Add control input (affects velocity)
        if u_t is not None and self.actuation_dim > 0:
            if self.use_linear_control and self.B is not None:
                z_dot_tp1 = z_dot_tp1 + (u_t @ self.B.T)
            elif self.control_to_state is not None:
                z_dot_tp1 = z_dot_tp1 + self.control_to_state(u_t)
        
        # Symplectic Euler integration (same as harmonic oscillator):
        # Update position using the PREDICTED velocity (not current velocity)
        # This ensures z_dot_tp1 is the derivative of z_tp1 by construction
        z_tp1 = z_t + dt * z_dot_tp1
        
        return z_tp1, z_dot_tp1

    def ldm_loss(self, o, z_t, z_dot_t, u_t, VAE, dt):

        z_tp1_pred, v_tp1_pred = self.forward(
            z_t[:-1], z_dot_t[:-1], u_t[:-1], dt
        )

        # state consistency
        z_loss = nn.functional.mse_loss(z_tp1_pred, z_t[1:])
        v_loss = nn.functional.mse_loss(v_tp1_pred*dt, z_dot_t[1:]*dt)

        # reconstruction
        recon = VAE.decode(z_tp1_pred)
        recon_loss = nn.functional.mse_loss(recon, o[1:])

        return recon_loss, z_loss + v_loss

    def ldm_loss_multistep(self, o, z_t, z_dot_t, u_t, VAE, delta_t, n_steps=2):
        """
        Multi-step loss for BOTH latent consistency and observation reconstruction.
        """
        batch_size = z_t.shape[0] - n_steps
        
        # Initialize with ground truth
        z_curr = z_t[:batch_size]
        z_dot_curr = z_dot_t[:batch_size]
        
        total_dyn_loss = 0.0
        total_recon_loss = 0.0
        
        # Autoregressive rollout
        for step in range(n_steps):
            u_curr = u_t[step:batch_size+step]
            
            # Predict next state (uses previous prediction, not ground truth)
            z_next, z_dot_next = self.forward(z_curr, z_dot_curr, u_curr, delta_t)
            
            # Ground truth targets
            z_target = z_t[step+1:batch_size+step+1]
            z_dot_target = z_dot_t[step+1:batch_size+step+1]
            o_target = o[step+1:batch_size+step+1]
            
            # BOTH LOSSES computed at each horizon
            # 1. Latent dynamics consistency
            total_dyn_loss += F.mse_loss(z_next, z_target)
            total_dyn_loss += F.mse_loss(z_dot_next * delta_t, z_dot_target * delta_t)
            
            # 2. Observation reconstruction
            recon = VAE.decode(z_next)
            total_recon_loss += F.mse_loss(recon, o_target)
            
            # Use prediction for next iteration (autoregressive)
            z_curr = z_next
            z_dot_curr = z_dot_next
        
        # Average over steps
        avg_dyn_loss = total_dyn_loss / n_steps
        avg_recon_loss = total_recon_loss / n_steps
        
        return avg_recon_loss, avg_dyn_loss

EPS = 1e-6

def softplus_pos(x):
    return torch.nn.functional.softplus(x) + EPS

class FullyCoupledOscNet(nn.Module):
    """
    Fully coupled harmonic oscillator network with MLP as external force.
    Equation: M x_ddot + D x_dot + K x = F(u) + F_nl(x)
    Following paper: learning M_inv, K, D using upper triangular Cholesky with specific constraints
    """
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.harmonic_prediction_function = config["harmonic_prediction_function"]
        self.harmonic_use_nonlinear_forcing = config["harmonic_use_nonlinear_forcing"]
        self.n = config["latent_dim"]
        self.m_in = config["actuation_dim"]
        self.device = device
        self.dynamics_type = config["dynamics_type"]
        self.harmonic_damping_type = config["harmonic_damping_type"]

        # Paper constants for positive definiteness constraints
        self.eps1 = 1e-6
        self.eps2 = 2e-6

        # Learn M_inv directly (not M) using upper triangular Cholesky
        # self.minv_cholesky_raw = nn.Parameter(torch.empty(n, n))
        # nn.init.xavier_uniform_(self.minv_cholesky_raw)

        # INSERT_YOUR_CODE
        # Make M_inv just diagonal: only learn n parameters for the diagonal
        
        if "harmonic2d" in self.dynamics_type:
            self.minv_diag_raw = nn.Parameter(torch.empty(self.n//2))
        else:
            self.minv_diag_raw = nn.Parameter(torch.empty(self.n))

        self.m_scale = nn.Parameter(torch.ones(1))

        nn.init.ones_(self.minv_diag_raw)
        
        # Learn K using upper triangular Cholesky
        self.k_raw = nn.Parameter(torch.empty(self.n, self.n))
        self.k_ground_raw = nn.Parameter(torch.zeros(self.n))
        self.k_scale = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.k_raw)


        # Learn D using upper triangular Cholesky
        if self.harmonic_damping_type == "full":
            self.d_raw = nn.Parameter(torch.empty(self.n, self.n))
            self.d_ground_raw = nn.Parameter(torch.zeros(self.n))
            self.d_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.d_raw)
        elif self.harmonic_damping_type == "rayleigh":
            # self.alpha_raw = nn.Parameter(torch.zeros(1))
            # self.beta_raw = nn.Parameter(torch.zeros(1))
            self.alpha_raw = nn.Parameter(torch.tensor(2.0)) # exp(2.0) ≈ 7.4
            self.beta_raw = nn.Parameter(torch.tensor(-2.0)) # exp(-2.0) ≈ 0.1 (reduce stiffness coupling)
        else:
            raise ValueError(f"Unknown damping type: {self.harmonic_damping_type}")

        #if dynamics_type == "2dharmonic":
        if "harmonic2d" in self.dynamics_type:
            self.x0 = nn.Parameter(torch.zeros(self.n))
        else:
            self.x0 = torch.zeros(self.n, device=device)

        # Nonlinear forcing terms (b parameter and W matrix for tanh nonlinearity)
        if self.harmonic_use_nonlinear_forcing:
            self.W_raw = nn.Parameter(torch.empty(self.n, self.n))
            nn.init.xavier_uniform_(self.W_raw)
            self.b_raw = nn.Parameter(torch.zeros(self.n)) # Bias term


        # Control mapping: either linear B or MLP
        self.use_linear_control = config.get("use_linear_control", False)
        
        # Handle case where m_in = 0 (no actuation)
        if self.m_in > 0:
            if self.use_linear_control:
                # Linear B matrix: u directly maps to forces
                self.B = nn.Parameter(
                    1e-2 * torch.randn(self.n, self.m_in)
                )
                self.control_to_state = None
            else:
                # MLP mapping control input to external force F(u)
                self.control_to_state = nn.Sequential(
                    nn.Linear(self.m_in, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, self.n),
                )
                self.B = None
        else:
            # No actuation: create a dummy module that returns zeros
            self.control_to_state = None
            self.B = None


    def _apply_cholesky_constraints(self, U_raw):
        """
        Apply paper's Cholesky constraints to ensure positive definiteness.
        U_ii = log(1 + e^(U_ii + ε1)) + ε2
        where ε1 = 1e-6, ε2 = 2e-6
        """
        # Get upper triangular part
        U = torch.triu(U_raw)
        
        # Apply the specific diagonal operation from the paper
        diag_indices = torch.arange(self.n, device=U.device)
        U_diag = U[diag_indices, diag_indices]
        
        # U_ii = log(1 + e^(U_ii + ε1)) + ε2
        constrained_diag = torch.log(1 + torch.exp(U_diag + self.eps1)) + self.eps2
        
        # Replace diagonal with constrained values
        U = U.clone()
        U[diag_indices, diag_indices] = constrained_diag
        
        # Construct positive definite matrix: A = U^T @ U
        A = U.T @ U
        
        return A

    def build_physical_coupling_matrix(self, U_raw, ground_raw=None):
        #f = lambda x: F.softplus(x + self.eps1) + self.eps2
        # Alternative definition of f using exp (for experimentation)

        # Make all pairwise couplings positive, symmetric, zero diagonal
        K_pair = F.softplus(U_raw)
        K_pair = 0.5 * (K_pair + K_pair.T)
        K_pair.fill_diagonal_(0.0)

        # Compute total coupling per node
        row_sum = torch.sum(K_pair, dim=1)

        # Ground coupling
        K_ground = F.softplus(ground_raw) if ground_raw is not None else torch.zeros_like(row_sum)

        # Build physical Laplacian-like matrix
        A = -K_pair + torch.diag(row_sum + K_ground)

        return A




    def give_Minv_KD(self):
        """
        Build M_inv, K, D from their Cholesky factors using paper's approach.
        Returns M_inv directly (not M) as per paper methodology.
        """
        # Build M_inv using upper triangular Cholesky with constraints
        #M_inv = self._apply_cholesky_constraints(self.minv_cholesky_raw)
        if "harmonic2d" in self.dynamics_type:
            #M_inv = torch.diag(torch.nn.functional.softplus(self.minv_diag_raw).repeat_interleave(2))
            M_inv = torch.diag(F.softplus(self.minv_diag_raw).repeat_interleave(2) * torch.exp(self.m_scale))
        else:
            #M_inv = torch.diag(torch.nn.functional.softplus(self.minv_diag_raw))
            M_inv = torch.diag(F.softplus(self.minv_diag_raw) * torch.exp(self.m_scale))

        # Build K using upper triangular Cholesky with constraints  
        # K = self._apply_cholesky_constraints(self.k_raw)
        K = self.build_physical_coupling_matrix(self.k_raw, self.k_ground_raw) * torch.exp(self.k_scale)
        # K = self.build_physical_coupling_matrix(self.k_raw)
        
        # Build D using upper triangular Cholesky with constraints
        # D = self._apply_cholesky_constraints(self.d_raw)
        if self.harmonic_damping_type == "full":
            D = self.build_physical_coupling_matrix(self.d_raw, self.d_ground_raw) * torch.exp(self.d_scale)
        elif self.harmonic_damping_type == "rayleigh":
            # Efficiently invert diagonal M_inv to get M (assume M_inv is diagonal)
            M = torch.diag(1.0 / torch.diagonal(M_inv))
            #D = F.softplus(self.alpha_raw) * M + F.softplus(self.beta_raw) * K
            D = torch.exp(self.alpha_raw) * M + torch.exp(self.beta_raw) * K
        return M_inv, K, D


    def step_symplectic_euler_old(self, y, u, dt):
        """
        Step dynamics forward using symplectic Euler integration
        y: [batch, 2n] state [x, v]
        u: [batch, m_in] control input
        Adds Kx0 to the control force.
        """
        x, v = y[:, :self.n], y[:, self.n:]
        M_inv, K, D = self.give_Minv_KD()
        
        # Handle actuation
        if u is not None:
            if self.use_linear_control and self.B is not None:
                # Linear control: u -> B @ u (forces)
                bu = u @ self.B.T  # [batch, m_in] @ [m_in, n] -> [batch, n]
            elif self.control_to_state is not None:
                # MLP control: u -> MLP(u) (forces)
                bu = self.control_to_state(u)  # [batch, n]
            else:
                # No actuation: zero force
                bu = torch.zeros(y.shape[0], self.n, device=y.device)
        else:
            # No actuation: zero force
            bu = torch.zeros(y.shape[0], self.n, device=y.device)

        # Add Kx0 to the control force
        kx0 = (K @ self.x0).unsqueeze(0)  # [1, n]
        bu = bu + kx0  # [batch, n] + [1, n] -> [batch, n]

        forces = bu - (K @ x.T).T - (D @ v.T).T
        
        # Add nonlinear forcing (Stölzle & Santina 2024)
        if self.harmonic_use_nonlinear_forcing:
            nonlinear_force = torch.tanh(self.W_raw @ x.T + self.b_raw.unsqueeze(1)).T
            forces = forces + nonlinear_force
        
        a = (M_inv @ forces.T).T  # Use learned M_inv directly

        v_next = v + dt * a
        x_next = x + dt * v_next  # notice: uses updated v
        return torch.cat([x_next, v_next], dim=1)

    def step_symplectic_euler(self, y, u, dt):
        x, v = y[:, :self.n], y[:, self.n:]
        M_inv, K, D = self.give_Minv_KD()
        
        # Handle actuation
        if u is not None:
            if self.use_linear_control and self.B is not None:
                # Linear control: u -> B @ u (forces)
                bu = u @ self.B.T  # [batch, m_in] @ [m_in, n] -> [batch, n]
            elif self.control_to_state is not None:
                # MLP control: u -> MLP(u) (forces)
                bu = self.control_to_state(u)  # [batch, n]
            else:
                # No actuation: zero force
                bu = torch.zeros(y.shape[0], self.n, device=y.device)
        else:
            # No actuation: zero force
            bu = torch.zeros(y.shape[0], self.n, device=y.device)

        # Add Kx0 to the control force
        kx0 = (K @ self.x0).unsqueeze(0)  # [1, n]
        bu = bu + kx0  # [batch, n] + [1, n] -> [batch, n]

        # Exclude damping force from 'forces' variable initially
        forces_no_damping = bu - (K @ x.T).T # + nonlinear terms
        
        # Acceleration from springs/inputs only
        a_spring = (forces_no_damping @ M_inv.T)
        
        # Implicit Damping Update:
        # v_new = v + dt * (a_spring - M_inv * D * v_new)
        # (I + dt * M_inv * D) * v_new = v + dt * a_spring
        # v_new = (I + dt * M_inv * D)^-1 * (v + dt * a_spring)
        
        # Approximate Implicit Damping (assuming diagonal M, D for speed):
        # This divides velocity by (1 + decay) at every step. Extremely stable.
        decay_factor = 1.0 / (1.0 + dt * torch.diagonal(M_inv @ D))
        v_next = (v + dt * a_spring) * decay_factor.unsqueeze(0)
        
        x_next = x + dt * v_next
        return torch.cat([x_next, v_next], dim=1)

    def rollout_symplectic_euler(self, x0, v0, u_seq, dt):
        y = torch.cat([x0, v0], dim=1)
        xs, vs = [], []
        for u in u_seq:
            y = self.step_symplectic_euler(y, u, dt)
            xs.append(y[:, :self.n])
            vs.append(y[:, self.n:])
        xs = torch.stack(xs, dim=0)
        vs = torch.stack(vs, dim=0)
        return xs, vs

    def rollout_symplectic_euler_old(self, x0, v0, u_seq, dt):
        y = torch.cat([x0, v0], dim=1)
        xs, vs = [], []
        for u in u_seq:
            y = self.step_symplectic_euler_old(y, u, dt)
            xs.append(y[:, :self.n])
            vs.append(y[:, self.n:])
        xs = torch.stack(xs, dim=0)
        vs = torch.stack(vs, dim=0)
        return xs, vs

class HarmonicOscillatorDynamics(nn.Module):
    def __init__(self, config: dict, device='cpu'):
        super().__init__()
        self.actuation_dim = config["actuation_dim"]
        self.osc_net = FullyCoupledOscNet(config, device)
        self.latent_dim = config["latent_dim"]
        self.harmonic_prediction_function = config["harmonic_prediction_function"]
        self.dynamics_type = config.get("dynamics_type", "harmonic1d")

    def forward(self, z_t, z_dot_t, u_t, dt=1.0):
        # For harmonic2d with x0: z_t is centered (zero mean at rest). Convert to absolute for physics step, then back to centered output.
        
        y_t = torch.cat([z_t, z_dot_t], dim=1)

        if self.harmonic_prediction_function == "symplectic_euler":
            y_tp1 = self.osc_net.step_symplectic_euler(y_t, u_t, dt)
        elif self.harmonic_prediction_function == "symplectic_euler_old":
            y_tp1 = self.osc_net.step_symplectic_euler_old(y_t, u_t, dt)
        else:
            raise ValueError(f"Unknown harmonic prediction function: {self.harmonic_prediction_function}")

        z_tp1_abs = y_tp1[:, :self.latent_dim]
        z_dot_tp1 = y_tp1[:, self.latent_dim:]
        if "harmonic2d" in self.dynamics_type:
            z_tp1 = z_tp1_abs# - x0  # centered next state
        else:
            z_tp1 = z_tp1_abs
        return z_tp1, z_dot_tp1

    # def latent_centered(self, z_abs: torch.Tensor) -> torch.Tensor:
    #     """Convert absolute latent to centered (for dynamics input). When harmonic2d: z_abs - x0; else pass-through."""
    #     if "harmonic2d" not in self.dynamics_type:
    #         return z_abs
    #     x0 = self.osc_net.x0.unsqueeze(0)
    #     return z_abs - x0

    # def latent_absolute(self, z_centered: torch.Tensor) -> torch.Tensor:
    #     """Convert centered latent to absolute (for decoder input). When harmonic2d: z_centered + x0; else pass-through."""
    #     if "harmonic2d" not in self.dynamics_type:
    #         return z_centered
    #     x0 = self.osc_net.x0.unsqueeze(0)
    #     return z_centered + x0

    def ldm_loss(self, o: torch.Tensor, z_t: torch.Tensor, z_dot_t: torch.Tensor, u_t: torch.Tensor, VAE, delta_t) -> torch.Tensor:
        """
        Compute loss for harmonic oscillator dynamics learning.
        z_t is expected to be centered when using harmonic2d (zero mean at rest).

        Args:
            o: Observations [batch, ...]
            z_t: Latent states [batch, latent_dim] (centered for harmonic2d)
            z_dot_t: Latent velocities [batch, latent_dim]
            u_t: Control inputs [batch, actuation_dim]
            VAE: VAE model for decoding
            delta_t: Time step

        Returns:
            recon_loss: Reconstruction loss (per sample)
            consistency_loss: Dynamics consistency loss (per sample)
        """
        z_tp1_pred, z_dot_tp1_pred = self.forward(z_t, z_dot_t, u_t, delta_t)

        # Consistency loss: predicted vs actual next states (both centered)
        consistency_loss1 = torch.nn.functional.mse_loss(
            z_tp1_pred[:-1], z_t[1:]
        )

        consistency_loss2 = torch.nn.functional.mse_loss(
            z_dot_tp1_pred[:-1]*delta_t, z_dot_t[1:]*delta_t
        )

        consistency_loss = consistency_loss1 + consistency_loss2

        # Decode: use absolute latent so decoder sees same coordinate system as encoder output
        #z_tp1_for_decode = self.latent_absolute(z_tp1_pred)
        recon_o_pred = VAE.decode(z_tp1_pred)
        recon_loss = torch.nn.functional.mse_loss(
            recon_o_pred[:-1], o[1:]
        )

        return recon_loss, consistency_loss

    def ldm_loss_multistep(self, o, z_t, z_dot_t, u_t, VAE, delta_t, n_steps=2):
        """
        Multi-step loss for BOTH latent consistency and observation reconstruction.
        """
        batch_size = z_t.shape[0] - n_steps
        
        # Initialize with ground truth
        z_curr = z_t[:batch_size]
        z_dot_curr = z_dot_t[:batch_size]
        
        total_dyn_loss = 0.0
        total_recon_loss = 0.0
        
        # Autoregressive rollout
        for step in range(n_steps):
            u_curr = u_t[step:batch_size+step]
            
            # Predict next state (uses previous prediction, not ground truth)
            z_next, z_dot_next = self.forward(z_curr, z_dot_curr, u_curr, delta_t)
            
            # Ground truth targets
            z_target = z_t[step+1:batch_size+step+1]
            z_dot_target = z_dot_t[step+1:batch_size+step+1]
            o_target = o[step+1:batch_size+step+1]
            
            # BOTH LOSSES computed at each horizon
            # 1. Latent dynamics consistency
            total_dyn_loss += F.mse_loss(z_next, z_target)
            total_dyn_loss += F.mse_loss(z_dot_next * delta_t, z_dot_target * delta_t)
            
            # 2. Observation reconstruction
            recon = VAE.decode(z_next)
            total_recon_loss += F.mse_loss(recon, o_target)
            
            # Use prediction for next iteration (autoregressive)
            z_curr = z_next
            z_dot_curr = z_dot_next
        
        # Average over steps
        avg_dyn_loss = total_dyn_loss / n_steps
        avg_recon_loss = total_recon_loss / n_steps
        
        return avg_recon_loss, avg_dyn_loss

def create_dynamics_model(config: dict, device='cpu'):
    """
    Factory function to create the appropriate dynamics model.
    
    Args:
        dynamics_type: 'koopman', 'mlp', 'harmonic', 'actuation_harmonic', or 'physics_koopman'
        latent_dim: Dimension of latent space
        actuation_dim: Dimension of actuation space
        device: Device to place the model on
    
    Returns:
        Dynamics model (KoopmanModel, MLPModel, HarmonicOscillatorDynamics, ActuationDependentHarmonicDynamics, or PhysicsInformedKoopman)
    """
    if config["dynamics_type"].lower() == 'koopman':
        return KoopmanModel(config)
    elif config["dynamics_type"].lower() == 'mlp':
        return MLPModel(config)
    elif "harmonic" in config["dynamics_type"].lower():
        return HarmonicOscillatorDynamics(config, device)
    else:
        raise ValueError(f"Unknown dynamics_type: {config['dynamics_type']}. Must be 'koopman', 'mlp', 'harmonic', '2dharmonic'")