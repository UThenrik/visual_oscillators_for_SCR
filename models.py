import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from torch.nn.modules.loss import _Loss

def create_encoder(input_channels=1, latent_dim=16, is_vae=True, image_size=32):
    """Simple encoder architecture with 4x4 kernels and stride=2."""
    if image_size <= 32:
        num_conv_layers = 3
    elif image_size <= 64:
        num_conv_layers = 4
    elif image_size <= 96:
        num_conv_layers = 4
    else:
        num_conv_layers = 5
    
    layers = []
    in_channels = input_channels
    
    for i in range(num_conv_layers):
        out_channels = 32 * (2 ** i)
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])
        in_channels = out_channels
    
    if image_size == 96:
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
    
    layers.extend([
        nn.Flatten(),
        nn.Linear(in_channels * 4 * 4, latent_dim * (2 if is_vae else 1))
    ])
    
    return nn.Sequential(*layers)


def create_decoder(latent_dim=16, output_channels=1, image_size=32):
    """Simple decoder architecture with 4x4 kernels and stride=2."""
    if image_size <= 32:
        num_deconv_layers = 3
    elif image_size <= 64:
        num_deconv_layers = 4
    elif image_size <= 96:
        num_deconv_layers = 4
    else:
        num_deconv_layers = 5
    
    layers = []
    
    if image_size <= 32:
        encoder_channels = 128
    elif image_size <= 64:
        encoder_channels = 256
    elif image_size <= 96:
        encoder_channels = 256
    else:
        encoder_channels = 512
    
    layers.extend([
        nn.Linear(latent_dim, encoder_channels * 4 * 4),
        nn.Unflatten(1, (encoder_channels, 4, 4))
    ])
    
    in_channels = encoder_channels
    for i in range(num_deconv_layers):
        if i == num_deconv_layers - 1:
            out_channels = output_channels
            activation = nn.Sigmoid()
        else:
            out_channels = in_channels // 2
            activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        layers.extend([
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            activation
        ])
        in_channels = out_channels
    
    if image_size == 96:
        layers.append(nn.AdaptiveAvgPool2d((96, 96)))
    
    return nn.Sequential(*layers)


def create_broadcast_decoder(latent_dim=16, output_channels=1, image_size=32, feature_dim=16, 
                             background_token_value=0.0, attention_downsample_factor=1, 
                             dynamics_type="harmonic1d", gumbel_noise_strength=0.0):
    """Broadcast decoder that broadcasts latent dimensions to image dimensions."""
    return AttentionBroadcastDecoder(latent_dim, output_channels, image_size, feature_dim, 
                                     background_token_value, attention_downsample_factor, 
                                     dynamics_type, gumbel_noise_strength)


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

    attn_sq = attention_weights ** 2

    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
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
        self.gumbel_noise_strength = gumbel_noise_strength

        if "harmonic2d" in dynamics_type:
            self.dim_per_attention = 2
        else:
            self.dim_per_attention = 1

        if self.dim_per_attention == 2:
            self.coord_scale = nn.Parameter(torch.ones(2))
            self.coord_rotation = nn.Parameter(torch.tensor(0.0))
            self.coord_bias = nn.Parameter(torch.zeros(1, 1, 2))

        self.attention_size = image_size // attention_downsample_factor
        output_dim = latent_dim // self.dim_per_attention
        self.attention_net = nn.Sequential(
            nn.Conv2d(latent_dim + 2, 128, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=1, padding='same')
        )
        
        self.background_features = nn.Parameter(
            torch.randn(1, feature_dim*self.dim_per_attention, self.attention_size, self.attention_size)
        )
        
        self.latent_expanders = nn.ModuleList([
            nn.Linear(1, feature_dim) for _ in range(latent_dim)
        ])
        
        decoder_input_channels = feature_dim*self.dim_per_attention + 2
        if self.attention_downsample_factor == 1:
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_input_channels, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        elif self.attention_downsample_factor == 2:
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_input_channels, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        elif self.attention_downsample_factor == 4:
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

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Kaiming initialization."""
        negative_slope = 0.2

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
        
        y_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        x_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)
        coords = coords.expand(batch_size, -1, -1, -1)
        
        z_broadcast = z.view(batch_size, self.latent_dim, 1, 1).expand(-1, -1, self.attention_size, self.attention_size)
        latent_with_coords = torch.cat([z_broadcast, coords], dim=1)
        
        attention_logits = self.attention_net(latent_with_coords)

        zeros_bg = self.background_token_value * torch.ones(
            attention_logits.shape[0],
            1,
            attention_logits.shape[2],
            attention_logits.shape[3],
            device=attention_logits.device,
            dtype=attention_logits.dtype
        )

        attention_logits = torch.cat([attention_logits, zeros_bg], dim=1)

        if self.training and self.gumbel_noise_strength > 0:
            logit_std = attention_logits.std(dim=1, keepdim=True).detach().clamp_min(1e-3)
            u = torch.rand_like(attention_logits)
            gumbel_noise = -torch.log(-torch.log(u.clamp(1e-6, 1 - 1e-6)))
            gumbel_noise = gumbel_noise.clamp(-5, 5)
            attention_logits = attention_logits + self.gumbel_noise_strength * logit_std * gumbel_noise

        attention_weights_full = F.softmax(attention_logits, dim=1)
        attention_weights = attention_weights_full[:, :self.latent_dim // self.dim_per_attention, :, :]
        background_weights = attention_weights_full[:, self.latent_dim // self.dim_per_attention:, :, :]
    
        if upsample:
            attention_weights = F.interpolate(attention_weights, size=(self.image_size, self.image_size), mode='bilinear')
            background_weights = F.interpolate(background_weights, size=(self.image_size, self.image_size), mode='bilinear')

        peak_location = None
        if return_peak_location:
            peak_location = attention_com_positions(attention_weights)
        if return_background_weights and return_peak_location:
            return attention_weights, background_weights, peak_location
        elif return_background_weights:
            return attention_weights, background_weights
        elif return_peak_location:
            return attention_weights, peak_location
        else:
            return attention_weights


    def coord_transform(self, z):
        cos_r = torch.cos(self.coord_rotation)
        sin_r = torch.sin(self.coord_rotation)
        scale_matrix = torch.diag(self.coord_scale).to(z.device)
        rotation_matrix = torch.stack([
            torch.stack([cos_r, -sin_r]),
            torch.stack([sin_r, cos_r])
        ]).to(z.device)
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
        latent_positions = z.view(batch, n_nodes, self.dim_per_attention)
        positions = self.coord_transform(latent_positions)

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
        
        attention_weights, background_weights = self.get_attention_weights(z, return_background_weights=True, upsample=False)
        attention_weights_full = torch.cat([attention_weights, background_weights], dim=1)
        
        z_reshaped = z.view(batch_size, self.latent_dim, 1, 1)
        z_upsampled = F.interpolate(z_reshaped, size=(self.attention_size, self.attention_size), mode='nearest')
        
        z_expanded_list = []
        for i, expander in enumerate(self.latent_expanders):
            z_i = z_upsampled[:, i:i+1, :, :].unsqueeze(-1)
            z_i_expanded = expander(z_i)
            z_expanded_list.append(z_i_expanded)
        
        if self.dim_per_attention == 1:
            z_expanded = torch.cat(z_expanded_list, dim=1)
        else:
            z_expanded_full = torch.cat(z_expanded_list, dim=1)
            batch, latent_dim, H, W, feature_dim = z_expanded_full.shape
            num_groups = latent_dim // self.dim_per_attention
            z_grouped = z_expanded_full.view(batch, num_groups, self.dim_per_attention, H, W, feature_dim)
            z_expanded = z_grouped.permute(0, 1, 3, 4, 2, 5).reshape(batch, num_groups, H, W, self.dim_per_attention * feature_dim)

        background_features = self.background_features.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        background_features_expanded = background_features.permute(0, 1, 3, 4, 2)
        concatenated_features = torch.cat([z_expanded, background_features_expanded], dim=1)
        
        z_attended = attention_weights_full.unsqueeze(-1) * concatenated_features
        z_attended = torch.sum(z_attended, dim=1)
        z_attended = z_attended.permute(0, 3, 1, 2)
        
        y_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        x_coords = torch.linspace(-1, 1, self.attention_size, device=z.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0)
        coords = coords.expand(batch_size, -1, -1, -1)
        
        z_with_coords = torch.cat([z_attended, coords], dim=1)
        output = self.decoder(z_with_coords)
        
        return output

def attention_velocity_loss(oscillator_position, oscillator_velocity, attention_weights, attention_dot, delta_t=1.0, eps=1e-6):
    batch, n_nodes_double = oscillator_velocity.shape
    n_nodes = n_nodes_double // 2
    if n_nodes < 2:
        return torch.tensor(0.0, device=oscillator_velocity.device, dtype=oscillator_velocity.dtype)

    attention_weights = (attention_weights * 0.1 + attention_weights.detach() * 0.9)
    attention_dot = (attention_dot * 0.1 + attention_dot.detach() * 0.9)
    
    com_pos = attention_com_positions(attention_weights)
    com_vel = attention_com_velocities(attention_weights, attention_dot)

    oscillator_position = oscillator_position.view(batch, n_nodes, 2)
    oscillator_velocity = oscillator_velocity.view(batch, n_nodes, 2)

    osc_pos_i = oscillator_position.unsqueeze(2)
    osc_pos_j = oscillator_position.unsqueeze(1)
    osc_rel_pos = osc_pos_i - osc_pos_j
    rel_dist_osc = osc_rel_pos.norm(dim=-1).clamp(min=eps)

    osc_vel_i = oscillator_velocity.unsqueeze(2)
    osc_vel_j = oscillator_velocity.unsqueeze(1)
    osc_rel_vel_vec = osc_vel_i - osc_vel_j
    osc_rel_dir = osc_rel_pos / rel_dist_osc.unsqueeze(-1)
    osc_rel_vel_signed = (osc_rel_vel_vec * osc_rel_dir).sum(dim=-1)
    osc_disp = osc_rel_vel_signed * delta_t
    
    com_pos_i = com_pos.unsqueeze(2)
    com_pos_j = com_pos.unsqueeze(1)
    com_rel_pos = com_pos_i - com_pos_j
    rel_dist_com = com_rel_pos.norm(dim=-1).clamp(min=eps)

    com_vel_i = com_vel.unsqueeze(2)
    com_vel_j = com_vel.unsqueeze(1)
    com_rel_vel_vec = com_vel_i - com_vel_j
    com_rel_dir = com_rel_pos / rel_dist_com.unsqueeze(-1)
    com_rel_vel_signed = (com_rel_vel_vec * com_rel_dir).sum(dim=-1)
    com_disp = com_rel_vel_signed * delta_t

    rel_mean_vel_osc = delta_t*(torch.abs(osc_vel_i.norm(dim=-1))+torch.abs(osc_vel_j.norm(dim=-1)))/2
    rel_mean_vel_com = delta_t*(torch.abs(com_vel_i.norm(dim=-1))+torch.abs(com_vel_j.norm(dim=-1)))/2

    osc_scaled = osc_disp / rel_mean_vel_osc
    com_scaled = com_disp / rel_mean_vel_com

    mask = ~torch.eye(n_nodes, dtype=torch.bool, device=oscillator_velocity.device)
    osc_masked = osc_scaled[:, mask]
    com_masked = com_scaled[:, mask]

    loss = F.mse_loss(osc_masked.clamp(min=-1, max=1), com_masked.clamp(min=-1, max=1))
    return loss


def attention_velocity_from_observation_velocity(vae, o_curr, o_dot):
    """Maps observation-space velocity to attention velocity via the attention function's Jacobian."""
    def attention_function(obs):
        if vae.is_vae:
            mu, _ = vae.encode(obs)
        else:
            mu = vae.encode(obs)
        attn, background = vae.decoder.get_attention_weights(mu, return_background_weights=True, upsample=True)
        return attn, background

    _, (attn_dot, background_dot) = torch.autograd.functional.jvp(
        attention_function, (o_curr,), (o_dot,), create_graph=True, strict=False
    )
    return attn_dot, background_dot


def attention_consistency_loss_via_velocity(obs_dot, attn_dot, background_dot):
    """Loss based on attention velocity consistency - computed per latent dimension."""
    obs_diff = torch.abs(obs_dot)
    obs_diff_norm = torch.norm(obs_diff, dim=1, keepdim=True)
    obs_diff_norm = obs_diff_norm / (obs_diff_norm.max() + 1e-8)
    
    attn_velocity_mag = torch.abs(attn_dot)
    consistency_loss = attn_velocity_mag * (1 - obs_diff_norm)
    return consistency_loss.mean()


class VAE(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=32, is_vae=True, use_attention_decoder=True, attention_feature_dim=16, attention_downsample_factor=2, background_token_value=0.0, image_size=32, dynamics_type="harmonic1d", gumbel_noise_strength=0.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.is_vae = is_vae
        self.image_size = image_size
        self.encoder = create_encoder(input_channels, latent_dim, is_vae, image_size)
        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.decoder = create_broadcast_decoder(
                latent_dim, input_channels, image_size=image_size, 
                feature_dim=attention_feature_dim, background_token_value=background_token_value, 
                attention_downsample_factor=attention_downsample_factor, 
                dynamics_type=dynamics_type, gumbel_noise_strength=gumbel_noise_strength
            )
        else:
            self.decoder = create_decoder(latent_dim, input_channels, image_size)
        
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
        """Encode input to latent parameters."""
        h = self.encoder(x)
        if self.is_vae:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar
        else:
            mu = h
            return mu
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to output."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        if self.is_vae:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), z, mu, logvar
        else:
            mu = self.encode(x)
            return self.decode(mu), mu 


    def latent_velocity_from_observation_velocity(self, o_curr: torch.Tensor, o_dot: torch.Tensor) -> torch.Tensor:
        """Maps observation-space velocity to latent velocity via the encoder's Jacobian."""
        def encoder_mu(inp: torch.Tensor) -> torch.Tensor:
            h = self.encoder(inp)
            if self.is_vae:
                mu, _ = h.chunk(2, dim=1)
            else:
                mu = h
            return mu

        _, z_dot = torch.autograd.functional.jvp(encoder_mu, (o_curr,), (o_dot,), create_graph=False, strict=False)
        return z_dot


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld


class KoopmanModel(nn.Module):

	def __init__(self, config: dict, init_K: str = "identity"):
		super().__init__()
		self.latent_dim = config["latent_dim"]
		self.state_dim = 2 * self.latent_dim
		self.actuation_dim = config["actuation_dim"]

		self.K = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
		if init_K == "identity":
			nn.init.eye_(self.K)
		elif init_K == "zeros":
			nn.init.zeros_(self.K)
		else:
			nn.init.xavier_uniform_(self.K)

		if self.actuation_dim > 0:
			self.control_to_state = nn.Sequential(
				nn.Linear(self.actuation_dim, 16),
				nn.LeakyReLU(),
				nn.Linear(16, self.state_dim),
			)
		else:
			self.control_to_state = None

	def forward(self, z_t: torch.Tensor, z_dot_t: torch.Tensor, u_t: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
		x_t = torch.cat([z_t, z_dot_t], dim=1)
		kx = x_t @ self.K.T

		if self.control_to_state is not None and u_t is not None:
			bu = self.control_to_state(u_t)
		else:
			bu = torch.zeros_like(kx)

		z_tp1 = kx[..., :self.latent_dim] + bu[..., :self.latent_dim]
		z_dot_tp1 = kx[..., self.latent_dim:] + bu[..., self.latent_dim:]
		return z_tp1, z_dot_tp1

	def ldm_loss(self, o, z_t, z_dot_t, u_t, VAE, delta_t):
		z_tp1_pred, z_dot_tp1_pred = self.forward(z_t, z_dot_t, u_t, delta_t)

		consistency_loss1 = nn.functional.mse_loss(z_tp1_pred[:-1], z_t[1:])
		consistency_loss2 = nn.functional.mse_loss(z_dot_tp1_pred[:-1] * delta_t, z_dot_t[1:] * delta_t)
		consistency_loss = consistency_loss1 + consistency_loss2

		recon_o_tp1_pred = VAE.decode(z_tp1_pred)
		recon_loss = nn.functional.mse_loss(recon_o_tp1_pred[:-1], o[1:])

		return recon_loss, consistency_loss


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
        self.eps1 = 1e-6
        self.eps2 = 2e-6

        if "harmonic2d" in self.dynamics_type:
            self.minv_diag_raw = nn.Parameter(torch.empty(self.n//2))
        else:
            self.minv_diag_raw = nn.Parameter(torch.empty(self.n))

        self.m_scale = nn.Parameter(torch.ones(1))
        nn.init.ones_(self.minv_diag_raw)
        
        self.k_raw = nn.Parameter(torch.empty(self.n, self.n))
        self.k_ground_raw = nn.Parameter(torch.zeros(self.n))
        self.k_scale = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.k_raw)

        if self.harmonic_damping_type == "full":
            self.d_raw = nn.Parameter(torch.empty(self.n, self.n))
            self.d_ground_raw = nn.Parameter(torch.zeros(self.n))
            self.d_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.d_raw)
        elif self.harmonic_damping_type == "rayleigh":
            self.alpha_raw = nn.Parameter(torch.zeros(1))
            self.beta_raw = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown damping type: {self.harmonic_damping_type}")

        if "harmonic2d" in self.dynamics_type:
            self.x0 = nn.Parameter(torch.zeros(self.n))
        else:
            self.x0 = torch.zeros(self.n, device=device)

        if self.harmonic_use_nonlinear_forcing:
            self.W_raw = nn.Parameter(torch.empty(self.n, self.n))
            nn.init.xavier_uniform_(self.W_raw)
            self.b_raw = nn.Parameter(torch.zeros(self.n))

        if self.m_in > 0:
            self.control_to_state = nn.Sequential(
                nn.Linear(self.m_in, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU(),
                nn.Linear(32, self.n),
            )
        else:
            self.control_to_state = None

    def build_physical_coupling_matrix(self, U_raw, ground_raw=None):
        K_pair = F.softplus(U_raw)
        K_pair = 0.5 * (K_pair + K_pair.T)
        K_pair.fill_diagonal_(0.0)
        row_sum = torch.sum(K_pair, dim=1)
        K_ground = F.softplus(ground_raw) if ground_raw is not None else torch.zeros_like(row_sum)
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


    def build_continuous_AB(self):
        """
        Build continuous-time state-space matrices A and B
        for dx/dt = v, dv/dt = M^-1 (F - K x - D v)
        Using the paper's approach with learned M_inv, K, D matrices
        """
        M_inv, K, D = self.give_Minv_KD()

        Z = torch.zeros(self.n, self.n, device=M_inv.device)
        I = torch.eye(self.n, device=M_inv.device)

        A_top = torch.cat([Z, I], dim=1)
        A_bot = torch.cat([-M_inv @ K, -M_inv @ D], dim=1)
        A_sys = torch.cat([A_top, A_bot], dim=0)

        # B_sys: control (MLP output F) acts on acceleration
        B_sys = torch.cat([
            torch.zeros(self.n, self.n, device=M_inv.device),
            M_inv
        ], dim=0)

        # Also return K and self.x0 for use in step
        return A_sys, B_sys, K

    def step(self, y, u, dt):
        """
        Step dynamics forward using matrix exponential (ZOH)
        y: [batch, 2n] state [x, v]
        u: [batch, m_in] control input
        Adds Kx0 to the control force.
        """

        A_sys, B_sys, K = self.build_continuous_AB()
        n2 = A_sys.shape[0]
        device = A_sys.device

        # MLP -> external force
        bu = self.control_to_state(u)  # [batch, n]

        # Add Kx0 to the control force
        kx0 = (K @ self.x0).unsqueeze(0)  # [1, n]
        forces = bu + kx0  # [batch, n] + [1, n] -> [batch, n]

        Phi = torch.matrix_exp(A_sys * dt)
        eye = torch.eye(n2, device=device)

        # Gamma = integral_0^dt exp(A t) dt B_sys
        # Using solve for stability: A Gamma = (Phi - I) B
        Gamma = torch.linalg.solve(A_sys, (Phi - eye) @ B_sys)

        # batch update
        y_next = (Phi @ y.T).T + (Gamma @ forces.T).T
        return y_next

    def step_symplectic_euler(self, y, u, dt):
        """
        Step dynamics forward using symplectic Euler integration
        y: [batch, 2n] state [x, v]
        u: [batch, m_in] control input
        Adds Kx0 to the control force.
        """
        x, v = y[:, :self.n], y[:, self.n:]
        M_inv, K, D = self.give_Minv_KD()
        
        # Handle actuation
        if self.control_to_state is not None and u is not None:
            bu = self.control_to_state(u)
        else:
            # No actuation: zero force
            batch_size = y.shape[0]
            bu = torch.zeros(batch_size, self.n, device=y.device)

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

    def rollout(self, x0, v0, u_seq, dt):
        y = torch.cat([x0, v0], dim=1)
        xs, vs = [], []
        for u in u_seq:
            y = self.step(y, u, dt)
            xs.append(y[:, :self.n])
            vs.append(y[:, self.n:])
        xs = torch.stack(xs, dim=0)
        vs = torch.stack(vs, dim=0)
        return xs, vs

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

class HarmonicOscillatorDynamics(nn.Module):
    def __init__(self, config: dict, device='cpu'):
        super().__init__()

        self.osc_net = FullyCoupledOscNet(config, device)
        self.latent_dim = config["latent_dim"]
        self.harmonic_prediction_function = config["harmonic_prediction_function"]

    def forward(self, z_t, z_dot_t, u_t, dt=1.0):
        y_t = torch.cat([z_t, z_dot_t], dim=1)
        if self.harmonic_prediction_function == "analytical":
            y_tp1 = self.osc_net.step(y_t, u_t, dt)
        elif self.harmonic_prediction_function == "symplectic_euler":
            y_tp1 = self.osc_net.step_symplectic_euler(y_t, u_t, dt)
        else:
            raise ValueError(
                f"Unsupported harmonic_prediction_function: {self.harmonic_prediction_function}. "
                "Use 'analytical' or 'symplectic_euler'."
            )

        z_tp1 = y_tp1[:, :self.latent_dim]
        z_dot_tp1 = y_tp1[:, self.latent_dim:]
        return z_tp1, z_dot_tp1


    def ldm_loss(self, o: torch.Tensor, z_t: torch.Tensor, z_dot_t: torch.Tensor, u_t: torch.Tensor, VAE, delta_t) -> torch.Tensor:
        """
        Compute loss for harmonic oscillator dynamics learning.
        
        Args:
            o: Observations [batch, ...]
            z_t: Latent states [batch, latent_dim]
            z_dot_t: Latent velocities [batch, latent_dim]
            u_t: Control inputs [batch, actuation_dim]
            VAE: VAE model for decoding
            delta_t: Time step
        
        Returns:
            recon_loss: Reconstruction loss (per sample)
            consistency_loss: Dynamics consistency loss (per sample)
        """
        z_tp1_pred, z_dot_tp1_pred = self.forward(z_t, z_dot_t, u_t, delta_t)

        # Consistency loss: predicted vs actual next states
        consistency_loss1 = torch.nn.functional.mse_loss(
            z_tp1_pred[:-1], z_t[1:]
        )

        consistency_loss2 = torch.nn.functional.mse_loss(
            z_dot_tp1_pred[:-1]*delta_t, z_dot_t[1:]*delta_t
        )

        consistency_loss = consistency_loss1 + consistency_loss2

        # Decode using only the z component of the predicted state
        recon_o_pred = VAE.decode(z_tp1_pred)
        recon_loss = torch.nn.functional.mse_loss(
            recon_o_pred[:-1], o[1:]
        )

        return recon_loss, consistency_loss

def create_dynamics_model(config: dict, device='cpu'):
    """
    Factory function to create the appropriate dynamics model.
    
    Args:
        dynamics_type: 'koopman', 'harmonic', 'actuation_harmonic', or 'physics_koopman'
        latent_dim: Dimension of latent space
        actuation_dim: Dimension of actuation space
        device: Device to place the model on
    
    Returns:
        Dynamics model (KoopmanModel, HarmonicOscillatorDynamics, ActuationDependentHarmonicDynamics, or PhysicsInformedKoopman)
    """
    if config["dynamics_type"].lower() == 'koopman':
        return KoopmanModel(config)
    elif "harmonic" in config["dynamics_type"].lower():
        return HarmonicOscillatorDynamics(config, device)
    else:
        raise ValueError(f"Unknown dynamics_type: {config['dynamics_type']}. Must be 'koopman', 'harmonic', '2dharmonic'")
