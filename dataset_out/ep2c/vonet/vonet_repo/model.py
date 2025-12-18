## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Utility functions
def init_weights(module, init_fn=nn.init.kaiming_normal_):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        return self.relu(out)

# CNN Backbone: Simple ResNet-like feature extractor
class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, base_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResidualBlock(base_channels)
        self.layer2 = ResidualBlock(base_channels)
        # Final conv to get desired feature_dim
        self.final_conv = nn.Conv2d(base_channels, feature_dim, kernel_size=1)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x):
        x = self.initial(x)  # [B, base_channels, H/4, W/4]
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.final_conv(x)  # [B, feature_dim, H/4, W/4]
        return features

# U-Net with transformer bottleneck
class UNet(nn.Module):
    def __init__(self, in_channels: int=128+128, base_channels: int=64, depth: int=5, num_slots: int=11):
        super().__init__()
        self.depth = depth
        self.num_slots = num_slots
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            block = nn.Sequential(
                nn.Conv2d(in_channels if i==0 else channels, channels*2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels*2, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.downs.append(block)
            channels *= 2
        # Bottleneck transformer
        self.bottleneck_channels = channels
        self.transformer_decoder = MaskTransformerDecoder(num_slots=num_slots, feature_dim=channels, num_layers=3, n_heads=3)
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(depth):
            up_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels//2, affine=True),
                nn.ReLU(inplace=True)
            )
            self.ups.append(up_block)
            channels //= 2
        # Final conv to produce mask logits
        self.final_conv = nn.Conv2d(channels, num_slots+1, kernel_size=1)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x, slot_contexts):
        # x: backbone features [B, C, H, W]
        # slot_contexts: list of context vectors for each slot: shape [B, K, context_dim]
        skip_connections = []
        out = x
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
        # Prepare shared features for transformer
        feat_shape = out.shape  # [B, C, H', W']
        B, C, Hp, Wp = feat_shape
        # Expand slot contexts for communication (B,K,C)
        # Concatenate slot contexts along batch dimension for transformer
        # Generate initial slot mask estimates (delta logs) in a way that can be refined by the U-Net
        # For simplicity, assume the delta mask is output of U-Net from initial input (see authors' proposal)
        # but here we process using a shared transformer bottleneck
        # We flatten spatial dimensions
        feat_seq = rearrange(out, 'b c h w -> b (h w) c')
        # Feed into transformer decoder to get slot embeddings
        slot_embeddings = self.transformer_decoder(feat_seq)
        # Reshape back to spatial map per slot
        slot_embeddings = rearrange(slot_embeddings, 'b (h w) k d -> b k h w d', h=Hp, w=Wp)
        # Generate mask logits for each slot
        mask_logits = self.final_conv(slot_embeddings)  # [B, K+1, H, W]
        # Compute softmax across slot dimension (excluding background? or include background as extra slot)
        mask_prob = F.softmax(mask_logits, dim=1)
        return mask_prob

# Transformer decoder for masks' interaction
class MaskTransformerDecoder(nn.Module):
    def __init__(self, num_slots: int=11, feature_dim: int=64, num_layers: int=3, n_heads: int=3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_heads, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Initialize learnable query tokens for slots if needed, but in this case, input is features
        # We assume input features are already prepared with spatial flattened sequence
        # For simplicity, no positional encoding is used
    def forward(self, feat_sequence):
        # feat_sequence: [B, H*W, C]
        # For this module, use a placeholder learnable token: here, treat feat_sequence as the keys & queries is feat_sequence
        # In design, avoid learnable tokens, just pass the features directly
        # For matching the description, suppose we process the features with a set of slot tokens:
        # Noticing that authors communicate among slots at bottleneck, so the input is features per spatial location
        # For implementation, we just pass features through transformer decoder
        # For more fidelity, implement a set of queries per slot
        # For simplicity, assume each slot corresponds to a query: use slot embeddings as queries
        # Since we're not storing them globally here, we'll implement a set of learnable slot queries
        # but as per design, the query vectors are learned or fixed; assuming fixed:
        # Let's implement slot queries as a parameter
        self.slot_queries = getattr(self, 'slot_queries', None)
        if self.slot_queries is None:
            self.slot_queries = nn.Parameter(torch.randn(1, self.transformer.num_layers, feat_sequence.shape[-1]))
        B = feat_sequence.shape[0]
        query = self.slot_queries.expand(B, -1, -1)  # [B, K, C]
        # expand to (K, B, C) for transformer
        query = rearrange(query, 'b k c -> k b c')
        memory = rearrange(feat_sequence, 'b s c -> s b c')  # transpose for src memory
        # Use transformer decoder
        out = self.transformer(tgt=query, memory=memory)
        # out: [K, B, C]
        out = rearrange(out, 'k b c -> b k c')
        return out

# Slot Encoder: extract per-slot features from masked features
class SlotEncoder(nn.Module):
    def __init__(self, feature_dim: int=128, slot_feature_dim: int=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, slot_feature_dim)
        )
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, features, masks):
        # features: [B, C, H, W]
        # masks: [B, K, H, W]
        B, C, H, W = features.shape
        K = masks.shape[1]
        # Expand features for masking
        features_exp = features.unsqueeze(1).expand(-1, K, -1, -1, -1)  # [B, K, C, H, W]
        masks_exp = masks.unsqueeze(2)  # [B, K, 1, H, W]
        masked_feats = features_exp * masks_exp  # [B, K, C, H, W]
        # Average pooling over spatial dims
        pooled_feats = masked_feats.view(B, K, C, -1).mean(-1)  # [B, K, C]
        # Pass through MLP
        slot_feats = self.mlp(pooled_feats)  # [B, K, slot_feature_dim]
        return slot_feats

# Slot Trajectory RNN - GRU with LayerNorm
class SlotTrajectoryRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, slot_feats, r_prev):
        # slot_feats: [B, K, input_dim]
        # r_prev: [B, K, hidden_dim]
        # Process each slot independently
        B, K, D = slot_feats.shape
        r_prev_flat = r_prev.reshape(B*K, -1)  # [B*K, D]
        slot_feats_flat = slot_feats.reshape(B*K, -1)  # [B*K, D]
        # Concatenate slot feature + previous state
        input_seq = slot_feats_flat.unsqueeze(1)  # [B*K, 1, D]
        r_prev_seq = r_prev_flat.unsqueeze(1)  # same shape
        # Run through GRU
        # We can model each slot as a sequence length 1 for simplicity
        r_output, _ = self.gru(input_seq, r_prev_seq.unsqueeze(0))
        r_new = r_output.squeeze(1)  # [B*K, hidden_dim]
        r_new = self.layer_norm(r_new + self.mlp(r_new))
        # Reshape back
        r_new = r_new.view(B, K, -1)
        return r_new

# Variational posterior encoder for z_{t,k}
class VariationalPosterior(nn.Module):
    def __init__(self, slot_feature_dim=128, latent_dim=128):
        super().__init__()
        self.mu_layer = nn.Linear(slot_feature_dim, latent_dim)
        self.logvar_layer = nn.Linear(slot_feature_dim, latent_dim)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, slot_state):
        # slot_state: [B, K, slot_feature_dim]
        mu = self.mu_layer(slot_state)  # [B, K, latent_dim]
        logvar = self.logvar_layer(slot_state)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

# Prior transformer to predict r'_{t,k}
class PriorTransformer(nn.Module):
    def __init__(self, num_slots: int=11, slot_dim: int=128, num_layers: int=2, n_heads: int=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=n_heads, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_mu = nn.Linear(slot_dim, slot_dim)
        self.mlp_logvar = nn.Linear(slot_dim, slot_dim)
        self.num_slots = num_slots
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, prev_r):
        # prev_r: [B, K, D]
        B, K, D = prev_r.shape
        # No positional encoding, raw input
        r_seq = prev_r  # [B, K, D]
        r_seq = r_seq.permute(1, 0, 2)  # [K, B, D]
        encoded = self.transformer(r_seq)  # [K, B, D]
        encoded = encoded.permute(1, 0, 2)  # [B, K, D]
        mu_prior = self.mlp_mu(encoded)  # [B, K, D]
        logvar_prior = self.mlp_logvar(encoded)
        return mu_prior, logvar_prior

# Transformer-based Scene Decoder (autoregressive)
class SceneDecoder(nn.Module):
    def __init__(self, slot_num, slot_dim=128, decoder_layers=3, decoder_heads=3, image_size=128):
        super().__init__()
        self.slot_num = slot_num
        self.slot_dim = slot_dim
        self.image_size = image_size
        # We flatten the image into patches for autoregressive decoding
        self.patch_size = 8  # e.g., 8x8 patches -> (128/8)=16 patches
        self.num_patches = (image_size // self.patch_size) ** 2
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        # Input: concatenate slot embeddings to scene tokens
        self.scene_tokens = nn.Parameter(torch.randn(1, self.num_patches, slot_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=slot_dim, nhead=decoder_heads, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        # Output projection to RGB (or features)
        self.output_projection = nn.Linear(slot_dim, 3 * self.patch_size * self.patch_size)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, z_slots, x_prev=None):
        # z_slots: [B, K, D]
        B, K, D = z_slots.shape
        # Expand slot embeddings for each scene patch
        # For simplicity, expand z_slots over patches
        z_expanded = z_slots.unsqueeze(2).expand(-1, -1, self.num_patches, -1)  # [B, K, num_patches, D]
        # Flatten across slots and patches
        scene_queries = z_expanded.view(B, self.num_patches * K, D).permute(1, 0, 2)  # [skip: can process individually]
        # The decoder attends over all slot embeddings for each patch
        # Generate initial scene tokens, possibly learned or fixed
        scene_tokens = self.scene_tokens.expand(B, -1, -1)  # [B, num_patches, D]
        scene_tokens = scene_tokens.permute(1, 0, 2)  # [num_patches, B, D]
        # Decode scene patches autoregressively
        decoded = self.transformer_decoder(tgt=scene_tokens, memory=scene_queries)
        # decoded: [num_patches, B, D]
        decoded = decoded.permute(1, 0, 2)  # [B, num_patches, D]
        # Map to pixel patches
        patches = self.output_projection(decoded)  # [B, num_patches, 3*patch_size*patch_size]
        # Reshape to image
        batch_img = self._assemble_image_from_patches(patches, B)
        return batch_img

    def _assemble_image_from_patches(self, patches, B):
        # Convert patches to [B, 3, H, W]
        patches = patches.view(B, self.num_patches, 3, self.patch_size, self.patch_size)
        # Reassemble patches into full image
        h_blocks = w_blocks = int(self.image_size / self.patch_size)
        img = torch.zeros(B, 3, self.image_size, self.image_size, device=patches.device)
        idx = 0
        for i in range(h_blocks):
            for j in range(w_blocks):
                img[:, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patches[:, idx]
                idx += 1
        return img

# Main VONet Model class
class VONet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Config parsing
        self.num_slots = config.get('model', {}).get('slot_number', 11)
        self.slot_dim = config.get('model', {}).get('slot_embedding_dim', 128)
        self.feature_dim = 128
        self.backbone_channels = 64
        self.attention_unet_depth = config.get('model', {}).get('attention_unet_depth', 5)
        self.attention_unet_channels = config.get('model', {}).get('attention_unet_channels', 64)
        self.transformer_mask_layers = config.get('model', {}).get('transformer_mask_layers', 3)
        self.transformer_mask_heads = config.get('model', {}).get('transformer_mask_heads', 3)
        self.transformer_prior_layers = config.get('model', {}).get('transformer_prior_layers', 2)
        self.transformer_prior_heads = config.get('model', {}).get('transformer_prior_heads', 3)
        self.decoder_layers = config.get('model', {}).get('decoder_layers', 3)
        self.decoder_heads = config.get('model', {}).get('decoder_heads', 3)
        self.image_size = 128

        # Shared feature extractor
        self.backbone = CNNBackbone(in_channels=3, feature_dim=self.feature_dim, base_channels=self.backbone_channels)

        # Attention module
        self.attention_unet = UNet(in_channels=self.feature_dim + self.slot_dim,
                                     base_channels=self.attention_unet_channels,
                                     depth=self.attention_unet_depth,
                                     num_slots=self.num_slots)
        # Slot encoder
        self.slot_encoder = SlotEncoder(feature_dim=self.feature_dim, slot_feature_dim=self.slot_dim)

        # Slot trajectory RNN (GRU + MLP + LayerNorm)
        self.slot_rnn = SlotTrajectoryRNN(input_dim=self.slot_dim, hidden_dim=self.slot_dim)

        # Variational encoder for z_{t,k}
        self.posterior_z = VariationalPosterior(slot_feature_dim=self.slot_dim, latent_dim=self.slot_dim)

        # Prior transformer for r'_{t,k}
        self.prior_transformer = PriorTransformer(num_slots=self.num_slots,
                                                  slot_dim=self.slot_dim,
                                                  num_layers=self.transformer_prior_layers,
                                                  n_heads=self.transformer_prior_heads)
        # Scene decoder
        self.scene_decoder = SceneDecoder(self.num_slots, slot_dim=self.slot_dim,
                                          decoder_layers=self.decoder_layers,
                                          decoder_heads=self.decoder_heads,
                                          image_size=self.image_size)

        # Initialization
        self.apply_weights()

        # Placeholder for previous slot states (r_{t-1,k})
        self.register_buffer('initial_slot_states', torch.randn(1, self.num_slots, self.slot_dim))

    def apply_weights(self):
        init_weights(self)

    def extract_features(self, x):
        """Extract backbone features from input images."""
        return self.backbone(x)

    def generate_attention(self, features, c_prev):
        """Generate masks for all slots using parallel attention module."""
        masks = self.attention_unet(features, c_prev)
        # masks: shape [B, K+1, H, W]
        return masks

    def encode_slots(self, features, masks):
        """Extract per-slot features from the features weighted by attention masks."""
        slot_feats = self.slot_encoder(features, masks)
        return slot_feats

    def update_slot_states(self, slot_feats, r_prev):
        """Update slot states using RNN/GRU and residual connection."""
        r_new = self.slot_rnn(slot_feats, r_prev)
        return r_new

    def compute_z_posterior(self, r_tk):
        """Compute the variational posterior (q) parameters."""
        mu, logvar = self.posterior_z(r_tk)
        z = self.posterior_z.sample(mu, logvar)
        return z, mu, logvar

    def predict_slot_prior(self, r_prev):
        """Predict the future slot states using prior transformer."""
        mu_prior, logvar_prior = self.prior_transformer(r_prev)
        # Sample from prior
        std = torch.exp(0.5 * logvar_prior)
        epsilon = torch.randn_like(std)
        r_prior = mu_prior + std * epsilon
        return r_prior, mu_prior, logvar_prior

    def decode_scene(self, z_slots):
        """Decode scene from slot embeddings."""
        scene_rec = self.scene_decoder(z_slots)
        return scene_rec

    def forward(self, x, r_prev, c_prev):
        """
        x: input image tensor [B, 3, H, W]
        r_prev: previous slot states [B, K, D]
        c_prev: context vectors for each slot [B, K, D]
        """
        features = self.extract_features(x)  # shape [B, C, H', W']
        masks = self.generate_attention(features, c_prev)  # [B, K+1, H, W]
        # Extract foreground masks (excluding null/background at index 0)
        masks_probs = masks[:, 1:, :, :]  # [B, K, H, W]
        # Normalize masks to sum to 1 + background, but authors use softmax separately
        # For inference, can use these masks directly
        slot_features = self.encode_slots(features, masks[:,1:,:,:])  # [B,K,128]
        r_t = self.update_slot_states(slot_features, r_prev)  # [B,K,128]
        # Variational encoding
        z_t, mu_z, logvar_z = self.compute_z_posterior(r_t)  # [B,K,128]
        # Prior prediction
        r_prior, mu_prior, logvar_prior = self.predict_slot_prior(r_prev)
        # Decode scene
        recon_scene = self.decode_scene(z_t)
        return recon_scene, masks, r_t, r_prior, mu_z, logvar_z, mu_prior, logvar_prior

# Initialize the network with configuration
def build_vonet_from_config(config: dict) -> VONet:
    return VONet(config)
