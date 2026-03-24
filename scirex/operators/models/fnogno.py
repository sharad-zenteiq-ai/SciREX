from typing import Any, Dict, Optional, Tuple, Literal, Callable

import jax 
import jax.numpy as jnp
from flax import linen as nn

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlock
from ..layers.gno_block import GNOBlock

class FNOGNO(nn.Module):
    """
    FNOGNO: Fourier/Geometry Neural Operator. 
    Maps from a regular N-d grid to an arbitrary query point cloud.
    (Contains FNO + Output GNO)
    """
    in_channels: int
    out_channels: int
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 3
    gno_pos_embed_type: str = "transformer"
    gno_transform_type: str = "linear"
    fno_n_modes: Tuple[int, ...] = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4
    
    # Other GNO params
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000
    gno_radius: float = 0.033
    gno_channel_mlp_hidden_layers: Tuple[int, ...] = (512, 256)
    gno_channel_mlp_non_linearity: Callable = nn.gelu
    
    # Other FNO params
    fno_use_channel_mlp: bool = True
    fno_non_linearity: Callable = nn.gelu
    fno_norm: bool = False
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    fno_channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"

    # Neighbor limit
    max_neighbors: int = 10

    @nn.compact
    def __call__(
        self,
        in_p: jnp.ndarray,
        out_p: jnp.ndarray,
        f: jnp.ndarray,
        ada_in: Optional[jnp.ndarray] = None,
        out_neighbors: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        in_p : (batch, n_x, n_y, n_z, coord_dim) latent grid coordinates.
        out_p: (batch, n_out, coord_dim) output point cloud
        f: (batch, n_x, n_y, n_z, in_channels) features on the latent grid
        ada_in: Not heavily utilized by default, here for parity.
        out_neighbors: Optional precomputed topological neighbors for GNO.
        """
        batch_size = 1 if f.ndim == 4 else f.shape[0] 
        
        # 1. Latent Embedding
        in_p_features = jnp.concatenate([f, in_p], axis=-1)
        
        lifting_hidden = self.fno_lifting_channel_ratio * self.fno_hidden_channels
        x_latent = ChannelMLP(
            out_channels=self.fno_hidden_channels,
            hidden_channels=lifting_hidden,
            n_layers=3,
            activation=self.fno_non_linearity,
        )(in_p_features)

        # Apply FNO blocks
        for _ in range(self.fno_n_layers):
            x_latent = FNOBlock(
                hidden_channels=self.fno_hidden_channels,
                n_modes=self.fno_n_modes,
                activation=self.fno_non_linearity,
                use_norm=self.fno_norm,
                skip_type=self.fno_skip,
                channel_mlp_skip=self.fno_channel_mlp_skip,
                use_channel_mlp=self.fno_use_channel_mlp,
            )(x_latent)

        # 2. Integrate Latent
        # Flatten for GNO: (batch, n_x*n_y*n_z, channels)
        x_latent_flat = x_latent.reshape(batch_size, -1, self.fno_hidden_channels)
        in_p_flat = in_p.reshape(batch_size, -1, self.gno_coord_dim)
        
        # Process geometries for GNOBlock (which expects unbatched graph topologies generally)
        if in_p_flat.ndim > 2 and in_p_flat.shape[0] == 1:
            in_p_flat = jnp.squeeze(in_p_flat, 0)
        
        if out_p.ndim > 2 and out_p.shape[0] == 1:
            out_p = jnp.squeeze(out_p, 0)
            
        gno = GNOBlock(
            in_channels=self.fno_hidden_channels,
            out_channels=self.fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.gno_radius,
            reduction="sum", 
            pos_embedding_type=self.gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            channel_mlp_layers=list(self.gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.gno_transform_type,
            max_neighbors=self.max_neighbors,
        )

        # Evaluate Output GNO
        out = gno(
            y=in_p_flat,
            x=out_p,
            f_y=x_latent_flat,
            neighbors=out_neighbors
        )
        
        if out.ndim == 2:
            out = jnp.expand_dims(out, 0)

        # 3. Final Pointwise Projection
        proj_hidden = self.projection_channel_ratio * self.fno_hidden_channels
        out = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=proj_hidden,
            n_layers=2,
            activation=self.fno_non_linearity,
        )(out)

        return out
