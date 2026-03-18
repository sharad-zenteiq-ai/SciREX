from typing import Any, Dict, Optional, Tuple, Literal, Callable

import jax 
import jax.numpy as jnp
from flax import linen as nn

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlock
from ..layers.gno_block import GNOBlock

class GINO(nn.Module):
    """
    Graph-Informed Neural Operator (GINO).
    
    GINO couples Graph Neural Networks (GNO) with Fourier Neural Operators (FNO).
    The typical pipeline is:
      1. input GNO: maps arbitrary geometry queries to a regular latent grid.
      2. FNO: spatially propagates features globally over the regular grid.
      3. output GNO: projects the grid-based latent representation back to target geometry.
    """
    # required
    in_channels: int
    out_channels: int

    # GINO core
    latent_feature_channels: Optional[int] = None
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 3
    in_gno_radius: float = 0.033
    out_gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    
    in_gno_pos_embed_type: str = "transformer"
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000

    # FNO parameters
    fno_n_modes: Tuple[int, ...] = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 2
    fno_n_layers: int = 4

    # GNO MLP parameters
    in_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (80, 80, 80)
    out_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (512, 256)
    gno_channel_mlp_non_linearity: Callable = nn.gelu

    out_gno_tanh: Optional[str] = None

    # FNO extras
    fno_use_channel_mlp: bool = True
    fno_non_linearity: Callable = nn.gelu
    fno_norm: bool = False
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    fno_channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"
    
    # Internal neighbor limits (memory bounded in JAX)
    max_neighbors: int = 10

    @nn.compact
    def __call__(
        self,
        input_geom: jnp.ndarray,
        latent_queries: jnp.ndarray,
        output_queries: jnp.ndarray,
        x: Optional[jnp.ndarray] = None,
        latent_features: Optional[jnp.ndarray] = None,
        in_neighbors: Optional[Dict[str, jnp.ndarray]] = None,
        out_neighbors: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        Forward pass of GINO.
        
        input_geom: (batch, n_in, coord_dim) or (n_in, coord_dim)
        latent_queries: (batch, nx, ny, nz, coord_dim) or (nx, ny, nz, coord_dim)
        output_queries: (batch, n_out, coord_dim) or (n_out, coord_dim)
        x: (batch, n_in, in_channels) or (n_in, in_channels)
        latent_features: (batch, nx, ny, nz, latent_feature_channels)
        in_neighbors: Optional dictionary of pre-calculated neighbors for input GNO
        out_neighbors: Optional dictionary of pre-calculated neighbors for output GNO
        """
        batch_size = 1 if x is None else (1 if x.ndim == 2 else x.shape[0])

        # Ensure geometric inputs are consistently batch-free for GNO spatial querying
        if input_geom.ndim == 3: input_geom = jnp.squeeze(input_geom, 0)
        if output_queries.ndim == 3: output_queries = jnp.squeeze(output_queries, 0)

        # Squeeze neighbors if they have an extra batch dimension (1, ...)
        if in_neighbors is not None:
            in_neighbors = jax.tree_util.tree_map(
                lambda arr: jnp.squeeze(arr, 0) if (isinstance(arr, jnp.ndarray) and arr.ndim > 1 and arr.shape[0] == 1) else arr,
                in_neighbors
            )
        if out_neighbors is not None:
            out_neighbors = jax.tree_util.tree_map(
                lambda arr: jnp.squeeze(arr, 0) if (isinstance(arr, jnp.ndarray) and arr.ndim > 1 and arr.shape[0] == 1) else arr,
                out_neighbors
            )
        
        # Determine out channels for input GNO
        in_gno_out = self.in_channels if self.in_gno_transform_type.startswith("nonlinear") else self.fno_hidden_channels

        # 1. Input GNO
        gno_in = GNOBlock(
            in_channels=self.in_channels,
            out_channels=in_gno_out,
            coord_dim=self.gno_coord_dim,
            radius=self.in_gno_radius,
            pos_embedding_type=self.in_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            reduction="mean",
            channel_mlp_layers=list(self.in_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.in_gno_transform_type,
            max_neighbors=self.max_neighbors,
        )

        flat_latent_queries = latent_queries.reshape(-1, latent_queries.shape[-1])
        if flat_latent_queries.ndim == 3:
            flat_latent_queries = jnp.squeeze(flat_latent_queries, 0)
            
        in_p = gno_in(
            y=input_geom,
            x=flat_latent_queries,
            f_y=x,
            neighbors=in_neighbors
        )

        # Restore grid shape (batch, nx, ny, nz, channels)
        grid_shape = latent_queries.shape[:-1]
        if latent_queries.ndim == 5: # Includes batch
            grid_shape = latent_queries.shape[1:-1]
            
        in_p = in_p.reshape((batch_size, *grid_shape, -1))

        if latent_features is not None:
            # Broadcast or align latent features
            if latent_features.ndim == 4: # Missing batch
                latent_features = jnp.expand_dims(latent_features, 0)
            in_p = jnp.concatenate([in_p, latent_features], axis=-1)

        # 2. Lifting -> FNO Blocks
        lifting_hidden = self.fno_lifting_channel_ratio * self.fno_hidden_channels
        x_latent = ChannelMLP(
            out_channels=self.fno_hidden_channels,
            hidden_channels=lifting_hidden,
            n_layers=2,
            activation=self.fno_non_linearity,
        )(in_p)
        
        # Spatial processing through Fourier Layers
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

        # Flatten grid back to points for output GNO (batch, n_pts, channels)
        x_latent = x_latent.reshape(batch_size, -1, self.fno_hidden_channels)
        
        if self.out_gno_tanh in ("latent_embed", "both"):
            x_latent = jnp.tanh(x_latent)

        # 3. Output GNO
        gno_out = GNOBlock(
            in_channels=self.fno_hidden_channels,
            out_channels=self.fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction="sum",
            pos_embedding_type=self.out_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            channel_mlp_layers=list(self.out_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.out_gno_transform_type,
            max_neighbors=self.max_neighbors,
        )

        # Evaluate output GNO
        out_eval = gno_out(
            y=flat_latent_queries, 
            x=output_queries, 
            f_y=x_latent, # Pass full batch
            neighbors=out_neighbors
        )

        # Re-apply batch dimension mapping
        if batch_size > 1 and x_latent.shape[0] > 1: # Naive vmap for multiple identical queries
            pass # VMAP or manual mapping expansion usually applies here. GNOBlock typically takes single instance graphs.
            
        out_eval = jnp.expand_dims(out_eval, 0) if out_eval.ndim == 2 else out_eval

        # 4. Projection
        proj_hidden = self.projection_channel_ratio * self.fno_hidden_channels
        out = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=proj_hidden,
            n_layers=2,
            activation=self.fno_non_linearity,
        )(out_eval)

        return out