from typing import List, Optional, Callable, Literal

import jax.numpy as jnp
from flax import linen as nn

from .channel_mlp import LinearChannelMLP
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch
from .embeddings import SinusoidalEmbedding


class GNOBlock(nn.Module):
    in_channels: int
    out_channels: int
    coord_dim: int
    radius: float

    transform_type: str = "linear"
    weighting_fn: Optional[Callable] = None
    reduction: Literal["sum", "mean"] = "sum"

    pos_embedding_type: Optional[str] = "transformer"
    pos_embedding_channels: int = 32
    pos_embedding_max_positions: int = 10000

    channel_mlp_layers: Optional[List[int]] = None
    channel_mlp_non_linearity: Callable = nn.gelu
    channel_mlp: Optional[nn.Module] = None

    use_torch_scatter_reduce: bool = True
    use_open3d_neighbor_search: bool = True  # ignored in JAX

    max_neighbors: int = 12  # NEW

    def setup(self):
        # =========================
        # Positional embedding
        # =========================
        if self.pos_embedding_type in ("nerf", "transformer"):
            self.pos_embedding = SinusoidalEmbedding(
                num_frequencies=self.pos_embedding_channels,
                embedding_type=self.pos_embedding_type,
                max_positions=self.pos_embedding_max_positions,
            )
            embed_dim = self.coord_dim * self.pos_embedding_channels * 2
        else:
            self.pos_embedding = None
            embed_dim = self.coord_dim

        # =========================
        # Neighbor search
        # =========================
        self.neighbor_search = NeighborSearch(
            max_neighbors=self.max_neighbors,
            return_norm=self.weighting_fn is not None
        )

        # =========================
        # Kernel input dimension
        # =========================
        kernel_in_dim = embed_dim * 2

        if self.transform_type in ("nonlinear", "nonlinear_kernelonly"):
            kernel_in_dim += self.in_channels

        # =========================
        # Channel MLP
        # =========================
        if self.channel_mlp is not None:
            mlp = self.channel_mlp
        else:
            layers = list(self.channel_mlp_layers) if self.channel_mlp_layers else [128, 256, 128]

            if layers[0] != kernel_in_dim:
                layers = [kernel_in_dim] + layers
            if layers[-1] != self.out_channels:
                layers.append(self.out_channels)

            mlp = LinearChannelMLP(
                layers=layers,
                activation=self.channel_mlp_non_linearity,
            )

        # =========================
        # Integral transform
        # =========================
        self.integral_transform = IntegralTransform(
            channel_mlp=mlp,
            transform_type=self.transform_type,
            weighting_fn=self.weighting_fn,
            reduction=self.reduction,
        )

    # =============================
    # Forward
    # =============================
    def __call__(self, y, x, f_y=None):
        """
        y: [n, coord_dim]
        x: [m, coord_dim]
        f_y: [batch, n, in_channels] or [n, in_channels]
        """

        neighbors = self.neighbor_search(
            data=y,
            queries=x,
            radius=self.radius,
        )

        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x

        out = self.integral_transform(
            y=y_embed,
            x=x_embed,
            neighbors=neighbors,
            f_y=f_y,
        )

        return out
