from typing import Optional
from flax import linen as nn
import jax.numpy as jnp

class Projection(nn.Module):
    """
    Projection layer: projects the hidden representation to the desired output channels.
    
    Supports two modes:
    - Single-layer: Dense(width -> out_channels)
    - Two-layer MLP: Dense(width -> hidden_dim) + activation + Dense(hidden_dim -> out_channels)
      This matches the original WNO paper's projection (fc1 + GeLU + fc2).
    
    Parameters
    ----------
    out_channels : int
        Number of output channels.
    hidden_dim : int, optional
        If provided, uses a 2-layer MLP projection with this hidden dimension.
        The original WNO paper uses hidden_dim=192.
    activation : str
        Activation function between the two layers. Default is 'gelu'.
    """
    out_channels: int
    hidden_dim: Optional[int] = None
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.hidden_dim is not None:
            x = nn.Dense(self.hidden_dim)(x)
            if self.activation == "gelu":
                x = nn.gelu(x)
            elif self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "mish":
                x = x * jnp.tanh(nn.softplus(x))
        return nn.Dense(self.out_channels)(x)
