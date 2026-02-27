import jax.numpy as jnp
import numpy as np

class GaussianNormalizer:
    def __init__(self, x, eps=1e-7):
        """
        x: (n_samples, nx, ny, channels) or any shape where we want to normalize across samples.
        We compute mean and std across the first dimension (n_samples) AND spatial dimensions?
        Usually, for FNO/WNO, normalization is done per channel across all samples and spatial locations.
        """
        # Compute stats over all but the last dimension (channels)
        # or maybe just the first? 
        # Most implementations in this field normalize per-channel across all spatial grid points.
        
        # Determine axes to reduce over (everything except the last channel dimension)
        reduce_axes = tuple(range(len(x.shape) - 1))
        
        self.mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        self.std = jnp.std(x, axis=reduce_axes, keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean
