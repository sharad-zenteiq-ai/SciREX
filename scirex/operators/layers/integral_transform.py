from typing import Optional, Callable
import jax 
import jax.numpy as jnp
from flax import linen as nn

from .channel_mlp import LinearChannelMLP


def segment_csr_jax(data, splits, reduction="sum", valid_counts=None):
    """
    JAX equivalent of segment_csr for CSR neighborhoods.
    
    This version is optimized for JIT-safety in fixed-size neighborhoods.
    """
    num_segments = splits.shape[-1] - 1
    
    if data.ndim == 2:
        # --- Unbatched Case ---
        # Instead of using splits (which can be a tracer), we use the total edge count 
        # and assume a fixed-size neighborhood for JIT shape stability.
        num_edges = data.shape[0]
        K = num_edges // num_segments
        
        # row_ids becomes [0,0,0, 1,1,1, ...] etc.
        # Repeating with a static integer K is JIT-safe.
        row_ids = jnp.repeat(jnp.arange(num_segments), K)
        
        # Perform segment sum
        summed = jax.ops.segment_sum(data, row_ids, num_segments)
        
        if reduction == "sum":
            return summed
        
        # For mean, divide by actual valid neighbors
        if valid_counts is not None:
            denominator = jnp.maximum(valid_counts, 1.0)[:, None]
        else:
            denominator = jnp.full((num_segments, 1), K)
            
        return summed / denominator

    else:
        # --- Batched Case ---
        return jax.vmap(
            lambda d, s, v: segment_csr_jax(d, s, reduction, v)
        )(data, splits, valid_counts)


class IntegralTransform(nn.Module):
    channel_mlp: Optional[nn.Module] = None
    channel_mlp_layers: Optional[list] = None
    channel_mlp_non_linearity: Callable = nn.gelu

    transform_type: str = "linear"
    weighting_fn: Optional[Callable] = None
    reduction: str = "sum"

    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

    def setup(self):
        assert self.channel_mlp is not None or self.channel_mlp_layers is not None

        if self.channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=self.channel_mlp_layers,
                activation=self.channel_mlp_non_linearity,
            )
        
        # Add a projection layer if in_channels != out_channels for linear/nonlinear transforms
        if self.in_channels is not None and self.out_channels is not None and self.in_channels != self.out_channels:
            self.projection = nn.Dense(self.out_channels, use_bias=False)
        else:
            self.projection = None

    def __call__(self, y, neighbors, x=None, f_y=None, weights=None):
        if x is None:
            x = y

        # --- Parse Neighbors ---
        idx = neighbors["neighbors_index"]  # Expected (M, K)
        splits = neighbors["neighbors_row_splits"] # (M+1,)
        mask = neighbors.get("neighbors_mask") # (M, K)

        # Static neighborhood size for JIT-safety
        K = idx.shape[-1] 

        # Squeeze batch dimension from neighbors if present (static geometry assumption)
        if idx.ndim == 3: idx = idx[0]
        if splits.ndim == 2: splits = splits[0]
        if mask is not None and mask.ndim == 3: mask = mask[0]
        
        # --- Handle Batching ---
        batched = False
        batch_size = 1
        if f_y is not None and f_y.ndim == 3:
            batched = True
            batch_size = f_y.shape[0]
        elif x.ndim == 3:
            batched = True
            batch_size = x.shape[0]

        # --- Flatten Features to Edge-Wise (CSR) ---
        # Using reshapes and static K ensures JIT compilation success
        rep_features = y[idx].reshape(-1, y.shape[-1])
        
        if x.ndim == 3:
            # Repeat each of the M points K times
            self_features = jnp.repeat(x, K, axis=1) 
        else:
            self_features = jnp.repeat(x, K, axis=0)

        if f_y is not None:
            if f_y.ndim == 3:
                in_features = f_y[:, idx, :].reshape(batch_size, -1, f_y.shape[-1])
            else:
                in_features = f_y[idx].reshape(-1, f_y.shape[-1])
        else:
            in_features = None

        # --- Concatenate Features ---
        if batched:
            if rep_features.ndim == 2:
                rep_features = jnp.broadcast_to(rep_features, (batch_size,) + rep_features.shape)
            if self_features.ndim == 2:
                self_features = jnp.broadcast_to(self_features, (batch_size,) + self_features.shape)
            
            agg = jnp.concatenate([rep_features, self_features], axis=-1)
            if in_features is not None and self.transform_type in ("nonlinear", "nonlinear_kernelonly"):
                agg = jnp.concatenate([agg, in_features], axis=-1)
        else:
            agg = jnp.concatenate([rep_features, self_features], axis=-1)
            if in_features is not None and self.transform_type in ("nonlinear", "nonlinear_kernelonly"):
                agg = jnp.concatenate([agg, in_features], axis=-1)

        # 1. Compute Kernel
        rep = self.channel_mlp(agg)

        # 2. Apply f_y if linear/nonlinear transform
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            # Project in_features if channels don't match
            if self.projection is not None:
                in_features_proj = self.projection(in_features)
            else:
                in_features_proj = in_features
            
            rep = rep * in_features_proj

        # 3. GLOBAL MASKING
        if mask is not None:
            if batched:
                mask_val = mask.reshape(1, -1, 1)
            else:
                mask_val = mask.reshape(-1, 1)
            rep = rep * mask_val

        # 4. Handle weights
        nbr_weights = neighbors.get("neighbors_distance")
        if nbr_weights is None:
            nbr_weights = weights

        if nbr_weights is not None:
            # Reshape weights to enable multiplication [..., num_edges, 1]
            if nbr_weights.ndim == 1:
                nbr_weights = nbr_weights[:, None]
            if batched and nbr_weights.ndim == 2:
                nbr_weights = nbr_weights[None, :, :]
            elif batched and nbr_weights.ndim == 3 and nbr_weights.shape[0] != batch_size:
                nbr_weights = nbr_weights[0:1, :, :] # Static geometry broadcast
                
            if self.weighting_fn is not None:
                nbr_weights = self.weighting_fn(nbr_weights)

            rep = rep * nbr_weights
            reduction = "sum"  
        else:
            reduction = self.reduction

        # 5. Calculate valid_counts for Mean Reduction
        valid_counts = None
        if reduction == "mean" and mask is not None:
            if batched:
                valid_counts = mask.reshape(1, -1, K).sum(axis=-1)
                valid_counts = jnp.broadcast_to(valid_counts, (batch_size, valid_counts.shape[1]))
            else:
                valid_counts = mask.reshape(-1, K).sum(axis=-1)

        if batched:
            splits_b = jnp.broadcast_to(splits, (batch_size,) + splits.shape)
        else:
            splits_b = splits

        out = segment_csr_jax(rep, splits_b, reduction, valid_counts=valid_counts)
        return out