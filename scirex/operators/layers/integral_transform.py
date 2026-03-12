from typing import Optional, Callable
import jax 
import jax.numpy as jnp
from flax import linen as nn

from .channel_mlp import LinearChannelMLP


def segment_csr(data, splits, reduction="sum", valid_counts=None):
    """
    JAX equivalent of segment_csr for CSR neighborhoods.
    
    data: (E, C) or (B, E, C) where E is total edges (m * max_neighbors)
    splits: (M+1,) or (B, M+1) where M is number of queries
    reduction: "sum" or "mean"
    valid_counts: (M,) or (B, M) the actual number of neighbors within radius
    """
    # 1. Determine number of segments (queries)
    num_segments = splits.shape[-1] - 1
    
    if data.ndim == 2:
        # --- Unbatched Case ---
        # Generate row indices for each edge: [0, 0, 0, 1, 1, 1, ...]
        row_ids = jnp.repeat(
            jnp.arange(num_segments),
            splits[1:] - splits[:-1],
            total_repeat_length=data.shape[0]
        )
        
        # Perform the sum
        summed = jax.ops.segment_sum(data, row_ids, num_segments)
        
        if reduction == "sum":
            return summed
        
        # For mean, divide by actual valid neighbors (not max_neighbors)
        if valid_counts is not None:
            # Use max(1, count) to avoid division by zero for empty neighborhoods
            denominator = jnp.maximum(valid_counts, 1.0)[:, None]
        else:
            # Fallback to dividing by the fixed segment size
            denominator = (splits[1:] - splits[:-1])[:, None]
            
        return summed / denominator

    else:
        # --- Batched Case ---
        # Use vmap to handle the batch dimension efficiently
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

    def setup(self):
        assert self.channel_mlp is not None or self.channel_mlp_layers is not None

        if self.channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=self.channel_mlp_layers,
                activation=self.channel_mlp_non_linearity,
            )

    def __call__(self, y, neighbors, x=None, f_y=None, weights=None):
            if x is None:
                x = y
    
            idx = neighbors["neighbors_index"]
            splits = neighbors["neighbors_row_splits"]
            mask = neighbors.get("neighbors_mask") # [E] or [B, E]
    
            rep_features = y[idx] 
            batched = False
            batch_size = 1
            
            if f_y is not None:
                if f_y.ndim == 3:
                    batched = True
                    batch_size = f_y.shape[0]
                    in_features = f_y[:, idx, :]
                else:
                    in_features = f_y[idx]
    
            num_reps = splits[1:] - splits[:-1]
            self_features = jnp.repeat(x, num_reps, axis=0)
            agg = jnp.concatenate([rep_features, self_features], axis=-1)
    
            if f_y is not None and self.transform_type in ("nonlinear_kernelonly", "nonlinear"):
                if batched:
                    agg = jnp.broadcast_to(agg, (batch_size,) + agg.shape)
                agg = jnp.concatenate([agg, in_features], axis=-1)
    
            # 1. Compute Kernel
            rep = self.channel_mlp(agg)
    
            # 2. Apply f_y if linear/nonlinear transform
            if f_y is not None and self.transform_type != "nonlinear_kernelonly":
                if rep.ndim == 2 and batched:
                    rep = jnp.broadcast_to(rep, (batch_size,) + rep.shape)
                rep = rep * in_features
    
            # 3. GLOBAL MASKING (Moved outside f_y check)
            # This ensures radius constraints are applied even if f_y is None
            if mask is not None:
                if batched:
                    rep = rep * mask.reshape(batch_size, -1, 1)
                else:
                    rep = rep * mask[:, None]
    
            # 4. Handle weights
            nbr_weights = neighbors.get("weights")
            if nbr_weights is None: nbr_weights = weights
    
            if nbr_weights is not None:
                if batched: nbr_weights = nbr_weights[None, :, None]
                else: nbr_weights = nbr_weights[:, None]
    
                if self.weighting_fn is not None:
                    nbr_weights = self.weighting_fn(nbr_weights)
    
                rep = rep * nbr_weights
                reduction = "sum"
            else:
                reduction = self.reduction
    
            # 5. Calculate valid_counts for Mean Reduction
            valid_counts = None
            if reduction == "mean" and mask is not None:
                # We need to know how many neighbors were valid per query
                # We assume a fixed width of max_neighbors from your NeighborSearch
                max_nb = num_reps[0] 
                if batched:
                    valid_counts = mask.reshape(batch_size, -1, max_nb).sum(axis=-1)
                else:
                    valid_counts = mask.reshape(-1, max_nb).sum(axis=-1)
    
            if batched:
                splits_b = jnp.broadcast_to(splits, (rep.shape[0],) + splits.shape)
            else:
                splits_b = splits
    
            # 6. Reduce
            out = segment_csr_jax(rep, splits_b, reduction, valid_counts=valid_counts)
            return out
