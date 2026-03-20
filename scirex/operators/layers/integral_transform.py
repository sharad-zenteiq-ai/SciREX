from typing import Optional, Callable
import jax 
import jax.numpy as jnp
from flax import linen as nn

from .channel_mlp import LinearChannelMLP
from .segment_csr import segment_sum, segment_mean


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
        idx = neighbors["neighbor_indices"]  # Expected (M, K)
        mask = neighbors.get("mask") # (M, K)

        # Static neighborhood size for JIT-safety
        num_target_nodes, K = idx.shape[-2:]

        # Squeeze batch dimension from neighbors if present (static geometry assumption)
        if idx.ndim == 3: idx = idx[0]
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
        # Ensure padded index (-1) is safe for gather by replacing it temporarily with 0
        safe_idx = jnp.where(idx == -1, 0, idx)
        rep_features = y[safe_idx].reshape(-1, y.shape[-1])
        
        if x.ndim == 3:
            # Repeat each of the M points K times
            self_features = jnp.repeat(x, K, axis=1) 
        else:
            self_features = jnp.repeat(x, K, axis=0)

        if f_y is not None:
            if f_y.ndim == 3:
                in_features = f_y[:, safe_idx, :].reshape(batch_size, -1, f_y.shape[-1])
            else:
                in_features = f_y[safe_idx].reshape(-1, f_y.shape[-1])
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
        nbr_weights = neighbors.get("distances")
        if nbr_weights is None:
            nbr_weights = weights

        if nbr_weights is not None:
            # Reshape weights to enable multiplication [..., num_edges, 1]
            if nbr_weights.ndim == 1:
                nbr_weights = nbr_weights[:, None]
            elif nbr_weights.ndim == 2:
                nbr_weights = nbr_weights.reshape(-1, 1)
                
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

        # 5. Segment Aggregate
        segment_ids = jnp.repeat(jnp.arange(num_target_nodes), K)
        mask_flat = mask.reshape(-1) if mask is not None else None
        
        if batched:
            if reduction == "mean":
                out = jax.vmap(lambda r: segment_mean(r, segment_ids, num_target_nodes, mask_flat))(rep)
            else:
                out = jax.vmap(lambda r: segment_sum(r, segment_ids, num_target_nodes))(rep)
        else:
            if reduction == "mean":
                out = segment_mean(rep, segment_ids, num_target_nodes, mask_flat)
            else:
                out = segment_sum(rep, segment_ids, num_target_nodes)

        # Optional: Print shapes at runtime internally as debugging utility using jax.debug.print
        # In JAX, shapes are static, so formatting string statically works. 
        # To strictly use `jax.debug.print` for shape validations:
        # Uncomment below lines to enable runtime shape debugging
        # jax.debug.print("neighbor_indices shape: {s}", s=jnp.array(idx.shape))
        # jax.debug.print("weights shape: {s}", s=jnp.array(nbr_weights.shape if nbr_weights is not None else ()))
        # jax.debug.print("outputs shape: {s}", s=jnp.array(out.shape))

        return out