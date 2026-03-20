"""
Segment / CSR Sparse Operations backend for GINO.

Implements segmentation and aggregation utilities strictly constrained by:
- JAX primitives only
- Fixed shapes for JIT compilation
- No Python loops or dynamic boolean masking
"""
import jax
import jax.numpy as jnp

def segment_sum(edge_values: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int):
    """
    Computes the sum of edge values for each segment.
    
    Args:
        edge_values: (N*K, C) feature values for each edge.
        segment_ids: (N*K,) array mapping each edge to a target node.
        num_segments: Number of target nodes (N).
        
    Returns:
        aggregated_values: (N, C)
    """
    return jax.ops.segment_sum(edge_values, segment_ids, num_segments=num_segments)

def segment_mean(edge_values: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int, mask: jnp.ndarray = None):
    """
    Computes the mean of edge values for each segment, accounting for padding.
    
    Args:
        edge_values: (N*K, C)
        segment_ids: (N*K,)
        num_segments: int, Number of target nodes (N)
        mask: optional (N*K,) boolean mask describing valid edges. Padded edges are ignored.
        
    Returns:
        aggregated_values: (N, C)
    """
    # 1. Sum up values as usual
    summed = segment_sum(edge_values, segment_ids, num_segments)
    
    # 2. Compute non-padded element counts per segment
    if mask is not None:
        # Sum the mask (1.0 for valid, 0.0 for invalid)
        valid_counts = jax.ops.segment_sum(mask.astype(edge_values.dtype), segment_ids, num_segments=num_segments)
    else:
        # If no mask, all edges defined by segment_ids are valid
        valid_counts = jax.ops.segment_sum(jnp.ones_like(segment_ids, dtype=edge_values.dtype), segment_ids, num_segments=num_segments)
        
    # Prevent division by zero
    valid_counts = jnp.maximum(valid_counts, 1.0)
    
    if summed.ndim > 1:
        valid_counts = jnp.expand_dims(valid_counts, axis=-1)
        
    return summed / valid_counts
