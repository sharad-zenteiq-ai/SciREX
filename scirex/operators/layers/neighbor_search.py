# neighbor_search_jax.py
import jax
import jax.numpy as jnp


class NeighborSearch:
    """
    JIT-safe fixed-size CSR neighbor search (no boolean indexing).
    Optimized to use inner-product formulation for distance computation.
    """

    def __init__(self, max_neighbors: int, return_norm: bool = False):
        self.max_neighbors = max_neighbors
        self.return_norm = return_norm

    def __call__(self, data: jnp.ndarray, queries: jnp.ndarray, radius: float):
        # 1. Compute all pairwise squared distances efficiently using: ||q - d||^2 = ||q||^2 + ||d||^2 - 2(q.d)
        # This replaces `(m, n, d)` intermediate broadcasting with `(m, n)` dot product (fast TensorCores).
        q_sq = jnp.sum(queries ** 2, axis=-1, keepdims=True)  # (m, 1)
        d_sq = jnp.sum(data ** 2, axis=-1, keepdims=True)     # (n, 1)
        dot  = jnp.dot(queries, data.T)                       # (m, n)
        
        # Clip to 0 to prevent negative distances from floating point numerical errors
        dists_sq = jnp.maximum(q_sq + d_sq.T - 2.0 * dot, 0.0) 
        dists = jnp.sqrt(dists_sq)

        # 2. Find the indices of the K-nearest points
        # Use jax.lax.top_k to get the smallest distances (negate them)
        val, neighbors_index = jax.lax.top_k(-dists, self.max_neighbors)
        actual_dists = -val # (m, max_neighbors)
        
        # 3. Create a validity mask: True if distance <= radius
        mask = actual_dists <= radius
        
        # 4. Construct outputs, using fixed-size arrays formatted as CSR
        out = {
            "neighbors_index": neighbors_index.reshape(-1).astype(jnp.int32),
            "neighbors_mask": mask.reshape(-1), 
            "neighbors_row_splits": jnp.arange(0, (queries.shape[0] + 1) * self.max_neighbors, self.max_neighbors).astype(jnp.int32)
        }
        
        # 5. Return distances if requested
        if self.return_norm:
            out["neighbors_distance"] = actual_dists.reshape(-1)
            
        return out
