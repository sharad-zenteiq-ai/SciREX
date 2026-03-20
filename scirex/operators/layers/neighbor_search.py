import jax
import jax.numpy as jnp

class NeighborSearch:
    """
    JIT-safe fixed-size neighbor search returning (N, K) padded arrays.
    """

    def __init__(self, max_neighbors: int, return_norm: bool = False):
        self.max_neighbors = max_neighbors
        self.return_norm = return_norm

    def __call__(self, points: jnp.ndarray, queries: jnp.ndarray = None, radius: float = jnp.inf):
        """
        Args:
            points: (M, d) data points to search within.
            queries: (N, d) query points. If None, queries = points.
            radius: maximum distance for a neighbor to be considered valid.
        
        Returns:
            Dict containing:
            - neighbor_indices: (N, K) boolean mask where valid neighbors = True
            - mask: (N, K) boolean mask where valid neighbors = True
            - distances: (N, K) (optional) padded with 0
        """
        if queries is None:
            queries = points
            
        # 1. Compute all pairwise squared distances efficiently using: ||q - d||^2 = ||q||^2 + ||d||^2 - 2(q.d)
        q_sq = jnp.sum(queries ** 2, axis=-1, keepdims=True)  # (N, 1)
        d_sq = jnp.sum(points ** 2, axis=-1, keepdims=True)     # (M, 1)
        dot  = jnp.dot(queries, points.T)                     # (N, M)
        
        # Clip to 0 to prevent negative distances from floating point numerical errors
        dists_sq = jnp.maximum(q_sq + d_sq.T - 2.0 * dot, 0.0) 
        dists = jnp.sqrt(dists_sq)

        # Pad dists if M < max_neighbors to prevent top_k failure
        n_queries, n_points = dists.shape
        pad_size = max(0, self.max_neighbors - n_points)
        if pad_size > 0:
            dists = jnp.pad(dists, ((0, 0), (0, pad_size)), constant_values=jnp.inf)
        
        # 2. Find the indices of the K-nearest points
        # Use jax.lax.top_k to get the smallest distances (negate them)
        val, neighbor_indices = jax.lax.top_k(-dists, self.max_neighbors)
        actual_dists = -val # (N, K)
        
        # 3. Create a validity mask: True if distance <= radius and valid index
        valid_idx_mask = neighbor_indices < n_points
        mask = (actual_dists <= radius) & valid_idx_mask
        
        # 4. Filter invalid indices with -1
        neighbor_indices = jnp.where(mask, neighbor_indices, -1)
        
        # 5. Filter distances with 0 for padded ones
        actual_dists = jnp.where(mask, actual_dists, 0.0)

        out = {
            "neighbor_indices": neighbor_indices.astype(jnp.int32),
            "mask": mask,
        }
        
        if self.return_norm:
            out["distances"] = actual_dists
            
        return out
