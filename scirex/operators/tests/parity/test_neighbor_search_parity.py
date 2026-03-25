import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
import numpy as np
import pytest

from scirex.operators.layers.neighbor_search import NeighborSearch as JAX_NeighborSearch

try:
    from neuralop.layers.neighbor_search import native_neighbor_search
except ImportError:
    pass

@pytest.mark.parametrize("num_points", [10, 50])
@pytest.mark.parametrize("num_queries", [10, 20])
@pytest.mark.parametrize("radius", [0.5, 1.5])
def test_neighbor_search_parity(num_points, num_queries, radius):
    torch.set_default_dtype(torch.float64)
    coord_dim = 2
    
    rng = jax.random.PRNGKey(42)
    rng_p, rng_q = jax.random.split(rng)
    
    points_jax = jax.random.uniform(rng_p, (num_points, coord_dim), dtype=jnp.float64)
    queries_jax = jax.random.uniform(rng_q, (num_queries, coord_dim), dtype=jnp.float64)
    
    # JAX forward
    max_neighbors = num_points
    jax_ns = JAX_NeighborSearch(max_neighbors=max_neighbors, return_norm=True)
    jax_out = jax_ns(points=points_jax, queries=queries_jax, radius=radius)
    
    jax_idx = np.array(jax_out["neighbor_indices"])
    jax_mask = np.array(jax_out["mask"])
    
    # Sort the jax neighbors for each query to match index-sorting
    jax_flat_idx = []
    for i in range(num_queries):
        valid = jax_mask[i]
        indices = jax_idx[i][valid]
        indices = np.sort(indices)
        jax_flat_idx.extend(indices)
    jax_flat_idx = np.array(jax_flat_idx)
    
    # PyTorch forward
    points_pt = torch.from_numpy(np.array(points_jax)).to(torch.float64)
    queries_pt = torch.from_numpy(np.array(queries_jax)).to(torch.float64)
    
    pt_out = native_neighbor_search(points_pt, queries_pt, radius=radius, return_norm=True)
    pt_flat_idx = pt_out["neighbors_index"].numpy()
    
    # Compare indices
    assert len(jax_flat_idx) == len(pt_flat_idx), f"Number of neighbors mismatch: {len(jax_flat_idx)} vs {len(pt_flat_idx)}"
    np.testing.assert_array_equal(jax_flat_idx, pt_flat_idx)
    
    # Compare row splits
    num_neighbors_per_query = jax_mask.sum(axis=1)
    jax_row_splits = np.concatenate([[0], np.cumsum(num_neighbors_per_query)])
    pt_row_splits = pt_out["neighbors_row_splits"].numpy()
    
    np.testing.assert_array_equal(jax_row_splits, pt_row_splits)
    print(f"JAX neighbors index: {jax_flat_idx[:10]}... (Total: {len(jax_flat_idx)})")
    print(f"PT neighbors index:  {pt_flat_idx[:10]}... (Total: {len(pt_flat_idx)})")
    print(f"Parity test PASSED for neighbor search! (num_points={num_points}, num_queries={num_queries}, radius={radius})\n")

if __name__ == "__main__":
    print("Running Neighbor Search Parity Test...")
    test_neighbor_search_parity(num_points=50, num_queries=20, radius=0.5)
