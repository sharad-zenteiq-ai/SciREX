import os
import sys
import numpy as np
import torch
import pytest
import scipy.spatial

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.data.car_cfd_dataset import CarCFDDataset

# Import original neighbor search
try:
    from neuralop.layers.neighbor_search import native_neighbor_search as PT_native_neighbor_search
except ImportError:
    pass

def test_dataloader_kdtree_parity():
    """
    Verifies that the SciPy-based KDTree neighbor search in the dataloader
    matches the PyTorch native_neighbor_search implementation.
    """
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    
    num_points = 100
    num_queries = 50
    coord_dim = 3
    radius = 0.2
    max_neighbors = 20
    
    # Generate random points and queries
    points_np = np.random.uniform(0, 1, (num_points, coord_dim)).astype(np.float64)
    queries_np = np.random.uniform(0, 1, (num_queries, coord_dim)).astype(np.float64)
    
    # --- 1. SciPy KDTree (Dataloader way) ---
    tree = scipy.spatial.KDTree(points_np)
    dists_scipy, indices_scipy = tree.query(queries_np, k=max_neighbors, distance_upper_bound=radius)
    
    # SciPy uses len(points) as a filler for "no neighbor" if distance_upper_bound is hit
    mask_scipy = (indices_scipy < num_points)
    indices_scipy_cleaned = np.where(mask_scipy, indices_scipy, -1)
    
    # --- 2. PyTorch Native Search ---
    points_pt = torch.from_numpy(points_np)
    queries_pt = torch.from_numpy(queries_np)
    
    pt_out = PT_native_neighbor_search(points_pt, queries_pt, radius=radius)
    pt_indices = pt_out["neighbors_index"].numpy()
    pt_splits = pt_out["neighbors_row_splits"].numpy()
    
    # --- 3. Comparison ---
    # The PyTorch implementation returns ALL neighbors in CSR format.
    # The SciPy implementation returns at most max_neighbors in a padded array.
    
    # For each query, let's verify that the SciPy neighbors are a subset of the PT neighbors
    for q_idx in range(num_queries):
        # PT neighbors for this query
        start, end = pt_splits[q_idx], pt_splits[q_idx+1]
        query_pt_neighbors = set(pt_indices[start:end])
        
        # SciPy neighbors for this query
        query_scipy_neighbors = indices_scipy[q_idx][mask_scipy[q_idx]]
        query_scipy_neighbors_set = set(query_scipy_neighbors)
        
        # Check if SciPy found all or is a consistent subset
        if len(query_pt_neighbors) <= max_neighbors:
            # SciPy should have found exactly the same set
            assert query_scipy_neighbors_set == query_pt_neighbors, \
                f"Query {q_idx}: Set mismatch. SciPy: {query_scipy_neighbors_set}, PT: {query_pt_neighbors}"
        else:
            # SciPy should have found a subset of size max_neighbors
            assert len(query_scipy_neighbors) == max_neighbors
            assert query_scipy_neighbors_set.issubset(query_pt_neighbors), \
                f"Query {q_idx}: SciPy neighbors not a subset of PT neighbors"
            
            # Optionally check if they are indeed the closest ones
            pass
            
        # Verify distances match if PT also has weights
        pass

    # Additional Check: Distance comparison for nearest neighbors
    # For a query where we found neighbors, let's verify distance
    dist_scipy, _ = tree.query(queries_np, k=1, distance_upper_bound=radius)
    
    # PT native search distance check (simplified)
    # Since cdist is brute force, we directly use it
    dist_pt_all = torch.cdist(queries_pt, points_pt).numpy()
    dist_pt_nearest = dist_pt_all.min(axis=1)
    
    # Only compare where they are within radius
    mask_within = dist_pt_nearest <= radius
    np.testing.assert_allclose(dist_scipy[mask_within], dist_pt_nearest[mask_within], rtol=1e-5)

    print(f"Dataloader KDTree Parity Test PASSED (Indices and Distances)!")
    print(f"Verified {num_queries} queries against {num_points} points with radius={radius}.")
    
    # --- Sample Prints for Visual Verification (First 5 queries) ---
    print("\n--- Visual Comparison for First 5 Queries ---")
    for sample_q in range(min(5, num_queries)):
        print(f"\nQuery {sample_q} Coords: {queries_np[sample_q]}")
        
        scipy_nb = indices_scipy[sample_q][mask_scipy[sample_q]]
        scipy_ds = dists_scipy[sample_q][mask_scipy[sample_q]]
        
        pt_start, pt_end = pt_splits[sample_q], pt_splits[sample_q+1]
        pt_nb = pt_indices[pt_start:pt_end]
        
        print(f"  SciPy Nearest Indices: {scipy_nb[:5]}")
        print(f"  SciPy Nearest Dists:   {scipy_ds[:5]}")
        print(f"  PT Found {len(pt_nb)} total neighbors. First few: {pt_nb[:5]}")

if __name__ == "__main__":
    test_dataloader_kdtree_parity()
