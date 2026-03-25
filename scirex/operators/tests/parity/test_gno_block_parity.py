import os
import sys

# Resolve ModuleNotFoundError when running as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np
import pytest
from flax import linen as fnn

try:
    from neuralop.layers.gno_block import GNOBlock as PT_GNOBlock
except ImportError:
    print(f"FAILED to import neuralop from {NEURALOP_PATH}")
    raise

from scirex.operators.layers.gno_block import GNOBlock as JAX_GNOBlock
from scirex.operators.layers.neighbor_search import NeighborSearch as JAX_NeighborSearch

# ==============================================================================
# WEIGHT MAPPING
# ==============================================================================

def map_jax_to_pt(jax_params, pt_model):
    """Maps JAX (Flax) parameters to PyTorch model state dict."""
    new_state_dict = {}
    
    # MLP weights
    jax_it_params = jax_params['integral_transform']
    jax_mlp_params = jax_it_params['channel_mlp']
    
    # Official neuralop hierarchy: integral_transform.channel_mlp.fcs[i]
    for i in range(len(pt_model.integral_transform.channel_mlp.fcs)):
        jax_dense = jax_mlp_params[f'dense_{i}']
        new_state_dict[f'integral_transform.channel_mlp.fcs.{i}.weight'] = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
        new_state_dict[f'integral_transform.channel_mlp.fcs.{i}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
        
    pt_model.load_state_dict(new_state_dict, strict=False)

def check_params_dtype(params):
    def check(p):
        if hasattr(p, 'dtype'):
            if p.dtype != jnp.float64:
                print(f"WARNING: Parameter {p} has dtype {p.dtype}, expected float64")
        elif isinstance(p, dict):
            for k, v in p.items():
                check(v)
    check(params)

# ==============================================================================
# TEST CASE
# ==============================================================================

@pytest.mark.parametrize("transform_type", ["linear", "nonlinear", "nonlinear_kernelonly"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gno_block_parity(transform_type, reduction, batch_size):
    # Use float64 for parity check
    torch.set_default_dtype(torch.float64)
    
    # Constants - Note: in_channels must equal out_channels for official neuralop linear transforms
    in_channels = 8
    out_channels = 8 
    coord_dim = 3
    radius = 0.5
    n_in = 50
    n_out = 30
    mlp_layers = [16, 16]

    # Generate random data
    rng = jax.random.PRNGKey(42)
    rng_y, rng_x, rng_f, rng_init = jax.random.split(rng, 4)
    
    y_jax = jax.random.uniform(rng_y, (n_in, coord_dim), dtype=jnp.float64)
    x_jax = jax.random.uniform(rng_x, (n_out, coord_dim), dtype=jnp.float64)
    
    if transform_type == "linear_kernelonly":
        f_y_jax = None
    else:
        if batch_size > 1:
            f_y_jax = jax.random.normal(rng_f, (batch_size, n_in, in_channels), dtype=jnp.float64)
        else:
            f_y_jax = jax.random.normal(rng_f, (n_in, in_channels), dtype=jnp.float64)

    # JAX Model
    jax_model = JAX_GNOBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=coord_dim,
        radius=radius,
        transform_type=transform_type,
        reduction=reduction,
        channel_mlp_layers=mlp_layers,
        pos_embedding_type=None,
        channel_mlp_non_linearity=fnn.gelu # Use GELU
    )
    
    # Initialize JAX
    jax_params = jax_model.init(rng_init, y_jax, x_jax, f_y=f_y_jax)
    # check_params_dtype(jax_params)
    
    # Get neighbors from JAX implementation
    ns = JAX_NeighborSearch(max_neighbors=10)
    jax_neighbors = ns(points=y_jax, queries=x_jax, radius=radius)
    
    # JAX Forward
    jax_out = jax_model.apply(jax_params, y_jax, x_jax, f_y=f_y_jax, neighbors=jax_neighbors)
    
    # PyTorch Model (Official neuralop)
    pt_model = PT_GNOBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=coord_dim,
        radius=radius,
        transform_type=transform_type,
        reduction=reduction,
        channel_mlp_layers=mlp_layers,
        channel_mlp_non_linearity=F.gelu, # Use GELU
        pos_embedding_type=None, # Match JAX
        use_torch_scatter_reduce=False, # Avoid dependency
        use_open3d_neighbor_search=False # Avoid dependency
    ).to(torch.float64)
    
    # Map weights
    map_jax_to_pt(jax_params['params'], pt_model)
    pt_model.eval()
    
    # Prepare PyTorch inputs
    y_pt = torch.from_numpy(np.array(y_jax)).to(torch.float64)
    x_pt = torch.from_numpy(np.array(x_jax)).to(torch.float64)
    f_y_pt = torch.from_numpy(np.array(f_y_jax)).to(torch.float64) if f_y_jax is not None else None
    
    # Convert JAX neighbors (M, K) to CSR format for official neuralop
    jax_idx = np.array(jax_neighbors["neighbor_indices"])
    jax_mask = np.array(jax_neighbors["mask"])
    
    # Filter valid indices
    neighbors_index = jax_idx[jax_mask]
    
    # Compute row splits
    num_neighbors_per_query = jax_mask.sum(axis=1)
    neighbors_row_splits = np.concatenate([[0], np.cumsum(num_neighbors_per_query)])
    
    pt_neighbors = {
        "neighbors_index": torch.from_numpy(neighbors_index).long(),
        "neighbors_row_splits": torch.from_numpy(neighbors_row_splits).long(),
    }
    
    # PyTorch Forward
    with torch.no_grad():
        pt_out = pt_model.integral_transform(
            y=y_pt, 
            x=x_pt, 
            neighbors=pt_neighbors, 
            f_y=f_y_pt
        )
    
    # Compare results
    jax_out_np = np.array(jax_out)
    pt_out_np = pt_out.numpy()
    
    np.testing.assert_allclose(jax_out_np, pt_out_np, rtol=1e-3, atol=1e-3)
    print(f"Parity test PASSED for {transform_type} {reduction} {batch_size}!")
    

def test_gno_simple_minimal():
    """Ultra-minimal test case to isolate the source of mismatch."""
    torch.set_default_dtype(torch.float64)
    in_channels, out_channels, coord_dim = 2, 2, 1
    radius = 10.0
    n_in, n_out = 1, 1
    mlp_layers = [16] # Use non-empty list

    y_jax = jnp.array([[0.0]], dtype=jnp.float64)
    x_jax = jnp.array([[0.1]], dtype=jnp.float64)
    f_y_jax = jnp.array([[1.0, 2.0]], dtype=jnp.float64)

    jax_model = JAX_GNOBlock(
        in_channels=in_channels, out_channels=out_channels, coord_dim=coord_dim,
        radius=radius, transform_type="linear", reduction="sum",
        channel_mlp_layers=mlp_layers, pos_embedding_type=None,
        channel_mlp_non_linearity=fnn.relu # Use ReLU from flax
    )
    
    rng = jax.random.PRNGKey(42)
    jax_params = jax_model.init(rng, y_jax, x_jax, f_y=f_y_jax)
    check_params_dtype(jax_params)
    
    ns = JAX_NeighborSearch(max_neighbors=1)
    jax_neighbors = ns(points=y_jax, queries=x_jax, radius=radius)
    jax_out = jax_model.apply(jax_params, y_jax, x_jax, f_y=f_y_jax, neighbors=jax_neighbors)

    pt_model = PT_GNOBlock(
        in_channels=in_channels, out_channels=out_channels, coord_dim=coord_dim,
        radius=radius, transform_type="linear", reduction="sum",
        channel_mlp_layers=mlp_layers, pos_embedding_type=None,
        channel_mlp_non_linearity=F.relu, # Use ReLU from torch functional
        use_torch_scatter_reduce=False, use_open3d_neighbor_search=False
    ).to(torch.float64)
    
    map_jax_to_pt(jax_params['params'], pt_model)
    pt_model.eval()

    y_pt = torch.from_numpy(np.array(y_jax)).to(torch.float64)
    x_pt = torch.from_numpy(np.array(x_jax)).to(torch.float64)
    f_y_pt = torch.from_numpy(np.array(f_y_jax)).to(torch.float64)

    jax_idx = np.array(jax_neighbors["neighbor_indices"])
    jax_mask = np.array(jax_neighbors["mask"])
    neighbors_index = jax_idx[jax_mask]
    num_neighbors_per_query = jax_mask.sum(axis=1)
    neighbors_row_splits = np.concatenate([[0], np.cumsum(num_neighbors_per_query)])
    pt_neighbors = {
        "neighbors_index": torch.from_numpy(neighbors_index).long(),
        "neighbors_row_splits": torch.from_numpy(neighbors_row_splits).long(),
    }

    with torch.no_grad():
        pt_out = pt_model.integral_transform(y=y_pt, x=x_pt, neighbors=pt_neighbors, f_y=f_y_pt)

    print(f"JAX out: {jax_out}")
    print(f"PT out: {pt_out}")
    
    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-12, atol=1e-12)
    print("Minimal test PASSED!")

if __name__ == "__main__":
    test_gno_simple_minimal()
