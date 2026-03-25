import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as fnn
import torch
import torch.nn.functional as F
import numpy as np
import pytest

from scirex.operators.layers.integral_transform import IntegralTransform as JAX_IntegralTransform
from scirex.operators.layers.neighbor_search import NeighborSearch as JAX_NeighborSearch

try:
    from neuralop.layers.integral_transform import IntegralTransform as PT_IntegralTransform
except ImportError:
    pass

def map_jax_to_pt_integral(jax_params, pt_model):
    new_state_dict = {}
    
    # MLP weights
    jax_mlp_params = jax_params.get('channel_mlp', {})
    
    # Check if projection exists
    if 'projection' in jax_params:
        jax_proj = jax_params['projection']['kernel']
        new_state_dict['projection.weight'] = torch.from_numpy(np.array(jax_proj.T)).to(torch.float64)
        
    for i in range(len(pt_model.channel_mlp.fcs)):
        jax_dense = jax_mlp_params[f'dense_{i}']
        new_state_dict[f'channel_mlp.fcs.{i}.weight'] = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
        new_state_dict[f'channel_mlp.fcs.{i}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
        
    pt_model.load_state_dict(new_state_dict, strict=False)

@pytest.mark.parametrize("transform_type", ["linear", "nonlinear"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_integral_transform_parity(transform_type, reduction, batch_size):
    torch.set_default_dtype(torch.float64)
    
    in_channels = 8
    out_channels = 8 
    coord_dim = 3
    radius = 0.5
    n_in = 50
    n_out = 30
    mlp_layers = [16, 16]

    rng = jax.random.PRNGKey(42)
    rng_y, rng_x, rng_f, rng_init = jax.random.split(rng, 4)
    
    y_jax = jax.random.uniform(rng_y, (n_in, coord_dim), dtype=jnp.float64)
    x_jax = jax.random.uniform(rng_x, (n_out, coord_dim), dtype=jnp.float64)
    
    if batch_size > 1:
        f_y_jax = jax.random.normal(rng_f, (batch_size, n_in, in_channels), dtype=jnp.float64)
    else:
        f_y_jax = jax.random.normal(rng_f, (n_in, in_channels), dtype=jnp.float64)

    kernel_in_dim = 2 * coord_dim
    if transform_type in ("nonlinear", "nonlinear_kernelonly"):
        kernel_in_dim += in_channels
        
    mlp_layers = [kernel_in_dim, 16, out_channels]
    
    from scirex.operators.layers.channel_mlp import LinearChannelMLP
    mlp = LinearChannelMLP(
        layers=mlp_layers,
        activation=fnn.gelu
    )
    
    jax_model = JAX_IntegralTransform(
        in_channels=in_channels,
        out_channels=out_channels,
        transform_type=transform_type,
        reduction=reduction,
        channel_mlp=mlp,
    )
    
    ns = JAX_NeighborSearch(max_neighbors=10)
    jax_neighbors = ns(points=y_jax, queries=x_jax, radius=radius)
    
    jax_params = jax_model.init(rng_init, y_jax, neighbors=jax_neighbors, x=x_jax, f_y=f_y_jax)
    jax_out = jax_model.apply(jax_params, y_jax, neighbors=jax_neighbors, x=x_jax, f_y=f_y_jax)
    
    pt_model = PT_IntegralTransform(
        transform_type=transform_type,
        reduction=reduction,
        channel_mlp_layers=mlp_layers,
        channel_mlp_non_linearity=F.gelu,
        use_torch_scatter=False
    ).to(torch.float64)
    
    map_jax_to_pt_integral(jax_params['params'], pt_model)
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
        pt_out = pt_model(
            y=y_pt, 
            x=x_pt, 
            neighbors=pt_neighbors, 
            f_y=f_y_pt
        )
    
    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-2, atol=1e-2)
    
    jax_out_np = np.array(jax_out)
    pt_out_np = pt_out.numpy()
    
    # Helper for printing regardless of batch dim
    print_jax = jax_out_np[0, :2, :] if jax_out_np.ndim == 3 else jax_out_np[:2, :]
    print_pt = pt_out_np[0, :2, :] if pt_out_np.ndim == 3 else pt_out_np[:2, :]
    
    print(f"JAX out (first 2 nodes):\n{print_jax}")
    print(f"PT out (first 2 nodes):\n{print_pt}")
    print(f"Parity test PASSED for IntegralTransform! (transform_type={transform_type}, reduction={reduction}, batch_size={batch_size})\n")

if __name__ == "__main__":
    print("Running Integral Transform Parity Test...")
    test_integral_transform_parity(transform_type="linear", reduction="sum", batch_size=1)
    test_integral_transform_parity(transform_type="nonlinear", reduction="sum", batch_size=2)
