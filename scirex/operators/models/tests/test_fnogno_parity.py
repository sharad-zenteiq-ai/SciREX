import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from neuralop.models import GINO as PtGINO
from scirex.operators.models.gino import GINO as JaxGINO

def test_fnogno_parity():
    """
    Test parity between PyTorch neuraloperator GINO and JAX GINO (FNOGNO).
    """
    # 1. Dummy Input
    seed = 42
    batch_size = 1
    n_in = 16
    n_out = 16
    d = 3
    in_channels = 3
    out_channels = 3
    
    rng = np.random.default_rng(seed)
    input_geom = rng.normal(size=(batch_size, n_in, d)).astype(np.float32)
    x = rng.normal(size=(batch_size, n_in, in_channels)).astype(np.float32)
    output_queries = rng.normal(size=(batch_size, n_out, d)).astype(np.float32)
    
    nx, ny, nz = 8, 8, 8
    grid_x = np.linspace(0, 1, nx)
    grid_y = np.linspace(0, 1, ny)
    grid_z = np.linspace(0, 1, nz)
    mesh_x, mesh_y, mesh_z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    latent_geom = np.stack([mesh_x, mesh_y, mesh_z], axis=-1).astype(np.float32)
    latent_queries = np.expand_dims(latent_geom, axis=0).repeat(batch_size, axis=0)

    # Convert to Tensors and Arrays
    # PyTorch
    pt_input_geom = torch.tensor(input_geom)
    pt_x = torch.tensor(x)
    pt_latent_queries = torch.tensor(latent_queries)
    pt_output_queries = torch.tensor(output_queries)

    # JAX
    j_input_geom = jnp.array(input_geom)
    j_x = jnp.array(x)
    j_output_queries = jnp.array(output_queries)
    j_latent_queries = jnp.array(latent_queries)

    # 2. PyTorch Placeholder Call
    pt_model = PtGINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=d,
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        in_gno_channel_mlp_hidden_layers=[16, 16],
        out_gno_channel_mlp_hidden_layers=[16, 16],
        gno_use_torch_scatter=False,
        gno_use_open3d=False
    )
    
    pt_out = pt_model(
        input_geom=pt_input_geom,
        latent_queries=pt_latent_queries,
        output_queries=pt_output_queries,
        x=pt_x
    )

    # 3. JAX Placeholder Call
    jax_model = JaxGINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=d,
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        in_gno_channel_mlp_hidden_layers=(16, 16),
        out_gno_channel_mlp_hidden_layers=(16, 16)
    )

    prng_key = jax.random.PRNGKey(seed)
    params = jax_model.init(
        prng_key,
        input_geom=j_input_geom,
        latent_queries=j_latent_queries,
        output_queries=j_output_queries,
        x=j_x
    )
    
    jax_out = jax_model.apply(
        params,
        input_geom=j_input_geom,
        latent_queries=j_latent_queries,
        output_queries=j_output_queries,
        x=j_x
    )

    # 4. Placeholder Assertions
    # Note: Weight transfer is not implemented yet, so we cannot assert full numerical parity.
    # We will just verify shapes match.
    assert pt_out.shape == jax_out.shape, f"Shape mismatch: {pt_out.shape} vs {jax_out.shape}"
    assert not torch.isnan(pt_out).any(), "PyTorch output contains NaNs."
    assert not jnp.isnan(jax_out).any(), "JAX output contains NaNs."
