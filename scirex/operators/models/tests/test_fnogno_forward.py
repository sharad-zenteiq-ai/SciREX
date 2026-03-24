import pytest
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.models.gino import GINO

def test_fnogno_forward_pass():
    """
    Test the forward pass of the JAX GINO (FNOGNO) model using dummy inputs.
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
    
    # Convert to JAX arrays
    j_input_geom = jnp.array(input_geom)
    j_x = jnp.array(x)
    j_output_queries = jnp.array(output_queries)
    j_latent_queries = jnp.array(latent_queries)

    # 2. Placeholder Model Configuration
    model = GINO(
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

    # 3. Model init and call
    prng_key = jax.random.PRNGKey(seed)
    params = model.init(
        prng_key,
        input_geom=j_input_geom,
        latent_queries=j_latent_queries,
        output_queries=j_output_queries,
        x=j_x
    )
    
    out = model.apply(
        params,
        input_geom=j_input_geom,
        latent_queries=j_latent_queries,
        output_queries=j_output_queries,
        x=j_x
    )

    # 4. Assertion
    assert out is not None, "Model output should not be None."
    assert out.shape == (batch_size, n_out, out_channels), f"Expected shape {(batch_size, n_out, out_channels)}, got {out.shape}"
    assert not jnp.isnan(out).any(), "Model output contains NaNs."
