import os
# Prevent JAX from pre-allocating all GPU memory, allowing PyTorch to use some.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import torch
import jax
import jax.numpy as jnp
import numpy as np

from neuralop.models import GINO as PtGINO
from scirex.operators.models.gino import GINO as JaxGINO

def generate_dummy_inputs(seed=42, batch_size=1, n_in=16, n_out=16, d=3, nx=8, ny=8, nz=8, in_channels=3, out_channels=3):
    np.random.seed(seed)
    
    # Input geometry and features
    input_geom = np.random.randn(batch_size, n_in, d).astype(np.float32)
    x = np.random.randn(batch_size, n_in, in_channels).astype(np.float32)
    
    # Output queries
    output_queries = np.random.randn(batch_size, n_out, d).astype(np.float32)
    
    # Latent queries (regular grid)
    # usually grid over [0,1]^3
    grid_x = np.linspace(0, 1, nx)
    grid_y = np.linspace(0, 1, ny)
    grid_z = np.linspace(0, 1, nz)
    mesh_x, mesh_y, mesh_z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    latent_geom = np.stack([mesh_x, mesh_y, mesh_z], axis=-1).astype(np.float32)
    latent_queries = np.expand_dims(latent_geom, axis=0).repeat(batch_size, axis=0)
    
    return {
        "input_geom": input_geom,
        "x": x,
        "output_queries": output_queries,
        "latent_queries": latent_queries,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "d": d
    }

def main():
    print("Setting up parity testing environment...")
    
    inputs = generate_dummy_inputs()
    
    # PyTorch setup
    print("\n[PyTorch] Running PtGINO...")
    pt_model = PtGINO(
        in_channels=inputs["in_channels"],
        out_channels=inputs["out_channels"],
        gno_coord_dim=inputs["d"],
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        in_gno_channel_mlp_hidden_layers=[16, 16],
        out_gno_channel_mlp_hidden_layers=[16, 16],
        gno_use_torch_scatter=False,
        gno_use_open3d=False
    )
    
    # Prepare PyTorch Tensors
    pt_input_geom = torch.tensor(inputs["input_geom"])
    pt_x = torch.tensor(inputs["x"])
    pt_latent_queries = torch.tensor(inputs["latent_queries"])
    pt_output_queries = torch.tensor(inputs["output_queries"])
    
    pt_out = pt_model(
        input_geom=pt_input_geom,
        latent_queries=pt_latent_queries,
        output_queries=pt_output_queries,
        x=pt_x
    )
    
    print(f"PyTorch Output Shape: {pt_out.shape}")
    print(f"PyTorch Output Mean: {pt_out.mean().item():.6f}, Std: {pt_out.std().item():.6f}")
    
    # JAX setup
    print("\n[JAX] Running JaxGINO...")
    jax_model = JaxGINO(
        in_channels=inputs["in_channels"],
        out_channels=inputs["out_channels"],
        gno_coord_dim=inputs["d"],
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        in_gno_channel_mlp_hidden_layers=(16, 16),
        out_gno_channel_mlp_hidden_layers=(16, 16)
    )
    
    # Prepare JAX arrays
    jax_x = jnp.array(inputs["x"])
    jax_input_geom = jnp.array(inputs["input_geom"])
    jax_latent_queries = jnp.array(inputs["latent_queries"])
    jax_output_queries = jnp.array(inputs["output_queries"])
    
    rng = jax.random.PRNGKey(42)
    # init
    params = jax_model.init(
        rng,
        input_geom=jax_input_geom,
        latent_queries=jax_latent_queries,
        output_queries=jax_output_queries,
        x=jax_x
    )
    
    jax_out = jax_model.apply(
        params,
        input_geom=jax_input_geom,
        latent_queries=jax_latent_queries,
        output_queries=jax_output_queries,
        x=jax_x
    )
    
    print(f"JAX Output Shape: {jax_out.shape}")
    print(f"JAX Output Mean: {jax_out.mean():.6f}, Std: {jax_out.std():.6f}")

if __name__ == "__main__":
    main()
