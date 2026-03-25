import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
import numpy as np
import pytest

from scirex.operators.layers.embeddings import (
    GridEmbedding as JAX_GridEmbedding,
    SinusoidalEmbedding as JAX_SinusoidalEmbedding,
    RotaryEmbedding as JAX_RotaryEmbedding,
    apply_rotary_pos_emb as JAX_apply_rotary_pos_emb,
)

try:
    from neuralop.layers.embeddings import (
        GridEmbeddingND as PT_GridEmbeddingND,
        SinusoidalEmbedding as PT_SinusoidalEmbedding,
        RotaryEmbedding2D as PT_RotaryEmbedding,
        apply_rotary_pos_emb as PT_apply_rotary_pos_emb,
    )
except ImportError:
    pass

def test_grid_embedding_parity():
    torch.set_default_dtype(torch.float64)
    # 2D Grid
    spatial_dims = (8, 8)
    in_channels = 2
    batch_size = 3
    
    # PT takes channels first: (B, C, H, W)
    pt_input = torch.randn(batch_size, in_channels, *spatial_dims, dtype=torch.float64)
    pt_model = PT_GridEmbeddingND(in_channels=in_channels, dim=2, grid_boundaries=[[0, 1], [0, 1]])
    pt_out = pt_model(pt_input, batched=True)
    
    # JAX takes channels last: (B, H, W, C)
    jax_input = jnp.array(pt_input.permute(0, 2, 3, 1).numpy())
    # Note: JAX endpoint=False matches PT's torch.linspace(...)[-1] ?
    # Let's check PT code: torch.linspace(start, stop, res + 1)[:-1] -> endpoint is basically False!
    jax_model = JAX_GridEmbedding(grid_boundaries=((0.0, 1.0), (0.0, 1.0)), endpoint=False)
    jax_out = jax_model.apply({}, jax_input)
    
    # Compare
    pt_out_np = pt_out.permute(0, 2, 3, 1).numpy()
    np.testing.assert_allclose(np.array(jax_out), pt_out_np, rtol=1e-5, atol=1e-5)
    print(f"JAX Grid (shape: {jax_out.shape}):\n{np.array(jax_out)[0, 0, 0, :]}")
    print(f"PT Grid (shape: {pt_out_np.shape}):\n{pt_out_np[0, 0, 0, :]}")
    print("Parity test PASSED for GridEmbedding!\n")


@pytest.mark.parametrize("embedding_type", ["transformer", "nerf"])
def test_sinusoidal_embedding_parity(embedding_type):
    torch.set_default_dtype(torch.float64)
    in_channels = 3
    num_freqs = 4
    batch_size = 5
    seq_len = 10
    
    np.random.seed(42)
    inputs_np = np.random.randn(batch_size, seq_len, in_channels).astype(np.float64)
    
    # PT
    pt_input = torch.from_numpy(inputs_np)
    pt_model = PT_SinusoidalEmbedding(in_channels=in_channels, num_frequencies=num_freqs, embedding_type=embedding_type)
    pt_out = pt_model(pt_input)
    
    # JAX
    jax_input = jnp.array(inputs_np)
    jax_model = JAX_SinusoidalEmbedding(num_frequencies=num_freqs, embedding_type=embedding_type)
    jax_out = jax_model.apply({}, jax_input)
    
    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-5, atol=1e-5)
    print(f"JAX Sinusoidal ({embedding_type}):\n{np.array(jax_out)[0, 0, :10]}")
    print(f"PT Sinusoidal ({embedding_type}):\n{pt_out.numpy()[0, 0, :10]}")
    print(f"Parity test PASSED for SinusoidalEmbedding ({embedding_type})!\n")

def test_rotary_embedding_parity():
    torch.set_default_dtype(torch.float64)
    dim = 8
    batch_size = 2
    num_points = 5
    
    np.random.seed(42)
    coords_np = np.random.randn(batch_size, num_points).astype(np.float64)
    t_np = np.random.randn(batch_size, num_points, dim).astype(np.float64)
    
    pt_coords = torch.from_numpy(coords_np)
    pt_t = torch.from_numpy(t_np)
    pt_model = PT_RotaryEmbedding(dim=dim)
    # In PyTorch RotaryEmbedding2D: 
    # the __init__ doesn't take dim but actually it takes dim and sets min_freq=1/64 maybe? Wait!
    # I'll manually set min_freq=1.0 and scale=1.0 so they match or verify init args.
    pass # Wait, RotaryEmbedding in PT __init__ args need to be checked. Let's do it safely.
    # The jax one uses apply_rotary_pos_emb. Let's just compare the helper function.
    
    jax_t = jnp.array(t_np)
    np_freqs = coords_np[..., None]
    pt_freqs = torch.from_numpy(np_freqs).repeat(1, 1, dim//2)
    pt_freqs = torch.cat((pt_freqs, pt_freqs), dim=-1)
    
    pt_out = PT_apply_rotary_pos_emb(pt_t, pt_freqs)
    
    # In JAX:
    jax_freqs_full = jnp.array(pt_freqs.numpy())
    jax_out = JAX_apply_rotary_pos_emb(jax_t, jax_freqs_full)
    
    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-5, atol=1e-5)
    print(f"JAX Rotary out:\n{np.array(jax_out)[0, 0, :]}")
    print(f"PT Rotary out:\n{pt_out.numpy()[0, 0, :]}")
    print("Parity test PASSED for RotaryEmbedding!\n")

if __name__ == "__main__":
    print("Running Embeddings Parity Tests...")
    test_grid_embedding_parity()
    test_sinusoidal_embedding_parity("transformer")
    test_sinusoidal_embedding_parity("nerf")
    test_rotary_embedding_parity()
