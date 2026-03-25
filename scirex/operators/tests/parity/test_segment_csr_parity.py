import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
import numpy as np
import pytest

from scirex.operators.layers.segment_csr import segment_sum as JAX_segment_sum
from scirex.operators.layers.segment_csr import segment_mean as JAX_segment_mean

try:
    from neuralop.layers.segment_csr import segment_csr as PT_segment_csr
except ImportError:
    pass

@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("num_segments", [5, 10])
def test_segment_csr_parity(reduction, num_segments):
    torch.set_default_dtype(torch.float64)
    # Generate mock CRS lengths
    np.random.seed(42)
    lengths = np.random.randint(1, 5, size=num_segments)
    num_edges = int(lengths.sum())
    channels = 4
    
    # Generate edge values
    edge_values_np = np.random.randn(num_edges, channels).astype(np.float64)
    
    # JAX inputs
    edge_values_jax = jnp.array(edge_values_np)
    
    # PyTorch inputs
    edge_values_pt = torch.from_numpy(edge_values_np)
    splits_np = np.concatenate([[0], np.cumsum(lengths)])
    splits_pt = torch.from_numpy(splits_np).long()
    
    if reduction == "sum":
        # PT
        pt_out = PT_segment_csr(edge_values_pt, splits_pt, reduction="sum", use_scatter=False)
        
        # JAX
        segment_ids = np.repeat(np.arange(num_segments), lengths)
        jax_out = JAX_segment_sum(edge_values_jax, jnp.array(segment_ids), num_segments)
    else:
        # PT
        pt_out = PT_segment_csr(edge_values_pt, splits_pt, reduction="mean", use_scatter=False)
        
        # JAX
        segment_ids = np.repeat(np.arange(num_segments), lengths)
        jax_out = JAX_segment_mean(edge_values_jax, jnp.array(segment_ids), num_segments)
        
    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-5, atol=1e-5)
    print(f"JAX out ({reduction}):\n{jax_out}")
    print(f"PT out ({reduction}):\n{pt_out.numpy()}")
    print(f"Parity test PASSED for segment_csr (reduction={reduction}, num_segments={num_segments})!\n")

if __name__ == "__main__":
    print("Running Segment CSR Parity Test...")
    test_segment_csr_parity(reduction="sum", num_segments=5)
    test_segment_csr_parity(reduction="mean", num_segments=5)
