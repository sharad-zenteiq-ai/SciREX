import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.layers.fno_block import FNOBlock as JaxFNOBlock

try:
    from neuralop.layers.fno_block import FNOBlocks as PtFNOBlocks
except ImportError:
    pass

def map_jax_to_pt_fno_block(jax_params, pt_model, n_modes, index=0):
    """
    Maps JAX (Flax) FNOBlock parameters to PyTorch FNOBlocks state dict (for a specific layer index).
    """
    new_state_dict = {}
    
    # helper for ChannelMLP (PyTorch uses Conv1d)
    def map_channel_mlp(jax_p, pt_p_prefix):
        i = 0
        while f"dense_{i}" in jax_p:
            jax_dense = jax_p[f"dense_{i}"]
            new_state_dict[f"{pt_p_prefix}.fcs.{i}.weight"] = torch.from_numpy(np.array(jax_dense["kernel"].T)).unsqueeze(-1).to(torch.float32)
            if "bias" in jax_dense:
                new_state_dict[f"{pt_p_prefix}.fcs.{i}.bias"] = torch.from_numpy(np.array(jax_dense["bias"])).to(torch.float32)
            i += 1

    # SpectralConv
    jax_spec = jax_params["SpectralConv_0"]
    m1, m2, m3 = n_modes
    h1, h2 = m1 // 2, m2 // 2
    
    m_pt = list(n_modes)
    m_pt[-1] = n_modes[-1] // 2 + 1
    pt_w = torch.zeros((pt_model.in_channels, pt_model.out_channels, *m_pt), dtype=torch.complex64)
    
    corners = [
        (1, (slice(h1, None), slice(h2, None))), # (1, 1)
        (2, (slice(h1, None), slice(None, h2))), # (1, -1)
        (3, (slice(None, h1), slice(h2, None))), # (-1, 1)
        (4, (slice(None, h1), slice(None, h2))), # (-1, -1)
    ]
    
    for idx, pt_slice in corners:
        real = np.array(jax_spec[f"weights_real_{idx}"])
        imag = np.array(jax_spec[f"weights_imag_{idx}"])
        pt_w[(slice(None), slice(None)) + pt_slice + (slice(0, real.shape[-1]),)] = \
            torch.from_numpy(real + 1j * imag).to(torch.complex64)
            
    new_state_dict[f"convs.{index}.weight.tensor"] = pt_w
    if "bias" in jax_spec:
        new_state_dict[f"convs.{index}.bias"] = torch.from_numpy(np.array(jax_spec["bias"]).squeeze()).reshape(-1,1,1,1).to(torch.float32)

    # First Skip (FNO skip)
    jax_skip0 = jax_params["SkipConnection_0"]
    if "Dense_0" in jax_skip0:
        new_state_dict[f"fno_skips.{index}.conv.weight"] = torch.from_numpy(np.array(jax_skip0["Dense_0"]["kernel"].T)).unsqueeze(-1).to(torch.float32)
        if "bias" in jax_skip0["Dense_0"]:
            new_state_dict[f"fno_skips.{index}.conv.bias"] = torch.from_numpy(np.array(jax_skip0["Dense_0"]["bias"])).to(torch.float32)

    # ChannelMLP and its skip
    if "ChannelMLP_0" in jax_params:
        map_channel_mlp(jax_params["ChannelMLP_0"], f"channel_mlp.{index}")
        jax_skip1 = jax_params["SkipConnection_1"]
        if "SoftGating_0" in jax_skip1:
             new_state_dict[f"channel_mlp_skips.{index}.weight"] = \
                torch.from_numpy(np.array(jax_skip1["SoftGating_0"]["weight"])).reshape(1,-1,1,1,1).to(torch.float32)
        elif "Dense_0" in jax_skip1:
             new_state_dict[f"channel_mlp_skips.{index}.conv.weight"] = \
                torch.from_numpy(np.array(jax_skip1["Dense_0"]["kernel"].T)).unsqueeze(-1).to(torch.float32)

    pt_model.load_state_dict(new_state_dict, strict=False)

def test_fno_block_parity():
    torch.set_default_dtype(torch.float32)
    
    hidden_channels = 16
    n_modes = (4, 4, 4)
    n_layers = 1
    
    # 1. JAX Model
    jax_block = JaxFNOBlock(
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        skip_type="linear",
        channel_mlp_skip="soft-gating"
    )
    
    # 2. PyTorch Model
    pt_blocks = PtFNOBlocks(
        in_channels=hidden_channels,
        out_channels=hidden_channels,
        n_modes=n_modes,
        n_layers=n_layers,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        norm=None
    )
    
    # 3. Inputs
    x_np = np.random.randn(1, 8, 8, 8, hidden_channels).astype(np.float32)
    x_jax = jnp.array(x_np)
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3))).to(torch.float32)
    
    # 4. Init and Map
    rng = jax.random.PRNGKey(0)
    jax_params = jax_block.init(rng, x_jax, is_last=True)
    
    map_jax_to_pt_fno_block(jax_params["params"], pt_blocks, n_modes, index=0)
    pt_blocks.eval()
    
    # 5. Forward
    # In JAX, we pass is_last=True to match PT behavior for single layer (skipped activation)
    jax_out = jax_block.apply(jax_params, x_jax, is_last=True)
    
    with torch.no_grad():
        pt_out = pt_blocks(x_pt, index=0)
        
    # 6. Compare
    pt_out_np = pt_out.numpy().transpose(0, 2, 3, 4, 1)
    
    np.testing.assert_allclose(np.array(jax_out), pt_out_np, rtol=1e-5, atol=1e-5)
    print("FNOBlock Parity PASSED!")

if __name__ == "__main__":
    test_fno_block_parity()
