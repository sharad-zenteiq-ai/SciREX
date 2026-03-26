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
import itertools

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.layers.spectral_conv import SpectralConv as JaxSpectralConv

try:
    from neuralop.layers.spectral_convolution import SpectralConv as PtSpectralConv
except ImportError:
    pass

def map_jax_to_pt_spectral_conv(jax_params, pt_model, n_modes):
    """
    Maps JAX (Flax) SpectralConv parameters to PyTorch SpectralConv full weight tensor.
    Updated for the new SciREX SpectralConv with separated real/imag and adjusted eff_modes.
    """
    m1, m2, m3 = n_modes
    h1, h2 = m1 // 2, m2 // 2
    
    # PT weight shape: (In, Out, m1, m2, m3)
    # Signs for 3D corners: (1, 1), (1, -1), (-1, 1), (-1, -1)
    # Mapping for 3D RFFT in neuralop:
    # Corner 1 (1, 1) -> PT [m1//2:, m2//2:, :]
    # Corner 2 (1, -1) -> PT [m1//2:, :m2//2, :]
    # Corner 3 (-1, 1) -> PT [:m1//2, m2//2:, :]
    # Corner 4 (-1, -1) -> PT [:m1//2, :m2//2, :]
    
    # Prepare PT weight shape (last dim is halved)
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
        real = np.array(jax_params[f"weights_real_{idx}"])
        imag = np.array(jax_params[f"weights_imag_{idx}"])
        jax_complex = real + 1j * imag
        
        # JAX weight order in SciREX: (In, Out, modes_x, modes_y, modes_z)
        # PT weight tensor has full n_modes in last dimension, but only uses a slice.
        # We fill the beginning of the PT weight's last dimension.
        target_slice = (slice(None), slice(None)) + pt_slice + (slice(0, jax_complex.shape[-1]),)
        pt_w[target_slice] = torch.from_numpy(jax_complex).to(torch.complex64)
        
    new_state_dict = {"weight.tensor": pt_w}
    if "bias" in jax_params:
        # Bias in JAX is (1, 1, 1, Out). In PT it is (Out, 1, 1, 1)
        jax_bias = np.array(jax_params["bias"]).squeeze()
        new_state_dict["bias"] = torch.from_numpy(jax_bias).reshape(-1, 1, 1, 1).to(torch.float32)
         
    pt_model.load_state_dict(new_state_dict, strict=False)

def test_spectral_conv_3d_parity():
    # jax.config.update("jax_enable_x64", True) # Disable for this test
    torch.set_default_dtype(torch.float32)
    
    in_channels = 4
    out_channels = 8
    n_modes = (4, 4, 4)
    spatial_shape = (8, 8, 8)
    
    # 1. JAX Model
    jax_model = JaxSpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        bias=True
    )
    
    # 2. PyTorch Model
    pt_model = PtSpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        bias=True,
        fft_norm="forward" # New SciREX default
    )
    
    # 3. Inputs
    x_np = np.random.randn(2, 8, 8, 8, in_channels).astype(np.float32)
    x_jax = jnp.array(x_np)
    # neuralop expects (B, C, H, W, D)
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3))).to(torch.float32)
    
    # 4. Init and Map
    rng = jax.random.PRNGKey(0)
    jax_params = jax_model.init(rng, x_jax)
    # Map weights
    map_jax_to_pt_spectral_conv(jax_params["params"], pt_model, n_modes)
    pt_model.eval()
    
    # DEBUG: Check PT types/shapes
    print(f"PT weight dtype: {pt_model.weight.tensor.dtype}")
    print(f"PT weight shape: {pt_model.weight.tensor.shape}")
    
    # 5. Forward
    jax_out = jax_model.apply(jax_params, x_jax)
    with torch.no_grad():
        # rfft result to check its dtype
        x_pt_fft = torch.fft.rfftn(x_pt, dim=(2,3,4), norm="forward")
        print(f"PT Input FFT dtype: {x_pt_fft.dtype}")
        print(f"PT Input FFT shape: {x_pt_fft.shape}")
        
        pt_out = pt_model(x_pt)
        
    # 6. Compare (JAX uses channels_last, PT uses channels_first)
    pt_out_np = pt_out.numpy().transpose(0, 2, 3, 4, 1)
    
    print(f"JAX out shape: {jax_out.shape}")
    print(f"PT out shape:  {pt_out_np.shape}")
    
    np.testing.assert_allclose(np.array(jax_out), pt_out_np, rtol=1e-10, atol=1e-10)
    print("SpectralConv 3D Parity PASSED!")

if __name__ == "__main__":
    test_spectral_conv_3d_parity()
