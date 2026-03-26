import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.losses.data_losses import mse as jax_mse, lp_loss as jax_lp_loss

try:
    from neuralop.losses.data_losses import LpLoss as PtLpLoss
except ImportError:
    pass

def test_losses_parity():
    np.random.seed(42)
    batch_size = 2
    nx, ny, nz = 8, 8, 8
    channels = 4
    
    shape = (batch_size, nx, ny, nz, channels)
    # JAX inputs (channels_last)
    pred_np = np.random.randn(*shape).astype(np.float32)
    target_np = np.random.randn(*shape).astype(np.float32)
    
    pred_jax = jnp.array(pred_np)
    target_jax = jnp.array(target_np)
    
    # PT inputs (channels_first for LpLoss in neuralop usually? Wait, neuralop LpLoss handles arbitrary dims)
    # Actually PtLpLoss expect (B, ..., C). Let's check its source.
    pred_pt = torch.from_numpy(pred_np)
    target_pt = torch.from_numpy(target_np)
    
    # 1. MSE
    mse_jax = jax_mse(pred_jax, target_jax)
    mse_pt = torch.nn.functional.mse_loss(pred_pt, target_pt)
    
    np.testing.assert_allclose(np.array(mse_jax), mse_pt.numpy(), rtol=1e-5)
    print("MSE Loss Parity PASSED!")
    
    # 2. Lp Loss (Relative)
    # neuralop.utils.LpLoss(d=ndims, p=p, reduce_dims=spatial_dims)
    # For relative loss, it does sum(abs(p-t)^p) / sum(abs(t)^p)
    
    # JAX version
    lp_jax = jax_lp_loss(pred_jax, target_jax, p=2)
    
    # PT version
    # neuralop LpLoss always reduces spatial + channels if d is set to cover them.
    # SciREX lp_loss reduces everything except batch, then takes mean.
    # To match, we set d=4 (nx, ny, nz, channels) and reduction='mean'.
    pt_lp_loss_fn = PtLpLoss(d=4, p=2, reduction='mean')
    
    lp_pt = pt_lp_loss_fn(pred_pt, target_pt)
    
    # In JAX LpLoss, it returns the mean. 
    # In PtLpLoss, reduction='mean' does mean(diff / (norm + eps)).
    # So they should match exactly.
    np.testing.assert_allclose(np.array(lp_jax), lp_pt.numpy(), rtol=1e-5)
    print("Lp Loss Parity PASSED!")
    
    # 3. H1 Loss
    from scirex.operators.losses.data_losses import h1_loss as jax_h1_loss
    from neuralop.losses.data_losses import H1Loss as PtH1Loss
    
    # JAX H1 loss expects (B, NX, NY, ...)
    # Let's use 2D for H1, 1 channel to avoid reduction differences
    h1_channels = 1
    in_2d = np.random.randn(batch_size, 8, 8, h1_channels).astype(np.float32)
    targ_2d = np.random.randn(batch_size, 8, 8, h1_channels).astype(np.float32)
    
    h1_jax = jax_h1_loss(jnp.array(in_2d), jnp.array(targ_2d))
    
    # Pt H1 loss expects (B, C, H, W)
    in_2d_pt = torch.from_numpy(in_2d.transpose(0, 3, 1, 2))
    targ_2d_pt = torch.from_numpy(targ_2d.transpose(0, 3, 1, 2))
    
    # Pt H1Loss(d=2)
    pt_h1_loss_fn = PtH1Loss(d=2, reduction='sum')
    h1_pt = pt_h1_loss_fn(in_2d_pt, targ_2d_pt, quadrature=1.0)
    
    h1_jax = jax_h1_loss(jnp.array(in_2d), jnp.array(targ_2d))
    
    # Pt H1 loss expects (B, C, H, W)
    in_2d_pt = torch.from_numpy(in_2d.transpose(0, 3, 1, 2))
    targ_2d_pt = torch.from_numpy(targ_2d.transpose(0, 3, 1, 2))
    
    # Pt H1Loss(d=2)
    # quadrature=1.0 ensures grid scaling is removed for parity check
    pt_h1_loss_fn = PtH1Loss(d=2, reduction='sum')
    h1_pt = pt_h1_loss_fn(in_2d_pt, targ_2d_pt, quadrature=1.0)
    
    # Parity check
    np.testing.assert_allclose(np.array(h1_jax), h1_pt.numpy(), rtol=1e-5)
    print("H1 Loss Parity PASSED!")

if __name__ == "__main__":
    test_losses_parity()
