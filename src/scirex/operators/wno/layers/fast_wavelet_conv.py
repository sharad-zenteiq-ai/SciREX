from typing import Sequence, Tuple, Optional, Any, Callable
import jax 
import jax.numpy as jnp
from flax import linen as nn
from jax import random

class LiftingWaveletConv2D(nn.Module):
    """
    Fast Wavelet Convolution using the Lifting Scheme (Second Generation Wavelets).
    
    The Lifting Scheme factorizes the wavelet transform into:
    1. Split: Divide signal into even and odd polyphase components.
    2. Predict: Predict odd samples from even samples.
    3. Update: Update even samples using the prediction error.
    
    This implementation reduces computational complexity compared to standard convolution-based DWT.
    Currently implements the Haar wavelet lifting scheme (CDF 1/1).
    """
    in_channels: int
    out_channels: int
    level: int = 1
    activation: Callable = nn.gelu

    def setup(self):
        # Use Lecun Normal initialization for proper scaling
        # This fixes the signal attenuation issue
        self.w_approx = self.param(
            'w_approx',
            nn.initializers.lecun_normal(),
            (self.in_channels, self.out_channels, 1, 1)
        )
        
        # Weights for Detail Coefficients
        self.w_detail = self.param(
            'w_detail',
            nn.initializers.lecun_normal(),
            (self.in_channels, self.out_channels, 3)
        )
        
        # Learnable residual shortcut (like standard WNO block)
        self.residual = nn.Dense(self.out_channels)

    def cdf53_fwd_1d(self, x, axis):
        """CDF 5/3 Forward Lifting Step (1D)."""
        # x is (..., N, ...)
        N = x.shape[axis]
        
        # Split (Polyphase)
        # We assume N is even (standard WNO usually pads to power of 2)
        idx_e = jnp.arange(0, N, 2)
        idx_o = jnp.arange(1, N, 2)
        
        even = jnp.take(x, idx_e, axis=axis)
        odd = jnp.take(x, idx_o, axis=axis)
        
        # Predict: d = o - 0.5 * (e + e_next)
        # e_next: shift e left by 1 (periodic roll for simplicity/speed)
        # Ideally symmetric padding, but periodic matches spectral methods often
        e_next = jnp.roll(even, -1, axis=axis)
        d = odd - 0.5 * (even + e_next)
        
        # Update: s = e + 0.25 * (d + d_prev)
        # d_prev: shift d right by 1
        d_prev = jnp.roll(d, 1, axis=axis)
        s = even + 0.25 * (d + d_prev)
        
        return s, d

    def cdf53_inv_1d(self, s, d, axis):
        """CDF 5/3 Inverse Lifting Step (1D)."""
        # Inverse Update: e = s - 0.25 * (d + d_prev)
        d_prev = jnp.roll(d, 1, axis=axis)
        e = s - 0.25 * (d + d_prev)
        
        # Inverse Predict: o = d + 0.5 * (e + e_next)
        e_next = jnp.roll(e, -1, axis=axis)
        o = d + 0.5 * (e + e_next)
        
        # Merge
        # We need to interleave e and o along axis
        shape = list(s.shape)
        shape[axis] *= 2
        
        # Create output array
        out = jnp.zeros(shape, dtype=s.dtype)
        
        idx_e_slice = [slice(None)] * len(shape)
        idx_e_slice[axis] = slice(0, shape[axis], 2)
        out = out.at[tuple(idx_e_slice)].set(e)
        
        idx_o_slice = [slice(None)] * len(shape)
        idx_o_slice[axis] = slice(1, shape[axis], 2)
        out = out.at[tuple(idx_o_slice)].set(o)
        
        return out

    def lifting_step_fwd(self, x):
        """
        Forward generic lifting step (CDF 5/3 aka LeGall 5/3).
        Smooth biorthogonal wavelet suitable for PDEs.
        """
        # x shape: (B, H, W, C)
        
        # Row Lifting (Axis 2: Width)
        # L_row, H_row
        l_row, h_row = self.cdf53_fwd_1d(x, axis=2)
        
        # Col Lifting (Axis 1: Height) on both L and H
        # LL, LH
        ll, lh = self.cdf53_fwd_1d(l_row, axis=1)
        # HL, HH
        hl, hh = self.cdf53_fwd_1d(h_row, axis=1)
        
        return ll, lh, hl, hh

    def lifting_step_inv(self, ll, lh, hl, hh):
        """Inverse generic lifting step (CDF 5/3)."""
        # Inverse Col Lifting
        # LL, LH -> L_row
        l_row = self.cdf53_inv_1d(ll, lh, axis=1)
        
        # HL, HH -> H_row
        h_row = self.cdf53_inv_1d(hl, hh, axis=1)
        
        # Inverse Row Lifting
        # L_row, H_row -> X
        x = self.cdf53_inv_1d(l_row, h_row, axis=2)
        
        return x

    def __call__(self, x):
        # x shape: (B, H, W, C)
        
        # 1. Multi-level Forward Transform
        # We store detail coefficients at each level to reconstruct later
        details = []
        curr = x
        
        for i in range(self.level):
            # Split and Predict/Update
            ll, lh, hl, hh = self.lifting_step_fwd(curr)
            
            # Save details for reconstruction
            details.append((lh, hl, hh))
            
            # Continue with approximation (LL)
            curr = ll
            
        # 2. Apply Learnable Weights at the COARSEST level (Deepest recursion)
        # This matches standard WNO: operate on low-frequency content
        
        # Approx (LL) mixing
        # (B, H_coarse, W_coarse, Cin) @ (Cin, Cout, 1, 1) -> (B, H_coarse, W_coarse, Cout)
        ll_out = jnp.einsum('bhwi,ioxy->bhwo', curr, self.w_approx)
        
        # Detail mixing (Deepest level LH, HL, HH)
        lh_deep, hl_deep, hh_deep = details[-1]
        
        lh_out = jnp.einsum('bhwc,io->bhwo', lh_deep, self.w_detail[..., 0])
        hl_out = jnp.einsum('bhwc,io->bhwo', hl_deep, self.w_detail[..., 1])
        hh_out = jnp.einsum('bhwc,io->bhwo', hh_deep, self.w_detail[..., 2])
        
        # Update the deepest details with processed versions
        # Ideally we should keep other details (finer levels) as zero or passed through?
        # Standard WNO zeroes out finer details or learns them separately.
        # Here we zero them out to force low-rank/smooth approximation (typical for FNO/WNO).
        # To pass them through (skip connection style), we would need to project them to out_channels.
        # Since Cin might != Cout, we MUST project or zero them.
        # We choose to ZERO them for now (simplest, robust).
        
        new_details = []
        for i in range(self.level - 1):
            # Finer levels: Zero out (or could learn weights if we added more params)
            # Shapes match the original details at level i, but channel dim must be Out
            B, H, W, _ = details[i][0].shape
            zeros = jnp.zeros((B, H, W, self.out_channels), dtype=ll_out.dtype)
            new_details.append((zeros, zeros, zeros))
            
        # Add the deepest level (processed)
        new_details.append((lh_out, hl_out, hh_out))
        
        # 3. Inverse Transform (Reconstruct)
        curr = ll_out
        
        for i in reversed(range(self.level)):
            lh, hl, hh = new_details[i]
            curr = self.lifting_step_inv(curr, lh, hl, hh)
            
        out = curr
        
        # Learnable Residual Shortcut (Essential for deep networks)
        res = self.residual(x)
            
        return self.activation(out + res)
