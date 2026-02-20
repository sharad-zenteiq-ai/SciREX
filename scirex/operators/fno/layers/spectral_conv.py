from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

class SpectralConv2D(nn.Module):
    """
    SpectralConv2D (Flax): complex low-frequency Fourier weights applied in rfft2 domain.
    
    Notes:
    - Stores real/imag as separate params to keep serialization straightforward.
    - Works with inputs shaped (batch, nx, ny, channels).
    - Uses jnp.fft.rfft2 / irfft2 so it is real-valued in/out.
    """
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int
    param_scale: float = 0.02

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_ch)
        returns: (batch, nx, ny, out_ch)
        """
        batch, nx, ny, in_ch = x.shape
        assert in_ch == self.in_channels, "Input channels mismatch"

        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2)

        # Robust scaling for initialization (matching earlier successful runs)
        param_scale = 0.02
        def weight_init(rng, shape):
            return jax.random.normal(rng, shape) * param_scale

        # Two corners for 2D
        w1_r = self.param("w1_r", weight_init, (in_ch, self.out_channels, mx, my))
        w1_i = self.param("w1_i", weight_init, (in_ch, self.out_channels, mx, my))
        w2_r = self.param("w2_r", weight_init, (in_ch, self.out_channels, mx, my))
        w2_i = self.param("w2_i", weight_init, (in_ch, self.out_channels, mx, my))

        w1 = w1_r + 1j * w1_i
        w2 = w2_r + 1j * w2_i

        # FFT
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))

        def compl_mul2d(input, weights):
            w = jnp.transpose(weights, (2, 3, 0, 1))
            return jnp.einsum("b m n i, m n i o -> b m n o", input, w)

        out_ft = jnp.zeros((batch, nx, ny // 2 + 1, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :mx, :my, :].set(compl_mul2d(x_ft[:, :mx, :my, :], w1))
        out_ft = out_ft.at[:, -mx:, :my, :].set(compl_mul2d(x_ft[:, -mx:, :my, :], w2))

        y = jnp.fft.irfft2(out_ft, s=(nx, ny), axes=(1, 2))
        return y

class SpectralConv3D(nn.Module):
    """
    SpectralConv3D (Flax): complex low-frequency Fourier weights applied in rfft3 domain.
    """
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int
    modes_z: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, nz, in_ch)
        returns: (batch, nx, ny, nz, out_ch)
        """
        batch, nx, ny, nz, in_ch = x.shape
        assert in_ch == self.in_channels, "Input channels mismatch"

        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2)
        mz = min(self.modes_z, nz // 2 + 1)

        param_scale = 0.02
        def weight_init(rng, shape):
            return jax.random.normal(rng, shape) * param_scale

        # 4 corners for 3D
        w1 = self.param("w1_r", weight_init, (in_ch, self.out_channels, mx, my, mz)) + \
             1j * self.param("w1_i", weight_init, (in_ch, self.out_channels, mx, my, mz))
        w2 = self.param("w2_r", weight_init, (in_ch, self.out_channels, mx, my, mz)) + \
             1j * self.param("w2_i", weight_init, (in_ch, self.out_channels, mx, my, mz))
        w3 = self.param("w3_r", weight_init, (in_ch, self.out_channels, mx, my, mz)) + \
             1j * self.param("w3_i", weight_init, (in_ch, self.out_channels, mx, my, mz))
        w4 = self.param("w4_r", weight_init, (in_ch, self.out_channels, mx, my, mz)) + \
             1j * self.param("w4_i", weight_init, (in_ch, self.out_channels, mx, my, mz))

        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3))

        def compl_mul3d(input, weights):
            w = jnp.transpose(weights, (2, 3, 4, 0, 1))
            return jnp.einsum("b m n p i, m n p i o -> b m n p o", input, w)

        out_ft = jnp.zeros((batch, nx, ny, nz // 2 + 1, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :mx, :my, :mz, :].set(compl_mul3d(x_ft[:, :mx, :my, :mz, :], w1))
        out_ft = out_ft.at[:, -mx:, :my, :mz, :].set(compl_mul3d(x_ft[:, -mx:, :my, :mz, :], w2))
        out_ft = out_ft.at[:, :mx, -my:, :mz, :].set(compl_mul3d(x_ft[:, :mx, -my:, :mz, :], w3))
        out_ft = out_ft.at[:, -mx:, -my:, :mz, :].set(compl_mul3d(x_ft[:, -mx:, -my:, :mz, :], w4))

        y = jnp.fft.irfftn(out_ft, s=(nx, ny, nz), axes=(1, 2, 3))
        return y
