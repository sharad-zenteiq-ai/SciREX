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

        # params: (in_ch, out_ch, modes_x, modes_y)
        wr = self.param(
            "wr",
            lambda rng, shape: jax.random.normal(rng, shape) * self.param_scale,
            (in_ch, self.out_channels, self.modes_x, self.modes_y),
        )
        wi = self.param(
            "wi",
            lambda rng, shape: jax.random.normal(rng, shape) * self.param_scale,
            (in_ch, self.out_channels, self.modes_x, self.modes_y),
        )
        w_complex = wr + 1j * wi  # complex weights

        # FFT of input along spatial dims
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (batch, nx, ny//2+1, in_ch) complex

        ny_modes = x_ft.shape[2]
        mx = min(self.modes_x, nx)
        my = min(self.modes_y, ny_modes)

        # select low modes
        x_ft_low = x_ft[:, :mx, :my, :]  # (batch, mx, my, in_ch)

        # reorder weights to (mx, my, in_ch, out_ch)
        # w_complex: (in_ch, out_ch, modes_x, modes_y)
        w = w_complex[:, :, :mx, :my]  # (in_ch, out_ch, mx, my)
        w = jnp.transpose(w, (2, 3, 0, 1))  # (mx, my, in_ch, out_ch)

        # einsum to multiply fourier coefficients with complex weights
        # result: (batch, mx, my, out_ch)
        y_ft_low = jnp.einsum("b m n i, m n i o -> b m n o", x_ft_low, w)

        # build output frequency tensor and place low modes
        out_ft = jnp.zeros((batch, nx, ny // 2 + 1, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :mx, :my, :].set(y_ft_low)

        # inverse real FFT to spatial domain
        y = jnp.fft.irfft2(out_ft, s=(nx, ny), axes=(1, 2))
        return jnp.real(y)
