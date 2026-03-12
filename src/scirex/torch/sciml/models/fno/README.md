# FNO (Fourier Neural Operator) — 2D modules

This directory contains 2D Fourier Neural Operator implementations used for PDE learning and super-resolution.

## Files

- [`fno2d.py`](fno2d.py) — Implements **`SpectralConv2d`**, the spectral convolution layer used in Fourier layers, and **`FNO2d`**, the full 2D FNO architecture (lifting, Fourier layers, and projection).
- [`bicubic_fno.py`](bicubic_fno.py) — Implements **`BicubicFNO`**, a super-resolution variant that applies bicubic upsampling (no trainable parameters) followed by an `FNO2d` refinement.

## Quick usage

```python
from scirex.torch.sciml.models.fno.bicubic_fno import BicubicFNO

model = BicubicFNO(modes1=12, modes2=12, width=32)
# lr: Tensor of shape (batch, 16, 16)
hr_pred, hr_bicubic = model(lr)
# hr_pred: refined high-resolution output, hr_bicubic: bicubic upsampled baseline
```

## Notes

- `SpectralConv2d` uses truncated Fourier modes to perform efficient spectral convolutions.
- `BicubicFNO` returns a tuple `(x_out, x_bicubic)` where `x_out = x_bicubic + x_refined`.

