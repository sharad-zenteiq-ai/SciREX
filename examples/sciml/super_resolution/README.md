# Bicubic FNO for Fluid Super-Resolution

**A framework for super-resolution of fluid flows using Physics-Informed Neural Operators.**

## Installation ✅
Install the required Python packages:

```bash
pip install torch numpy scipy matplotlib tqdm
```

## Usage
Generate the synthetic Navier–Stokes dataset using the pseudo-spectral solver. The script simulates the flow on a fine grid ($1024 \times 1024$) and downsamples it to create low-resolution (LR) and high-resolution (HR) pairs.

### Physics parameters
- **Viscosity** ($\nu$): $1\times 10^{-3}$
- **Time step** ($\Delta t$): $1\times 10^{-4}$
- **Forcing**: Deterministic sinusoidal forcing

## Model architecture
The **BicubicFNO** model uses a dual-branch strategy:

- **Bicubic branch:** Parameter-free upsampling that captures dominant low-frequency structures.
- **FNO branch:** A Fourier Neural Operator that learns to predict high-frequency residuals ($x_{refined}$).

The final output is the sum:

$$u_{SR} = x_{bicubic} + x_{refined}$$


