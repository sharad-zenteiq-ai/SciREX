# 3D Poisson Equation Reproduction with FNO3D

This project demonstrates the reproduction of a 3D Poisson solver using Fourier Neural Operators (FNO). It covers the full lifecycle from synthetic data generation to model training and high-fidelity volumetric visualization.

## Governing Equation
The problem solves the Poisson equation in a 3D unit cube:

$$-Δu = f \quad \text{in } \Omega = [0, 1]^3$$

Where:
- **$u(x,y,z)$**: The solution field (Potential).
- **$f(x,y,z)$**: The source term field (Forcing function).
- **$Δ$**: The Laplace operator, defined as $\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$.

## Boundary Conditions
The problem assumes **Zero Dirichlet Boundary Conditions**:
$$u(x,y,z) = 0 \quad \text{on } \partial\Omega$$
This means the solution is constrained to be zero on all six faces of the unit cube.

## Dataset Specifications
- **Grid Resolution**: $32 \times 32 \times 32$ (Total of 32,768 nodes per sample).
- **Domain**: Unit Cube $[0, 1]^3$.
- **Input Features**: 4 Channels
    1. Source Term $f(x,y,z)$
    2. Mesh Coordinate $X$
    3. Mesh Coordinate $Y$
    4. Mesh Coordinate $Z$
- **Output Features**: 1 Channel (Solution Field $u(x,y,z)$).
- **Data Generation**:
    - **Source $f$**: Generated as the sum of 3 random 3D Gaussian blobs with randomized scales, positions, and amplitudes.
    - **Ground Truth $u$**: Computed using a high-order spectral solver to ensure the PDE is satisfied with high precision.
- **Sample Count**:
    - Training: 100 samples
    - Testing: 20 samples

## Model Architecture (FNO3D)
- **Type**: Fourier Neural Operator (3D).
- **Modes**: 8 modes in each dimension ($k_{max} = 8$).
- **Hidden Channels**: 32.
- **Layers**: 4 Fourier layers with Skip Connections and GeLU activation.

## Performance Metrics
- **Objective**: Minimize Mean Squared Error (MSE) between $u$ and $\hat{u}$.
- **Final Evaluation**: Measured using **Relative L2 Error** on the independent test set.
