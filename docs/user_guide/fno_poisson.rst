FNO Poisson Examples
===================

This user guide walks through the **end-to-end Poisson equation examples**
implemented using **Fourier Neural Operators (FNO)** in SciREX.
The Poisson equation is one of the most fundamental partial differential
equations in scientific computing. It appears in a wide range of physical
systems, including electrostatics, heat conduction, diffusion processes,
and incompressible fluid flow.

In all examples here, our goal is to learn the *operator* that maps a source
term :math:`f` to the corresponding solution :math:`u`:

:math:`-\nabla^2 u(x) = f(x)`

Rather than solving this equation from scratch for every new input, we train
an FNO model to approximate this mapping directly and efficiently.


2D Poisson Example
------------------

The 2D Poisson example is implemented in:

::

   scripts/poisson2d_fno_train_lploss.py

Problem Setup
----------------------------

We solve the equation on the unit square domain $[0, 1] \times [0, 1]$ with periodic boundary conditions.This choice makes the problem well-suited
for Fourier-based methods and allows us to use a spectral solver to generate
high-quality ground truth data.

Data Generation
----------------------------

We use **Gaussian Random Fields (GRFs)** to generate physically realistic, 
smooth source terms $f$. The exact ground truth $u$ is solved using an ultra-fast Spectral Poisson Solver.

This choice is intentional:

- GRFs produce smooth, spatially correlated inputs.
- The resulting solutions resemble physically meaningful fields.
- Training on such data improves generalization to realistic PDE settings.

For each generated source term, the corresponding ground truth solution
:math:`u` is computed using an ultra-fast **spectral Poisson solver**. This
ensures that the model is trained against highly accurate reference solutions.

Model Configuration
----------------------------

For the 2D problem, we use the following preset:

- **Model:** ``FNO_Medium2D``
- **Hidden channels:** 128
- **Fourier layers:** 4
- **Retained modes:** 24 × 24

This configuration provides a good balance between expressiveness and
computational cost.

Loss Function
----------------------------

Training uses a **Relative L2 Loss**, which is more informative than absolute
error for PDE solutions:

- A loss value of ``0.05`` roughly means that the prediction deviates by
  **5% from the true solution on average**.
- The loss automatically adapts to the scale of the solution field, making
  comparisons across samples more meaningful.

Training Pipeline
----------------------------

The training script manages deterministic GPU operations, dataset standard 
scaling, cosine learning rate schedules, and model serialization seamlessly.

As a result, running the script is largely hands-off once configured.


3D Poisson Example
------------------

The 3D variant extends the same idea to volumetric domains and is implemented in:

::

   scripts/poisson3d_fno_train_lploss.py

Problem Setup
----------------------------

The domain is extended to a cube $[0, 1] \times [0, 1] \times [0, 1]$ with periodic boundary conditions.

Learning operators in 3D is significantly more challenging due to increased
memory requirements and computational cost.

Grid Resolution and Complexity
----------------------------

To keep training tractable while still capturing 3D structure, we use a **32 × 32 × 32** spatial grid
This resolution strikes a practical balance between accuracy and resource usage.

Model Configuration
----------------------------

For the 3D case, we use:

- **Model:** ``FNO_Medium3D``
- **Retained Fourier modes:** 16 × 16 × 16

Despite using fewer modes than in 2D, the model is still able to learn rich
three-dimensional solution structures.

Visualization Strategy
----------------------------

Visualizing 3D PDE solutions directly is non-trivial. To address this, we
provide dedicated visualization scripts:

::

   experiments/visualization/poisson3d_fno_plot.py

These scripts generate:

- 2D slice views (e.g., mid-Z or mid-Y planes),
- 3D point-cloud renderings,
- color-mapped absolute error distributions.

This makes it much easier to interpret model behavior and error patterns.


Visualization Outputs
---------------------

For both 2D and 3D experiments, visualization scripts are provided:

::

   poisson2d_fno_plot.py
   poisson3d_fno_plot.py

Running these scripts performs the following steps automatically:

1. Load the trained model checkpoint.
2. Re-apply the exact data normalization used during training.
3. Run inference on test samples.
4. Generate side-by-side visual comparisons.

Each visualization typically includes:

1. **Source Term**  
   The input field :math:`f` provided to the FNO.

2. **Ground Truth Solution**  
   The reference solution computed using the spectral solver.

3. **FNO Prediction**  
   The model’s fast, learned approximation.

4. **Absolute Error Map**  
   A detailed visualization showing where and how much the prediction deviates
   from the ground truth.

These plots provide an intuitive and transparent way to evaluate both accuracy
and failure modes of the learned operator.