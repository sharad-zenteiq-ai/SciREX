Fourier Neural Operators
=======================

This document walks through the **Fourier Neural Operator (FNO)** implementation in **SciREX** and explains how it is used to solve *parametric partial differential equations (PDEs)*.
It represents a practical, end-to-end guide covering the operator theory and our native **JAX/Flax** implementation (``FNO2D`` and ``FNO3D``).


Operator Learning
---------------------------------

Many problems in science and engineering involve solving PDEs. Traditional
numerical solvers—such as **finite difference** or **finite element** methods—
work by discretizing the domain into a grid and solving the equation on that grid.
This approach is reliable, but it comes with limitations:

- very fine meshes are expensive,
- changing resolution often means re-running everything,
- solving many similar PDE instances is slow.

Operator learning takes a different perspective.

Instead of learning mappings between **vectors**, we learn mappings between
**functions**. The model does not solve one PDE at a time—it learns an *operator*
that maps inputs (coefficients, source terms, boundary conditions) directly to
solutions.

The key idea is to **learn in continuous function space**, not on a fixed grid.

The neural operator is mesh-independent, different from the standard deep 
learning methods such as CNNs. It can be trained on one mesh and evaluated on 
another. By parameterizing the model in function space, it learns the continuous 
solution function instead of discretized vectors.

Comparison
^^^^^^^^^^

+----------------------------+-----------------------------+
| Conventional PDE Solvers   | Neural Operators            |
+============================+=============================+
| Solve one PDE instance     | Learn a family of PDEs      |
+----------------------------+-----------------------------+
| Require explicit equations| Data-driven, black-box      |
+----------------------------+-----------------------------+
| Resolution-dependent       | Resolution & mesh invariant |
+----------------------------+-----------------------------+
| Slow on fine grids         | Fast inference after training|
+----------------------------+-----------------------------+


The Fourier Layer (Core Idea)
----------------------------

A standard convolutional neural network only sees *local neighborhoods*. If information needs to travel
across the domain, you stack many layers and hope it propagates. On the other hand,
PDE solutions are different: they are smooth, global functions
defined over continuous domains. It is more efficient to represent them in Fourier space and do a global 
convolution.

There are two main motivations for using the Fourier transform:

1. **Fast:** Convolution via Fourier transform on a regular grid is quasilinear, compared to $O(n^2)$ for standard integration.
2. **Efficient:** Because PDEs deal with continuous functions, they reside efficiently in Fourier space.
3. **Global context:** Fourier modes naturally capture long-range dependencies, which are common in PDEs.

A convolution in physical space becomes a pointwise multiplication in Fourier
space. Each Fourier layer follows three steps:

1. Transform to Fourier space:

   .. math::

      x \xrightarrow{\mathcal{F}} \hat{x}

2. Apply a learnable linear transform on the lower Fourier modes.

3. Transform back to physical space:

   .. math::

      \hat{x} \xrightarrow{\mathcal{F}^{-1}} x

The result is combined with a bias term (a standard linear layer), followed by a
nonlinear activation in the spatial domain. This nonlinearity helps recover high-frequency details and
handle non-periodic boundaries left out in the Fourier layers.


FNO Implementation in SciREX
----------------------------

In SciREX, FNO is implemented **from scratch** using **JAX and Flax**, located in:

::

   scirex/operators/

Rather than wrapping external libraries, each component is built explicitly
using modular Flax layers. This keeps the implementation transparent, extensible,
and easy to debug.

Below is a conceptual view of the spectral convolution used in 2D:

.. code-block:: python

   class SpectralConv2D(nn.Module):
       hidden_channels: int
       n_modes: tuple[int, int]

       @nn.compact
       def __call__(self, x):
           # 1. Transform to Fourier space
           x_ft = jnp.fft.rfft2(x)

           # 2. Apply learnable weights on selected low-frequency modes
           # (with careful truncation handling)

           # 3. Transform back to physical space
           x = jnp.fft.irfft2(out_ft)
           return x

This layer performs a **global convolution**, independent of grid resolution.


Full Model Architecture
-----------------------

The ``FNO2D`` and ``FNO3D`` classes follow a clean encoder–operator–decoder structure with global Fourier connections.

1. Lifting Layer
^^^^^^^^^^^^^^^^

A ``ChannelMLP`` expands the input features (e.g., $x, y$ coordinates and the source term) to a larger hidden dimension.

2. Iterative FNO Blocks
^^^^^^^^^^^^^^^^^^^^^

The core of the model consists of multiple FNO blocks. Each block contains:

- ``SpectralConv`` (global interaction in Fourier space),
- ``SkipConnection`` (local interaction in physical space),
- Instance Normalization,
- ``GELU`` activation,
- ``ChannelMLP`` for pointwise refinement.

This combination balances global structure with local detail.

3. Projection Layer
^^^^^^^^^^^^^^^^^^^

The final projection layer maps hidden features back to the target output
(for example, a scalar solution field :math:`u`).


Conclusion
----------

The Fourier Neural Operator provides a powerful framework for learning PDE
solution operators directly in function space. In SciREX, the implementation emphasizes the deep theoretical power of global spectral convolutions with rigorous, validated,
and optimized software engineering paradigms utilizing JAX.