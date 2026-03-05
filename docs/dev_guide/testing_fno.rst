FNO Development Testing Suite
=============================

Testing scientific machine learning models requires more than checking that the
code runs without syntax errors. In SciREX, the testing suite located in

::

   scirex/operators/tests/

is designed to verify **mathematical correctness, numerical stability, and
actual learning behavior** of our Fourier Neural Operator (FNO) models.The goal of these tests is simple: ensure that every component from Fourier
layers to the training loop behaves exactly as expected before it is used in
larger experiments.

Below is a breakdown of the key areas covered by the testing suite and why each
one matters.


Neural Operator Layers
----------------------

A large part of FNO’s functionality comes from its spectral layers. These layers operate on Fourier coefficients and interact with spatial grids in non-trivial ways. The tests in this section ensure that these low-level operations behave correctly.
+------------------------+---------------------------------------------------------+--------------------------------------------------------------+
| Test Focus             | What it checks                                          | Why it matters                                               |
+========================+=========================================================+==============================================================+
| Spectral Convolution   | Passes inputs with different grid sizes and             | Prevents index errors when slicing Fourier modes, especially |
|                        | dimensionalities into ``SpectralConv2D`` and            | when dealing with non-square grids or unusual resolutions.   |
|                        | ``SpectralConv3D``.                                     |                                                              |
+------------------------+---------------------------------------------------------+--------------------------------------------------------------+
| Domain Padding         | Verifies that padding added before the FFT can be       | Ensures that temporary padding does not shift the spatial    |
|                        | cleanly removed after inverse transforms.               | solution or introduce subtle boundary artifacts.             |
+------------------------+---------------------------------------------------------+--------------------------------------------------------------+
| Skip Connections       | Tests the behavior of different bypass options:         | Skip paths allow gradients to move directly through the      |
|                        | ``Identity``, ``Linear``, and ``Soft-gating``.          | network. Ensuring these paths work correctly is essential    |
|                        |                                                         | for stable training.                                         |
+------------------------+---------------------------------------------------------+--------------------------------------------------------------+


Model Forward Passes
--------------------

Once the individual layers are validated, the next step is verifying that the
entire model pipeline runs correctly.

+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+
| Test Focus           | What it checks                                            | Why it matters                                              |
+======================+===========================================================+=============================================================+
| Full Model Pipeline  | Runs complete forward passes for ``FNO2D`` and            | This acts as a major sanity check. If any layer produces    |
|                      | ``FNO3D`` using randomized batched inputs.                | incompatible shapes or misaligned dimensions, the model     |
|                      | The test verifies that input and output shapes            | will fail here immediately.                                 |
|                      | remain consistent throughout the pipeline.                |                                                             |
+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+


Data and Loss Behavior
----------------------

Scientific ML models are extremely sensitive to data formatting and numerical
stability. These tests ensure that the training inputs and loss functions remain
well behaved.

+-------------------------+---------------------------------------------------------+--------------------------------------------------------------+
| Test Focus              | What it checks                                          | Why it matters                                               |
+=========================+=========================================================+==============================================================+
| Poisson Data Generation | Confirms that Gaussian Random Field (GRF) generation    | Guarantees that source terms and solutions are generated     |
|                         | and spectral solvers produce properly aligned           | correctly before reaching the accelerator or training loop.  |
|                         | ``(f, u)`` pairs as ``ndarray`` structures.             |                                                              |
+-------------------------+---------------------------------------------------------+--------------------------------------------------------------+
| Relative L2 Loss        | Tests edge cases such as near-zero targets and verifies | Prevents numerical instabilities like ``NaN`` values during  |
|                         | that the loss computation remains stable.               | training, which can otherwise silently break optimization.   |
+-------------------------+---------------------------------------------------------+--------------------------------------------------------------+


Training Stability and Convergence
----------------------------------

Beyond correctness, the most important question is whether the model can
**actually learn**. These tests simulate miniature training loops to ensure that
gradients propagate correctly through the spectral architecture.

+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+
| Test Focus           | What it checks                                            | Why it matters                                              |
+======================+===========================================================+=============================================================+
| TrainState Setup     | Ensures parameter trees initialize correctly and that     | Flax training states can easily break if parameters or      |
|                      | gradients are tracked properly by the optimizer.          | optimizer structures are misconfigured.                     |
+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+
| Step Progression     | Runs roughly 50 optimization steps on fixed data and      | If the final loss is not smaller than the initial loss,     |
|                      | verifies that training reduces the loss value.            | it indicates gradients are not flowing correctly through    |
|                      |                                                           | the model.                                                  |
+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+
| Toy Overfit Check    | Trains a very small FNO model (16 channels) to learn      | A model should easily memorize a trivial mapping. If it     |
|                      | a simple relationship: ``target = 2 × input``.            | cannot do so within a few hundred steps, the architecture   |
|                      | Training continues until the loss becomes negligible.     | or training pipeline is likely flawed.                      |
+----------------------+-----------------------------------------------------------+-------------------------------------------------------------+


Executing the Test Suite
------------------------

All tests are integrated with **pytest** and are designed to run quickly so
that they can be executed during regular development.

To run the complete set of FNO-related tests:

.. code-block:: bash

   PYTHONPATH=. pytest scirex/operators/ -v

This command executes every test associated with the Fourier Neural Operator
implementation and provides detailed output for each test case.