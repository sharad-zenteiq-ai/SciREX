# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org


import jax
import jax.numpy as jnp
import opt_einsum


def _view_as_real(x: jnp.ndarray) -> jnp.ndarray:
    """Convert complex tensor to real representation (..., 2)."""
    return jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)


def _view_as_complex(x: jnp.ndarray) -> jnp.ndarray:
    """Convert real representation (..., 2) back to complex."""
    return x[..., 0] + 1j * x[..., 1]


def _einsum_complexhalf_two_input(eq: str, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Efficient two-input einsum for complex tensors using float32 precision.


    This is a workaround for performing half-precision complex einsum by
    decomposing into real and imaginary parts.
    """

    if "->" not in eq:
        raise ValueError("Einsum equation must contain '->'")

    # Convert to real + float32
    a = _view_as_real(a).astype(jnp.float32)
    b = _view_as_real(b).astype(jnp.float32)

    # Parse equation
    lhs, rhs = eq.split("->")
    in1, in2 = lhs.split(",")

    # Add extra dims for real/imag
    new_eq = f"{in1}r,{in2}s->rs{rhs}"

    # Perform einsum
    tmp = jnp.einsum(new_eq, a, b, optimize=True)

    # Reconstruct complex result
    real = tmp[0, 0] - tmp[1, 1]
    imag = tmp[1, 0] + tmp[0, 1]

    return _view_as_complex(jnp.stack([real, imag], axis=-1))


def _einsum_complexhalf_general(eq: str, *args: jnp.ndarray) -> jnp.ndarray:
    """
    General multi-input einsum for complex tensors in float32 precision.
    Uses opt_einsum to determine optimal contraction path.
    """

    if "->" not in eq:
        raise ValueError("Einsum equation must contain '->'")

    # Compute optimal contraction path
    _, path_info = opt_einsum.contract_path(eq, *args)
    contractions = [c[2] for c in path_info.contraction_list]

    input_labels = eq.split("->")[0].split(",")
    output_label = eq.split("->")[1]

    # Map tensors
    tensors = {label: _view_as_real(t).astype(jnp.float32)
               for label, t in zip(input_labels, args)}

    for partial_eq in contractions:
        lhs, out = partial_eq.split("->")
        in1, in2 = lhs.split(",")

        a, b = tensors[in1], tensors[in2]

        # Build new equation
        new_eq = f"{in1}r,{in2}s->rs{out}"

        tmp = jnp.einsum(new_eq, a, b, optimize=True)

        real = tmp[0, 0] - tmp[1, 1]
        imag = tmp[1, 0] + tmp[0, 1]

        tensors[out] = jnp.stack([real, imag], axis=-1)

    return _view_as_complex(tensors[output_label])


def einsum(eq: str, *operands: jnp.ndarray) -> jnp.ndarray:
    """
    Einsum for complex tensors using custom implementation.

    This function always routes through the custom complex-half
    compatible implementation, ensuring consistent behavior.
    """

    if len(operands) == 2:
        return _einsum_complexhalf_two_input(eq, *operands)

    return _einsum_complexhalf_general(eq, *operands)


einsum_jit = jax.jit(einsum, static_argnames=["eq"])