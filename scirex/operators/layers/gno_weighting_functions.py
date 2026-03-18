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

"""
GNO Weighting (Cutoff) Functions for Mollified Graph Neural Operators.

Provides smooth, compactly-supported cutoff functions used to weight
neighbor contributions in kernel integral transforms.  Each function
maps distances ``x ∈ [0, radius]`` to weights ``w ∈ [0, scale]`` that
decay to zero at the boundary.

Reference
---------
Ported from the PyTorch implementation in
`neuraloperator <https://github.com/neuraloperator/neuraloperator>`_:
``neuralop/layers/gno_weighting_functions.py``.
"""

from functools import partial

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Individual cutoff functions
# ---------------------------------------------------------------------------

def bump_cutoff(x, radius=1.0, scale=1.0, eps=1e-7):
    """
    Bump cutoff function with exponential decay.

    Formula: ``w(x) = scale * e * exp(-1 / (1 - d² + ε))``
    where ``d = x / radius``, ``x ∈ [0, radius]``.

    Parameters
    ----------
    x : jnp.ndarray
        Input distances (non-negative).
    radius : float
        Support radius; distances beyond this are clipped.
    scale : float
        Multiplicative scale applied to the output.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    jnp.ndarray
        Weighted values in ``[0, scale]``.
    """
    out = jnp.clip(x, 0.0, radius) / radius
    out = -1.0 / ((1.0 - out ** 2) + eps)
    return jnp.exp(out) * jnp.e * scale


def half_cos_cutoff(x, radius=1.0, scale=1.0):
    """
    Half-cosine cutoff function with smooth decay.

    Formula: ``w(x) = scale * (0.5 * cos(π * d) + 0.5)``
    where ``d = x / radius``, ``x ∈ [0, radius]``.

    Parameters
    ----------
    x : jnp.ndarray
        Input distances (non-negative).
    radius : float
        Support radius.
    scale : float
        Multiplicative scale.

    Returns
    -------
    jnp.ndarray
        Weighted values in ``[0, scale]``.
    """
    d = x / radius
    return scale * (0.5 * jnp.cos(jnp.pi * d) + 0.5)


def quadr_cutoff(x, radius=1.0, scale=1.0):
    """
    Quadratic cutoff function (piecewise).

    Formula::

        w(x) = scale * { 1 - 2d²,    if d < 0.5
                       { 2(1 - d)²,   if d ≥ 0.5

    where ``d = x / radius``, ``x ∈ [0, radius]``.

    Parameters
    ----------
    x : jnp.ndarray
        Input distances (non-negative).
    radius : float
        Support radius.
    scale : float
        Multiplicative scale.

    Returns
    -------
    jnp.ndarray
        Weighted values in ``[0, scale]``.
    """
    d = x / radius
    left = 1.0 - 2.0 * d ** 2
    right = 2.0 * (1.0 - d) ** 2
    return scale * jnp.where(d < 0.5, left, right)


def quartic_cutoff(x, radius=1.0, scale=1.0):
    """
    Quartic cutoff function (fourth-order polynomial).

    Formula: ``w(x) = (scale/r⁴)x⁴ - (2·scale/r²)x² + scale``

    Parameters
    ----------
    x : jnp.ndarray
        Input distances (non-negative).
    radius : float
        Support radius.
    scale : float
        Multiplicative scale.

    Returns
    -------
    jnp.ndarray
        Weighted values in ``[0, scale]``.
    """
    a = scale / radius ** 4
    c = -2.0 * scale / radius ** 2
    return a * x ** 4 + c * x ** 2 + scale


def octic_cutoff(x, radius=1.0, scale=1.0):
    """
    Octic cutoff function (eighth-order polynomial).

    Formula: ``w(x) = scale * (-3d⁸ + 8d⁶ - 6d⁴ + 1)``
    where ``d = x / radius``, ``x ∈ [0, radius]``.

    Parameters
    ----------
    x : jnp.ndarray
        Input distances (non-negative).
    radius : float
        Support radius.
    scale : float
        Multiplicative scale.

    Returns
    -------
    jnp.ndarray
        Weighted values in ``[0, scale]``.
    """
    d = x / radius
    return scale * (-3.0 * d ** 8 + 8.0 * d ** 6 - 6.0 * d ** 4 + 1.0)


# ---------------------------------------------------------------------------
# Registry & dispatch
# ---------------------------------------------------------------------------

WEIGHTING_FN_REGISTRY = {
    "bump": bump_cutoff,
    "half_cos": half_cos_cutoff,
    "quadr": quadr_cutoff,
    "quartic": quartic_cutoff,
    "octic": octic_cutoff,
}


def dispatch_weighting_fn(weight_function_name: str, sq_radius: float, scale: float):
    """
    Select and bind a GNO weighting function for use in a mollified GNO layer.

    Parameters
    ----------
    weight_function_name : str
        Key into ``WEIGHTING_FN_REGISTRY`` (e.g. ``"bump"``, ``"half_cos"``).
    sq_radius : float
        Squared radius of GNO neighborhoods for Nyström approximation.
    scale : float
        Factor by which to scale all weights.

    Returns
    -------
    Callable
        A partial function ``f(x) -> weighted_x`` with bound ``radius``
        and ``scale``.

    Raises
    ------
    NotImplementedError
        If ``weight_function_name`` is not in the registry.
    """
    base_func = WEIGHTING_FN_REGISTRY.get(weight_function_name)
    if base_func is None:
        raise NotImplementedError(
            f"weighting function should be one of "
            f"{list(WEIGHTING_FN_REGISTRY.keys())}, got {weight_function_name}"
        )
    return partial(base_func, radius=sq_radius, scale=scale)
