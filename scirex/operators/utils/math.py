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
Shared mathematical utilities for FNO operations.
"""
from typing import Tuple
import jax.numpy as jnp

def rfft2(x: jnp.ndarray) -> jnp.ndarray:
    """2D real-to-complex FFT along axes (1,2): (batch, nx, ny, ch) -> (..., ny//2+1, ch) complex"""
    return jnp.fft.rfft2(x, axes=(1, 2))

def irfft2(X: jnp.ndarray, s: Tuple[int, int]) -> jnp.ndarray:
    """Inverse 2D real FFT with explicit output spatial shape s=(nx, ny)."""
    return jnp.fft.irfft2(X, s=s, axes=(1, 2))
