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

from typing import Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp

class SpectralConv2D(nn.Module):
    """
    2D Spectral Convolution Layer.
    """
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, int]
    init_std: float = None # If None, uses (2 / (in + out))**0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, nx, ny, in_channels)
        batch, nx, ny, in_channels = x.shape
        modes_x, modes_y = self.n_modes

        # 1. FFT
        x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm="ortho")

        # 2. Weights
        weights_shape = (in_channels, self.out_channels, modes_x, modes_y)
        
        # Xavier/Glorot initialization scale (standard in original neuraloperator)
        if self.init_std is None:
            scale = (2.0 / (in_channels + self.out_channels))**0.5
        else:
            scale = self.init_std
            
        w1 = self.param('weights1', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
        w2 = self.param('weights2', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)

        # 3. Multiply with weights
        out_ft = jnp.zeros((batch, nx, ny // 2 + 1, self.out_channels), dtype=jnp.complex64)
        
        # Upper corners
        out_ft = out_ft.at[:, :modes_x, :modes_y, :].set(
            jnp.einsum("bxyi,ioxy->bxyo", x_ft[:, :modes_x, :modes_y, :], w1)
        )
        # Lower corners
        out_ft = out_ft.at[:, -modes_x:, :modes_y, :].set(
            jnp.einsum("bxyi,ioxy->bxyo", x_ft[:, -modes_x:, :modes_y, :], w2)
        )

        # 4. Inverse FFT
        x = jnp.fft.irfftn(out_ft, s=(nx, ny), axes=(1, 2), norm="ortho")
        return x

class SpectralConv3D(nn.Module):
    """
    3D Spectral Convolution Layer.
    """
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, int, int]
    init_std: float = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, nx, ny, nz, in_channels)
        batch, nx, ny, nz, in_channels = x.shape
        modes_x, modes_y, modes_z = self.n_modes

        # 1. FFT
        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3), norm="ortho")

        # 2. Weights
        weights_shape = (in_channels, self.out_channels, modes_x, modes_y, modes_z)
        
        if self.init_std is None:
            scale = (2.0 / (in_channels + self.out_channels))**0.5
        else:
            scale = self.init_std
            
        w1 = self.param('weights1', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
        w2 = self.param('weights2', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
        w3 = self.param('weights3', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
        w4 = self.param('weights4', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)

        # 3. Multiply with weights
        out_ft = jnp.zeros((batch, nx, ny, nz // 2 + 1, self.out_channels), dtype=jnp.complex64)
        
        # Four corners in 3rd dimension (due to rfft optimization in last dim)
        out_ft = out_ft.at[:, :modes_x, :modes_y, :modes_z, :].set(
            jnp.einsum("bxyzi,ioxyz->bxyzo", x_ft[:, :modes_x, :modes_y, :modes_z, :], w1)
        )
        out_ft = out_ft.at[:, -modes_x:, :modes_y, :modes_z, :].set(
            jnp.einsum("bxyzi,ioxyz->bxyzo", x_ft[:, -modes_x:, :modes_y, :modes_z, :], w2)
        )
        out_ft = out_ft.at[:, :modes_x, -modes_y:, :modes_z, :].set(
            jnp.einsum("bxyzi,ioxyz->bxyzo", x_ft[:, :modes_x, -modes_y:, :modes_z, :], w3)
        )
        out_ft = out_ft.at[:, -modes_x:, -modes_y:, :modes_z, :].set(
            jnp.einsum("bxyzi,ioxyz->bxyzo", x_ft[:, -modes_x:, -modes_y:, :modes_z, :], w4)
        )

        # 4. Inverse FFT
        x = jnp.fft.irfftn(out_ft, s=(nx, ny, nz), axes=(1, 2, 3), norm="ortho")
        return x
