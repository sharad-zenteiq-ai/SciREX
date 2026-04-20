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
Experiment configurations for FNO on Burgers' equation (2D and 3D).

These configs compose a *model preset* from ``configs.models`` with
experiment-specific training and data-generation parameters.

Usage
-----
    from configs.burgers_fno_config import FNO1DConfig

    config = FNO2DConfig()
    # config.model   → FNO_Medium2D instance  (architecture params)
    # config.*       → training / data params  (lr, batch_size, …)
"""
import scipy.io
import jax.numpy as jnp
import os
import zenodo_get 
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Burgers1p1Ddataloader:

    # Data generation
    resolution: int
    number_of_samples: int  

    def data_loader(extension):
        # Get the directory where this config file is located
        config_dir = Path(__file__).parent
        zenodo_get.download(
        record_or_doi="19368148",
        output_dir=os.path.join(os.getcwd(), "data"),)

        # Navigate to data folder in parent directory
        data_path = config_dir.parent / 'data' / 'burgers_v100_t100_r1024_N2048.mat'
        data = scipy.io.loadmat(data_path)

        input_function = data['input']   #<---- This input function is the field u at t=0 at x=(0,pi)
        output_function = jnp.expand_dims(data['output'],axis=3) #<---- This output function is the field u at t=linspace(0,1,100) for the same x grid as input function


        resolution = len(data['input'][0])#<---- This is the resolution of the grid on which the field u is defined. It is 1024 for this dataset.
        in_channels = 2 # 1 for u and 1 for x

        batch_size = len(input_function)

        nt = 101  # 1 step for t=0, plus 100 future steps
        in_channels_new = 3 # x, t, u

        # 1. Construct the complete (x, t) meshgrid for the entire space-time domain
        x_grid_array = jnp.linspace(0, jnp.pi, resolution)
        t_grid_array = jnp.linspace(0, 1, nt) 

        # 'ij' indexing ensures the mesh grid aligns with shape (nt, resolution) -> (101, 1024)
        T_mesh, X_mesh = jnp.meshgrid(t_grid_array, x_grid_array, indexing='ij')

        # Broadcast the meshgrids to include the batch dimension -> (2048, 101, 1024)
        X_grid = jnp.broadcast_to(X_mesh, (batch_size, nt, resolution))
        T_grid = jnp.broadcast_to(T_mesh, (batch_size, nt, resolution))

        # 2. Isolate the 'u' channel and apply the extension methods ONLY to 'u'
        # Initial condition shape: (2048, 1024). Expand to: (2048, 1, 1024)
        u0_expanded = jnp.expand_dims(jnp.array(input_function), axis=1)

        if extension == "broadcast":
            # Broadcast u(t=0) uniformly across all 101 time steps
            U_grid = jnp.broadcast_to(u0_expanded, (batch_size, nt, resolution))
            
        elif extension == "zeros":
            # Keep true values at u(t=0), but pad all t>0 steps with zeros
            zeros_u = jnp.zeros((batch_size, nt - 1, resolution))
            U_grid = jnp.concatenate([u0_expanded, zeros_u], axis=1)
            
        elif extension == "ones":
            # Keep true values at u(t=0), but pad all t>0 steps with ones
            ones_u = jnp.ones((batch_size, nt - 1, resolution))
            U_grid = jnp.concatenate([u0_expanded, ones_u], axis=1)
            
        else:
            raise ValueError("Invalid extension method. Choose from 'broadcast', 'zeros', or 'ones'.")

        # 3. Stack x, t, and u together along the final channel axis
        # Final shape: (batch_size, nt, resolution, 3) -> (2048, 101, 1024, 3)
        input_data_3D = jnp.stack([X_grid, T_grid, U_grid], axis=-1)

        return input_data_3D.shape, input_data_3D, output_function


