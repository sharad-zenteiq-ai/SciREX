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
class Burgers1Ddataloader:

    # Data generation
    resolution: int
    number_of_samples: int  

    def data_loader(self):
        # Get the directory where this config file is located
        config_dir = Path(__file__).parent
        zenodo_get.download(
        record_or_doi="19368148",
        output_dir=os.path.join(os.getcwd(), "data"),)

        # Navigate to data folder in parent directory
        data_path = config_dir.parent / 'data' / 'burgers_v100_t100_r1024_N2048.mat'
        data = scipy.io.loadmat(data_path)

        input_function = data['input']   #<---- This input function is the field u at t=0
        output_function = data['output'] #<---- This output function is the field u at t=linspace(0,1,100)


        self.resolution = len(data['input'])
        in_channels = 2 # 1 for u and 1 for x

        batch_size = len(input_function[0])

        # We have to manually curate this because the value of the x 
        # point for that grid point is not present in the data.
        x_grid_array = jnp.linspace(0,2*jnp.pi,self.resolution)
        
        # Now we multiply this x_grid_array to match the shape of input function
        x_grid = jnp.tile(x_grid_array, (batch_size,1))

        # Now we stack the x_grid and input function so that the input ot the model
        #takes the form (batch_size, resolution, in_channels)

        u0_input = jnp.array(input_function)
        input_data = jnp.stack([x_grid, u0_input], axis=2)

        # Similarly to the last line
        output_data = jnp.array(output_function)


        # We will also output the input_shape as a return of this function
        input_shape = (batch_size, self.resolution, in_channels)

        

        return input_shape, input_data, output_data


