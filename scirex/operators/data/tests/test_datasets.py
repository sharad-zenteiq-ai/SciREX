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
Unit tests for data generators and solvers, including Darcy and Poisson datasets.
"""

import os
import shutil
from pathlib import Path

import pytest
import numpy as np
import torch

from scirex.operators.data.darcy import random_darcy_batch
from scirex.operators.data.poisson import random_poisson_batch, random_poisson_3d_batch
from scirex.operators.data.data_utils import generate_poisson_data, generate_poisson_3d_data

test_data_dir = Path("./dataset_test")

@pytest.mark.parametrize("resolution", [16])
def test_DarcyDataset(resolution):
    a, u = random_darcy_batch(batch_size=2, nx=resolution, ny=resolution)
    
    assert a.shape == (2, resolution, resolution, 1)
    assert u.shape == (2, resolution, resolution, 1)
    assert isinstance(a, np.ndarray)
    assert isinstance(u, np.ndarray)

@pytest.mark.parametrize("resolution", [16])
def test_PoissonDataset(resolution):
    f, u = random_poisson_batch(batch_size=2, nx=resolution, ny=resolution)
    
    # f has 3 channels if include_mesh=True by default for poisson though it's manually constructed
    assert f.shape == (2, resolution, resolution, 3) 
    assert u.shape == (2, resolution, resolution, 1)
    assert isinstance(f, np.ndarray)
    assert isinstance(u, np.ndarray)

@pytest.mark.parametrize("resolution", [16])
def test_Poisson3DDataset(resolution):
    f, u = random_poisson_3d_batch(batch_size=2, nx=resolution, ny=resolution, nz=resolution)
    
    assert f.shape == (2, resolution, resolution, resolution, 4) 
    assert u.shape == (2, resolution, resolution, resolution, 1)
    assert isinstance(f, np.ndarray)
    assert isinstance(u, np.ndarray)

@pytest.mark.parametrize("resolution", [16])
def test_generate_poisson_data(resolution):
    input_data, output_data = generate_poisson_data(n_samples=2, nx=resolution, ny=resolution)
    
    assert input_data.shape == (2, 3, resolution, resolution) 
    assert output_data.shape == (2, 1, resolution, resolution)
    assert isinstance(input_data, torch.Tensor)
    assert isinstance(output_data, torch.Tensor)

@pytest.mark.parametrize("resolution", [16])
def test_generate_poisson_3d_data(resolution):
    input_data, output_data = generate_poisson_3d_data(n_samples=2, nx=resolution, ny=resolution, nz=resolution)
    
    assert input_data.shape == (2, 4, resolution, resolution, resolution) 
    assert output_data.shape == (2, 1, resolution, resolution, resolution)
    assert isinstance(input_data, torch.Tensor)
    assert isinstance(output_data, torch.Tensor)
