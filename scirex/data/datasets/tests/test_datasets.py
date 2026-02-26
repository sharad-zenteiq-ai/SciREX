import os
import shutil
from pathlib import Path

import pytest
import numpy as np
import torch

from ..darcy import random_darcy_batch
from ..poisson import random_poisson_batch
from ..poisson_3d import random_poisson_3d_batch
from ..data_utils import generate_poisson_data, generate_poisson_3d_data

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
