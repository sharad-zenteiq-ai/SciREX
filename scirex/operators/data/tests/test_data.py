import numpy as np
import pytest
from scirex.operators.data.poisson import random_poisson_2d_batch, random_poisson_3d_batch

def test_random_poisson_2d_batch():
    batch_size = 2
    nx, ny = 16, 16
    f, u = random_poisson_2d_batch(batch_size, nx, ny)
    
    # f should have (batch, nx, ny, 3) because of include_mesh (source + x + y)
    assert f.shape == (2, 16, 16, 3)
    # u should have (batch, nx, ny, 1)
    assert u.shape == (2, 16, 16, 1)
    assert f.dtype == np.float32
    assert u.dtype == np.float32

def test_random_poisson_3d_batch():
    batch_size = 1
    nx, ny, nz = 8, 8, 8
    f, u = random_poisson_3d_batch(batch_size, nx, ny, nz, include_mesh=True)
    
    # f should have (batch, nx, ny, nz, 4) (source + x + y + z)
    assert f.shape == (1, 8, 8, 8, 4)
    assert u.shape == (1, 8, 8, 8, 1)
