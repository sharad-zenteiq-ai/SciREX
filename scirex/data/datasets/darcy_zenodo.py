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
Load Darcy Flow dataset from numpy files.
The data should be converted from Zenodo format using scripts/convert_darcy_zenodo.py

Data source: https://zenodo.org/records/12784353
"""

import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


def load_darcy_numpy(
    root_dir: str,
    resolution: int = 64,
    n_train: int = 1000,
    n_test: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Darcy flow data from numpy files.
    
    First run: python scripts/convert_darcy_zenodo.py <resolution>
    This will download from Zenodo and convert to .npy format.
    
    Parameters
    ----------
    root_dir : str
        Directory containing .npy files
    resolution : int
        Grid resolution (16, 32, 64, 128, or 421)
    n_train : int
        Number of training samples to use
    n_test : int
        Number of test samples to use
        
    Returns
    -------
    a_train : np.ndarray
        Training permeability fields, shape (n_train, resolution, resolution, 1)
    u_train : np.ndarray
        Training pressure solutions, shape (n_train, resolution, resolution, 1)
    a_test : np.ndarray
        Test permeability fields, shape (n_test, resolution, resolution, 1)
    u_test : np.ndarray
        Test pressure solutions, shape (n_test, resolution, resolution, 1)
    """
    root_path = Path(root_dir)
    
    train_file = root_path / f"darcy_train_{resolution}.npy"
    test_file = root_path / f"darcy_test_{resolution}.npy"
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Darcy data not found at {root_dir}\n"
            f"Run: python scripts/convert_darcy_zenodo.py {resolution}"
        )
    
    # Load numpy files
    train_data = np.load(train_file, allow_pickle=True).item()
    test_data = np.load(test_file, allow_pickle=True).item()
    
    # Extract permeability (a) and solution (u)
    a_train = train_data['x'][:n_train]
    u_train = train_data['y'][:n_train]
    a_test = test_data['x'][:n_test]
    u_test = test_data['y'][:n_test]
    
    # Add channel dimension to match expected format (NHWC)
    a_train = a_train[..., None]  # (n_train, resolution, resolution, 1)
    u_train = u_train[..., None]
    a_test = a_test[..., None]
    u_test = u_test[..., None]
    
    print(f"Loaded Darcy data: train={a_train.shape}, test={a_test.shape}")
    return a_train, u_train, a_test, u_test


def generator_from_numpy(
    root_dir: str,
    resolution: int = 64,
    n_train: int = 1000,
    batch_size: int = 16,
    num_batches: int = 100,
    shuffle: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Create a batch generator from numpy Darcy data.
    
    Parameters
    ----------
    root_dir : str
        Directory containing .npy files
    resolution : int
        Grid resolution
    n_train : int
        Number of training samples
    batch_size : int
        Batch size
    num_batches : int
        Number of batches to yield
    shuffle : bool
        Whether to shuffle data each epoch
        
    Yields
    ------
    a_batch : np.ndarray
        Permeability batch, shape (batch_size, resolution, resolution, 1)
    u_batch : np.ndarray
        Pressure batch, shape (batch_size, resolution, resolution, 1)
    """
    a_train, u_train, _, _ = load_darcy_numpy(
        root_dir, resolution, n_train, n_test=100
    )
    
    n_samples = a_train.shape[0]
    indices = np.arange(n_samples)
    
    for batch_idx in range(num_batches):
        if shuffle and batch_idx % (n_samples // batch_size) == 0:
            np.random.shuffle(indices)
        
        # Wrap around if we run out of samples
        start_idx = (batch_idx * batch_size) % n_samples
        end_idx = start_idx + batch_size
        
        if end_idx <= n_samples:
            batch_indices = indices[start_idx:end_idx]
        else:
            # Wrap around
            batch_indices = np.concatenate([
                indices[start_idx:],
                indices[:end_idx - n_samples]
            ])
        
        yield a_train[batch_indices], u_train[batch_indices]
