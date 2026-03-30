import os
import sys
import numpy as np
import torch
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.data.car_cfd_dataset import CarCFDDataset as JAX_CarCFDDataset
from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset as PT_CarCFDDataset

DATA_ROOT = "/home/gazania/zan_folder/SciREX/scirex/operators/data/car_cfd_data"

def compare_pressure():
    # ── JAX / SciREX ──
    jax_ds = JAX_CarCFDDataset(
        root_dir=DATA_ROOT,
        n_train=1,
        n_test=0,
        query_res=[8, 8, 8],
        download=False,
        use_cache=False,
    )
    jax_sample = jax_ds.train_data[0]
    jax_press = jax_sample["press"]

    # ── PyTorch / neuralop ──
    # Resolve the path like neuralop does
    pt_root = DATA_ROOT + "/processed-car-pressure-data"
    
    # We need to mock open3d because it's required by PT MeshDataModule init
    # but we only want to check the pressure loading which happens after super().__init__
    # Wait, the init might fail if open3d is missing.
    try:
        pt_ds = PT_CarCFDDataset(
            root_dir=pt_root,
            n_train=1,
            n_test=0,
            query_res=[8, 8, 8],
            download=False,
        )
        pt_sample = pt_ds.train_data[0]
        pt_press = pt_sample["press"].numpy()
        
        print(f"JAX Pressure Shape: {jax_press.shape}")
        print(f"PT Pressure Shape:  {pt_press.shape}")
        
        # Note: Both normalize pressure using GaussianNormalizer.
        # Since we only loaded 1 sample, the mean and std will be based on that sample alone.
        # So they should be identical.
        
        diff = np.abs(jax_press - pt_press).max()
        print(f"Max Pressure Difference: {diff:.8e}")
    except ModuleNotFoundError:
        print("Skipping PT comparison because open3d is missing.")

if __name__ == "__main__":
    compare_pressure()
