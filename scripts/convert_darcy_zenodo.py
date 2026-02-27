"""
Script to generate synthetic Darcy flow data for WNO benchmarks.
Instead of downloading from Zenodo (which requires internet), this generates
standard FNO-style Darcy flow data locally using the built-in solver.

Usage:
    python scripts/convert_darcy_zenodo.py <resolution>
    
Example:
    python scripts/convert_darcy_zenodo.py 128
"""
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import the local solver
from scirex.data.datasets.darcy import solve_darcy_2d, generate_grf_2d_fno, threshold_permeability

def generate_dataset(resolution, n_samples, seed=0):
    print(f"Generating {n_samples} samples at {resolution}x{resolution}...")
    rng = np.random.default_rng(seed)
    
    a_list = []
    u_list = []
    
    nx, ny = resolution, resolution
    f = np.ones((nx, ny), dtype=np.float32)
    
    for i in tqdm(range(n_samples)):
        # Generate GRF (alpha=2.0, tau=3.0 standard FNO settings)
        grf = generate_grf_2d_fno(nx, ny, alpha=2.0, tau=3.0, rng=rng)
        
        # Threshold to binary permeability (values 3.0 and 12.0)
        # Note: Some papers use 4.0/12.0, some use 3.0/12.0. We use 3.0/12.0 here.
        a = threshold_permeability(grf, threshold=0.0, a_low=3.0, a_high=12.0)
        
        # Solve PDE
        u = solve_darcy_2d(a, f)
        
        a_list.append(a)
        u_list.append(u)
        
    return np.stack(a_list), np.stack(u_list)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_darcy_zenodo.py <resolution>")
        sys.exit(1)
        
    resolution = int(sys.argv[1])
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    data_dir = Path("scirex/data/datasets/darcy_fno")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train set
    train_file = data_dir / f"darcy_train_{resolution}.npy"
    if not train_file.exists():
        print(f"Generating training data ({n_train} samples)...")
        a_train, u_train = generate_dataset(resolution, n_samples=n_train, seed=42)
        # Save in the format expected by darcy_zenodo.py ('x' and 'y' keys)
        np.save(train_file, {'x': a_train, 'y': u_train})
        print(f"Saved {train_file}")
    else:
        print(f"Train file {train_file} already exists.")
        
    # 2. Test set
    test_file = data_dir / f"darcy_test_{resolution}.npy"
    if not test_file.exists():
        print(f"Generating test data ({n_test} samples)...")
        a_test, u_test = generate_dataset(resolution, n_samples=n_test, seed=100)
        np.save(test_file, {'x': a_test, 'y': u_test})
        print(f"Saved {test_file}")
    else:
        print(f"Test file {test_file} already exists.")
        
    print("✅ Data generation complete.")

if __name__ == "__main__":
    main()
