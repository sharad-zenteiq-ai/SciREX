import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.data.car_cfd_dataset import CarCFDDataset

# Configuration - Forced to match original repo defaults for comparison
DATA_ROOT = "/home/gazania/zan_folder/SciREX/scirex/operators/data/car_cfd_data"
RADIUS = 0.033 
QUERY_RES = [16, 16, 16] 
MAX_NEIGHBORS = 600 # Set high to ensure no truncation during comparison

def export_local():
    print(f"Initializing local dataset from {DATA_ROOT}...")
    ds = CarCFDDataset(
        root_dir=DATA_ROOT,
        n_train=1,
        n_test=0,
        query_res=QUERY_RES,
        download=False,
        in_gno_radius=RADIUS,
        max_neighbors=MAX_NEIGHBORS,
        use_cache=False # Force fresh computation for test
    )

    # Extract data for the first sample
    sample = ds.train_data[0]
    query_coords = ds._constant['query_points'].reshape(-1, 3)
    neighbor_indices = sample['in_neighbors']['neighbor_indices']
    neighbor_masks = sample['in_neighbors']['mask']

    rows = []
    print(f"Exporting neighbors for {len(query_coords)} query points...")

    for i in range(len(query_coords)):
        # Get actual IDs (where mask == 1) and sort them
        ids = neighbor_indices[i][neighbor_masks[i] == 1.0].tolist()
        sorted_ids = sorted([int(idx) for idx in ids])
        
        rows.append({
            "Query_ID": i,
            "X": float(query_coords[i, 0]),
            "Y": float(query_coords[i, 1]),
            "Z": float(query_coords[i, 2]),
            "Num_Neighbors": len(sorted_ids),
            "Sorted_IDs": ",".join(map(str, sorted_ids))
        })

    # Save to CSV
    df = pd.DataFrame(rows)
    output_path = "neighbors_scirex_local.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")
    print(f"Average neighbors per point: {df['Num_Neighbors'].mean():.2f}")
    print(f"Max neighbors found: {df['Num_Neighbors'].max()}")

if __name__ == "__main__":
    export_local()
