import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scirex.operators.data.car_cfd_dataset import CarCFDDataset
from configs.gino_carcfd_config import GINOCarCFDConfig

def final_check():
    print("--- 🏁 GINO Final Pipeline Verification ---")
    config = GINOCarCFDConfig()
    
    # Initialize the NEW dataset structure
    dataset = CarCFDDataset(
        root_dir=config.data_root,
        n_train=1,
        n_test=0,
        query_res=config.query_res,
        max_vertices=4096, # As per the new pull
        max_neighbors=config.max_neighbors,
        in_gno_radius=config.in_gno_radius,
        out_gno_radius=config.out_gno_radius,
        neighbor_cache_dir="./scirex/operators/data/neighbor_cache",
        use_cache=True
    )
    
    # Get a batch
    batch_gen = dataset.get_batch("train", batch_size=1)
    batch = next(batch_gen)
    
    print("\n[1] Shape Verification (Teammate's Padding Logic):")
    print(f"  - Car Vertices:  {batch['vertices'].shape} (Expected: (1, 4096, 3))")
    print(f"  - Pressure Field: {batch['press'].shape}    (Expected: (1, 4096, 1))")
    print(f"  - Latent Grid:    {batch['query_points'].shape} (Expected: (1, 32, 32, 32, 3))")

    print("\n[2] KD-Tree & Neighbor Cache Verification:")
    if "in_neighbors" not in batch:
        print("  ❌ ERROR: Neighbor data missing. Check cache path.")
        return

    in_nb = batch["in_neighbors"]
    mask = in_nb["neighbors_mask"]
    active_pts = np.sum(mask.any(axis=-1))
    
    print(f"  - Mapping Success: ✅ Pre-computed indices found.")
    print(f"  - Active Grid Influence: {active_pts} / 32768 points are connected to the car.")

    # [3] Final Architecture Visualization
    print("\n[3] Generating High-Fidelity 3D Visualization...")
    verts = batch["vertices"][0]
    press = batch["press"][0]
    grid = batch["query_points"][0].reshape(-1, 3)
    idx_map = in_nb["neighbors_index"][0]
    mask_map = in_nb["neighbors_mask"][0]

    fig = plt.figure(figsize=(16, 7))

    # LEFT Panel: The Car and its Physics Field
    ax1 = fig.add_subplot(121, projection='3d')
    p1 = ax1.scatter(verts[:,0], verts[:,1], verts[:,2], c=press[:,0], cmap='jet', s=3, alpha=0.9)
    fig.colorbar(p1, ax=ax1, label='Pressure Field (Ground Truth)', shrink=0.5)
    ax1.set_title("Input car Surface + Air Pressure")
    ax1.view_init(elev=20, azim=135)

    # RIGHT Panel: The Grid and Neighbor Mapping
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_axis_off()
    
    # Subsample grid for visualization
    g_sub = grid.reshape(32,32,32,3)[::4, ::4, ::4, :].reshape(-1, 3)
    ax2.scatter(g_sub[:,0], g_sub[:,1], g_sub[:,2], color='black', s=1, alpha=0.1)
    ax2.scatter(verts[:,0], verts[:,1], verts[:,2], color='green', s=0.5, alpha=0.05)

    # Visualize one grid point mapping to car
    valid_q = np.where(mask_map.sum(axis=-1) >= 4)[0]
    q_idx = valid_q[len(valid_q)//2]
    q_pt = grid[q_idx]
    
    neigh_indices = idx_map[q_idx][mask_map[q_idx]]
    neigh_pts = verts[neigh_indices]

    ax2.scatter(q_pt[0], q_pt[1], q_pt[2], color='red', s=80, edgecolors='black', label='Latent point')
    ax2.scatter(neigh_pts[:,0], neigh_pts[:,1], neigh_pts[:,2], color='red', s=15, label='Neighbors')
    
    for pt in neigh_pts:
        ax2.plot([q_pt[0], pt[0]], [q_pt[1], pt[1]], [q_pt[2], pt[2]], color='red', linestyle='--', linewidth=0.5, alpha=0.6)

    ax2.set_title("Grid Mapping (KD-Tree Result)")
    ax2.view_init(elev=20, azim=135)

    plt.savefig("gino_final_verification.png", dpi=200)
    print("\n✅ Verification complete. Visualization saved to: gino_final_verification.png")

if __name__ == "__main__":
    final_check()
