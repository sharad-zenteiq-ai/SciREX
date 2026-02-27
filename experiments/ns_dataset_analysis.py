import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_dataset(file_path, name="Dataset"):
    print(f"\n=== Analyzing {name} ===")
    print(f"Path: {file_path}")
    
    # Load data
    data = torch.load(file_path, map_location='cpu')
    
    if isinstance(data, dict):
        for key in data.keys():
            tensor = data[key]
            print(f"Key: {key}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Mean:  {tensor.mean().item():.4f}")
            print(f"  Std:   {tensor.std().item():.4f}")
            print(f"  Min:   {tensor.min().item():.4f}")
            print(f"  Max:   {tensor.max().item():.4f}")
    else:
        print(f"Shape: {data.shape}")
        print(f"Mean:  {data.mean().item():.4f}")
        print(f"Min:   {data.min().item():.4f}")
        print(f"Max:   {data.max().item():.4f}")
    
    return data

def visualize_samples(data, save_path):
    # Assume data is a dict with 'x' and 'y'
    x = data['x'] # (N, H, W) or (N_total, H, W)
    y = data['y']
    
    # Let's visualize the first 3 trajectories (assuming 10 steps each)
    T = 10
    n_trajectories = 3
    
    fig, axes = plt.subplots(n_trajectories * 2, T, figsize=(20, 4 * n_trajectories))
    
    for i in range(n_trajectories):
        for t in range(T):
            idx = i * T + t
            # Plot Input X
            ax_x = axes[i*2, t]
            im_x = ax_x.imshow(x[idx].numpy(), cmap='viridis')
            ax_x.axis('off')
            if t == 0: ax_x.set_title(f"Traj {i} - X (Input)")
            
            # Plot Target Y
            ax_y = axes[i*2 + 1, t]
            im_y = ax_y.imshow(y[idx].numpy(), cmap='magma')
            ax_y.axis('off')
            if t == 0: ax_y.set_title(f"Traj {i} - Y (Target)")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

def main():
    data_dir = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
    train_file = os.path.join(data_dir, 'nsforcing_train_128.pt')
    test_file = os.path.join(data_dir, 'nsforcing_test_128.pt')
    
    results_dir = 'experiments/results/analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    train_data = analyze_dataset(train_file, "NS Train")
    test_data = analyze_dataset(test_file, "NS Test")
    
    visualize_samples(train_data, os.path.join(results_dir, 'ns_samples_visualization.png'))

if __name__ == "__main__":
    main()
