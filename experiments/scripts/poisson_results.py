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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json

def create_results():
    results_dir = Path("experiments/results/fno_poisson")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    pkl_path = Path("experiments/results/poisson_evolution.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.dump = pickle.load(f)
    
    ground_truth = data["ground_truth"]
    predictions = data["predictions"]
    steps = data["steps"]
    
    # 2. Setup Plot for Animation (Fixed Aspect Ratio)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    titles = ["Ground Truth", "FNO Prediction", "Absolute Error"]
    
    # Initial state
    im0 = axes[0].imshow(ground_truth, cmap='viridis', aspect='equal', origin='lower')
    axes[0].set_title(titles[0])
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    im1 = axes[1].imshow(predictions[0], cmap='viridis', aspect='equal', origin='lower')
    axes[1].set_title(f"{titles[1]} (Step {steps[0]})")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    error = np.abs(ground_truth - predictions[0])
    im2 = axes[2].imshow(error, cmap='magma', aspect='equal', origin='lower')
    axes[2].set_title(titles[2])
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    def update(frame_idx):
        pred = predictions[frame_idx]
        step = steps[frame_idx]
        err = np.abs(ground_truth - pred)
        
        im1.set_data(pred)
        axes[1].set_title(f"{titles[1]} (Step {step})")
        
        im2.set_data(err)
        # Update colorbar limits for error if needed, but error should decrease
        # im2.set_clim(vmin=0, vmax=np.max(np.abs(ground_truth - predictions[0])))
        
        return [im1, im2]

    # Create Animation
    print("Generating GIF animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(predictions), interval=100, blit=False)
    ani.save(results_dir / "evolution.gif", writer='pillow', fps=10)
    plt.close()
    
    # 3. Final Comparison PNG
    print("Saving final comparison PNG...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    final_pred = predictions[-1]
    final_error = np.abs(ground_truth - final_pred)
    
    im0 = axes[0].imshow(ground_truth, cmap='viridis')
    axes[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(final_pred, cmap='viridis')
    axes[1].set_title(f"FNO Prediction (Step {steps[-1]})")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(final_error, cmap='magma')
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(results_dir / "final_comparison.png")
    plt.close()
    
    # 4. Save Metrics to JSON
    print("Saving metrics to JSON...")
    
    def relative_l2(pred, target):
        return np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)
    
    final_rel_l2 = relative_l2(final_pred, ground_truth)
    final_mse = np.mean((final_pred - ground_truth)**2)
    
    metrics = {
        "final_step": steps[-1],
        "test_sample_mse": float(final_mse),
        "test_sample_rel_l2": float(final_rel_l2),
        "total_training_steps": steps[-1] + 1
    }
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    create_results()
