"""
This file is used to evaluate trained models on the Navier-Stokes super-resolution task.
It computes metrics like MSE, MAE, RMSE and visualizes predictions from different models

"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from ..models.fno.bicubic_fno import BicubicFNO 
# Define models to evaluate
# model_vanilla = BicubicFNO(modes1=12, modes2=12, width=32)
# model_physics = BicubicFNO(modes1=12, modes2=12, width=32)  


def evaluate_models(models, model_names, val_loader, device='cuda'):
    """
    Evaluate multiple models and compute metrics
    """
    results = {}
    
    for model, name in zip(models, model_names):
        if model is not None:
            model.eval()
        
        mse_list = []
        mae_list = []
        
        all_pred = []
        all_gt = []
        all_bicubic = []
        all_lr = []
        
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                
                if name == 'Bicubic' or model is None:
                    # Baseline bicubic interpolation
                    hr_pred = F.interpolate(lr.unsqueeze(1), size=(128, 128), 
                                          mode='bicubic', align_corners=False).squeeze(1)
                    hr_bicubic = hr_pred
                else:
                    # Model prediction
                    hr_pred, hr_bicubic = model(lr)
                
                mse = F.mse_loss(hr_pred, hr).item()
                mae = F.l1_loss(hr_pred, hr).item()
                
                mse_list.append(mse)
                mae_list.append(mae)
                
                all_pred.append(hr_pred.cpu())
                all_gt.append(hr.cpu())
                all_bicubic.append(hr_bicubic.cpu())
                all_lr.append(lr.cpu())
        
        results[name] = {
            'mse': np.mean(mse_list),
            'mae': np.mean(mae_list),
            'rmse': np.sqrt(np.mean(mse_list)),
            'predictions': torch.cat(all_pred, dim=0),
            'ground_truth': torch.cat(all_gt, dim=0),
            'bicubic': torch.cat(all_bicubic, dim=0),
            'lr_inputs': torch.cat(all_lr, dim=0)
        }
        
        print(f"Evaluated {name}: MSE={results[name]['mse']:.6f}, MAE={results[name]['mae']:.6f}")
    
    return results


# Load best models
model_vanilla.load_state_dict(torch.load('bicubic_fno_vanilla_best.pth')['model_state_dict'])
model_physics.load_state_dict(torch.load('bicubic_fno_physics_best.pth')['model_state_dict'])

# Evaluate all models
models = [None, model_vanilla, model_physics]
model_names = ['Bicubic', 'Bicubic+FNO', 'Bicubic+FNO+Physics']

results = evaluate_models(models, model_names, val_loader, device)

# Print metrics
for name in model_names:
    print(f"\n{name}:")
    print(f"  MSE:  {results[name]['mse']:.6f}")
    print(f"  MAE:  {results[name]['mae']:.6f}")
    print(f"  RMSE: {results[name]['rmse']:.6f}")

# Visualize comparison
def visualize_comparison(results, sample_idx=0):
    """
    Visualize predictions from all models
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    lr_input = results['Bicubic']['lr_inputs'][sample_idx].numpy()
    gt = results['Bicubic']['ground_truth'][sample_idx].numpy()
    bicubic = results['Bicubic']['predictions'][sample_idx].numpy()
    fno = results['Bicubic+FNO']['predictions'][sample_idx].numpy()
    physics = results['Bicubic+FNO+Physics']['predictions'][sample_idx].numpy()
    
    vmin = min(gt.min(), bicubic.min(), fno.min(), physics.min())
    vmax = max(gt.max(), bicubic.max(), fno.max(), physics.max())
    
    # First row: Predictions
    im0 = axes[0, 0].imshow(lr_input, cmap='RdBu_r', extent=[0, 1, 0, 1], origin='lower')
    axes[0, 0].set_title('LR Input (16×16)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(bicubic, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                            extent=[0, 1, 0, 1], origin='lower')
    axes[0, 1].set_title('Bicubic', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(fno, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                            extent=[0, 1, 0, 1], origin='lower')
    axes[0, 2].set_title('Bicubic+FNO', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(physics, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                            extent=[0, 1, 0, 1], origin='lower')
    axes[0, 3].set_title('Bicubic+FNO+Physics', fontsize=14, fontweight='bold')
    axes[0, 3].set_xlabel('x')
    axes[0, 3].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Second row: Ground truth and errors
    im4 = axes[1, 0].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                            extent=[0, 1, 0, 1], origin='lower')
    axes[1, 0].set_title('Ground Truth HR', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    error_max = max(np.abs(gt - bicubic).max(), np.abs(gt - fno).max(), 
                    np.abs(gt - physics).max())
    
    im5 = axes[1, 1].imshow(np.abs(gt - bicubic), cmap='hot', vmin=0, vmax=error_max,
                            extent=[0, 1, 0, 1], origin='lower')
    axes[1, 1].set_title(f'Bicubic Error\nMAE: {np.abs(gt - bicubic).mean():.4f}', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(np.abs(gt - fno), cmap='hot', vmin=0, vmax=error_max,
                            extent=[0, 1, 0, 1], origin='lower')
    axes[1, 2].set_title(f'FNO Error\nMAE: {np.abs(gt - fno).mean():.4f}', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    im7 = axes[1, 3].imshow(np.abs(gt - physics), cmap='hot', vmin=0, vmax=error_max,
                            extent=[0, 1, 0, 1], origin='lower')
    axes[1, 3].set_title(f'Physics Error\nMAE: {np.abs(gt - physics).mean():.4f}', 
                         fontsize=14, fontweight='bold')
    axes[1, 3].set_xlabel('x')
    axes[1, 3].set_ylabel('y')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f'comparison_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Visualize multiple samples
for i in range(min(3, len(results['Bicubic']['predictions']))):
    print(f"Visualizing sample {i}...")
    visualize_comparison(results, sample_idx=i)