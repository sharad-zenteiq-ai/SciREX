"""
Training utilities for SciML models on Navier-Stokes super-resolution task.
Includes training loop with physics-informed loss.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..models.fno.bicubic_fno import BicubicFNO
from ..utils.physics_loss import PhysicsLoss

def train_epoch(model, train_loader, optimizer, criterion, physics_loss_fn, 
                physics_weight=0.0, device='cuda'):
    model.train()
    total_loss = 0
    total_mse = 0
    total_physics = 0
    
    for lr, hr in tqdm(train_loader, desc='Training'):
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        hr_pred, hr_bicubic = model(lr)
        
        # MSE loss
        mse_loss = criterion(hr_pred, hr)
        
        # Physics loss
        if physics_weight > 0:
            physics_loss = physics_loss_fn(hr_pred)
            loss = mse_loss + physics_weight * physics_loss
        else:
            physics_loss = torch.tensor(0.0)
            loss = mse_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_physics += physics_loss.item()
    
    n_batches = len(train_loader)
    return total_loss / n_batches, total_mse / n_batches, total_physics / n_batches


def validate(model, val_loader, criterion, physics_loss_fn, 
             physics_weight=0.0, device='cuda'):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_physics = 0
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            hr_pred, hr_bicubic = model(lr)
            
            # MSE loss
            mse_loss = criterion(hr_pred, hr)
            
            # Physics loss
            if physics_weight > 0:
                physics_loss = physics_loss_fn(hr_pred)
                loss = mse_loss + physics_weight * physics_loss
            else:
                physics_loss = torch.tensor(0.0)
                loss = mse_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_physics += physics_loss.item()
    
    n_batches = len(val_loader)
    return total_loss / n_batches, total_mse / n_batches, total_physics / n_batches


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, 
                physics_weight=0.0, device='cuda', model_name='model'):
    """
    Train the model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=5)
    criterion = nn.MSELoss()
    physics_loss_fn = PhysicsLoss(visc=1e-3)
    
    train_losses = []
    val_losses = []
    train_mse = []
    val_mse = []
    train_physics = []
    val_physics = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_mse_loss, train_phys_loss = train_epoch(
            model, train_loader, optimizer, criterion, physics_loss_fn, 
            physics_weight, device
        )
        
        # Validate
        val_loss, val_mse_loss, val_phys_loss = validate(
            model, val_loader, criterion, physics_loss_fn, 
            physics_weight, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mse.append(train_mse_loss)
        val_mse.append(val_mse_loss)
        train_physics.append(train_phys_loss)
        val_physics.append(val_phys_loss)
        
        print(f"Train Loss: {train_loss:.6f} (MSE: {train_mse_loss:.6f}, Physics: {train_phys_loss:.6f})")
        print(f"Val Loss: {val_loss:.6f} (MSE: {val_mse_loss:.6f}, Physics: {val_phys_loss:.6f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{model_name}_best.pth')
            print(f"Saved best model with val_loss: {val_loss:.6f}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_mse, label='Train')
    axes[1].plot(val_mse, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('MSE Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if physics_weight > 0:
        axes[2].plot(train_physics, label='Train')
        axes[2].plot(val_physics, label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Physics Loss')
        axes[2].set_title('Physics Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_physics': train_physics,
        'val_physics': val_physics
    }

# how to train:
# model_vanilla = BicubicFNO(modes1=12, modes2=12, width=32)
# history_vanilla = train_model(
#     model_vanilla, train_loader, val_loader, 
#     epochs=50, lr=1e-3, physics_weight=0.0, 
#     device=device, model_name='bicubic_fno_vanilla'
# )

# Train Bicubic FNO with physics constraints
# model_physics = BicubicFNO(modes1=12, modes2=12, width=32)
# history_physics = train_model(
#     model_physics, train_loader, val_loader, 
#     epochs=50, lr=1e-3, physics_weight=0.1, 
#     device=device, model_name='bicubic_fno_physics'
# )