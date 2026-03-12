import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
lr_data = np.load('ns_lr_16x16.npy')  # (N, 16, 16, T)
hr_data = np.load('ns_hr_128x128.npy')  # (N, 128, 128, T)

print(f"LR data shape: {lr_data.shape}")
print(f"HR data shape: {hr_data.shape}")

# Dataset class
class NSDataset(Dataset):
    def __init__(self, lr_data, hr_data, transform=None):
        """
        Dataset for Navier-Stokes super-resolution
        lr_data: (N, 16, 16, T)
        hr_data: (N, 128, 128, T)
        """
        self.lr_data = torch.from_numpy(lr_data).float()
        self.hr_data = torch.from_numpy(hr_data).float()
        self.transform = transform
        
    def __len__(self):
        N, _, _, T = self.lr_data.shape
        return N * T
    
    def __getitem__(self, idx):
        N, _, _, T = self.lr_data.shape
        sample_idx = idx // T
        time_idx = idx % T
        
        lr = self.lr_data[sample_idx, :, :, time_idx]
        hr = self.hr_data[sample_idx, :, :, time_idx]
        
        if self.transform:
            lr, hr = self.transform(lr, hr)
        
        return lr, hr

# Create train/val split
N, _, _, T = lr_data.shape
train_size = int(0.8 * N)

train_dataset = NSDataset(lr_data[:train_size], hr_data[:train_size])
val_dataset = NSDataset(lr_data[train_size:], hr_data[train_size:])

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)