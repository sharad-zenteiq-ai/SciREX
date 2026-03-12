import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno2d import FNO2d

class BicubicFNO(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super(BicubicFNO, self).__init__()
        self.fno = FNO2d(modes1, modes2, width)
        
    def forward(self, x):
        # x: (batch, 16, 16)
        # Bicubic upsampling to 128x128
        x = x.unsqueeze(1)  # (batch, 1, 16, 16)
        x_bicubic = F.interpolate(x, size=(128, 128), mode='bicubic', align_corners=False)
        x_bicubic = x_bicubic.squeeze(1)  # (batch, 128, 128)
        
        # FNO refinement
        x_refined = self.fno(x_bicubic)
        
        # Residual connection
        x_out = x_bicubic + x_refined
        
        return x_out, x_bicubic