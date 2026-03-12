"""
PhysicsLoss: Implements physics-informed loss for Navier-Stokes equations in 2D using vorticity formulation.

"""
import torch
import torch.nn as nn
import torch.fft
import math

class PhysicsLoss(nn.Module):
    def __init__(self, visc=1e-3):
        super(PhysicsLoss, self).__init__()
        self.visc = visc
        
    def compute_derivatives(self, w, h=1.0/128):
        """
        Compute spatial derivatives using finite differences
        w: (batch, H, W) vorticity field
        h: grid spacing
        """
        # Periodic boundary conditions
        # First derivative (central difference)
        w_x = (torch.roll(w, -1, dims=2) - torch.roll(w, 1, dims=2)) / (2 * h)
        w_y = (torch.roll(w, -1, dims=1) - torch.roll(w, 1, dims=1)) / (2 * h)
        
        # Second derivative (Laplacian)
        w_xx = (torch.roll(w, -1, dims=2) - 2*w + torch.roll(w, 1, dims=2)) / (h**2)
        w_yy = (torch.roll(w, -1, dims=1) - 2*w + torch.roll(w, 1, dims=1)) / (h**2)
        laplacian_w = w_xx + w_yy
        
        return w_x, w_y, laplacian_w
    
    def compute_stream_function(self, w):
        """
        Solve Poisson equation: ∇²ψ = w using FFT
        w: (batch, H, W)
        """
        batch_size, H, W = w.shape
        
        # Wavenumbers
        k_y = torch.cat([torch.arange(0, H//2, device=w.device), 
                        torch.arange(-H//2, 0, device=w.device)], 0).repeat(W, 1).T
        k_x = torch.cat([torch.arange(0, W//2, device=w.device), 
                        torch.arange(-W//2, 0, device=w.device)], 0).repeat(H, 1)
        
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
        lap[0, 0] = 1.0  # Avoid division by zero
        
        # Solve in Fourier space
        w_h = torch.fft.fftn(w, dim=(-2, -1))
        psi_h = w_h / lap.unsqueeze(0)
        psi = torch.fft.ifftn(psi_h, dim=(-2, -1)).real
        
        return psi
    
    def forward(self, w_pred):
        """
        Compute physics-informed loss
        w_pred: (batch, H, W) predicted vorticity
        """
        h = 1.0 / w_pred.shape[-1]
        
        # Compute derivatives
        w_x, w_y, laplacian_w = self.compute_derivatives(w_pred, h)
        
        # Compute stream function and velocity
        psi = self.compute_stream_function(w_pred)
        psi_x, psi_y, _ = self.compute_derivatives(psi, h)
        
        # Velocity field: u = ∂ψ/∂y, v = -∂ψ/∂x
        u = psi_y
        v = -psi_x
        
        # Vorticity equation residual (without time derivative and forcing)
        # ∂w/∂t + u·∇w = ν∇²w + f
        # We check: u·∇w - ν∇²w should be smooth
        advection = u * w_x + v * w_y
        diffusion = self.visc * laplacian_w
        residual = advection - diffusion
        
        # Physics loss: residual should be smooth (small gradients)
        residual_x = (torch.roll(residual, -1, dims=2) - torch.roll(residual, 1, dims=2)) / (2 * h)
        residual_y = (torch.roll(residual, -1, dims=1) - torch.roll(residual, 1, dims=1)) / (2 * h)
        
        physics_loss = torch.mean(residual_x**2 + residual_y**2)
        
        return physics_loss


def compute_energy_spectrum(w):
    """
    Compute energy spectrum from vorticity field
    w: (H, W) vorticity field
    Returns: k (wavenumbers), E(k) (energy spectrum)
    """
    H, W = w.shape
    
    # Compute velocity from vorticity using stream function
    k_y = torch.cat([torch.arange(0, H//2, device=w.device), 
                    torch.arange(-H//2, 0, device=w.device)], 0).repeat(W, 1).T
    k_x = torch.cat([torch.arange(0, W//2, device=w.device), 
                    torch.arange(-W//2, 0, device=w.device)], 0).repeat(H, 1)
    
    lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    
    w_h = torch.fft.fftn(w)
    psi_h = w_h / lap
    
    # Velocity in Fourier space
    u_h = 2j * math.pi * k_y * psi_h
    v_h = -2j * math.pi * k_x * psi_h
    
    # Energy spectrum
    E = 0.5 * (torch.abs(u_h)**2 + torch.abs(v_h)**2)
    
    # Radial average
    k_mag = torch.sqrt(k_x**2 + k_y**2).cpu().numpy()
    E_np = E.cpu().numpy()
    
    k_max = int(np.max(k_mag))
    k_bins = np.arange(0, k_max + 1)
    E_k = np.zeros(len(k_bins) - 1)
    
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if np.sum(mask) > 0:
            E_k[i] = np.mean(E_np[mask])
    
    k = (k_bins[:-1] + k_bins[1:]) / 2
    
    return k[1:], E_k[1:]  # Skip zero wavenumber