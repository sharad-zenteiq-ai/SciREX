from typing import Tuple, Optional, List
import jax
import jax.numpy as jnp
from flax import linen as nn

def get_wavelet_filters(name: str):
    """
    Returns the decomposition filters (h0, h1) for a given wavelet.
    For reconstruction (g0, g1), they can be derived.
    """
    if name == "haar":
        h0 = jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)
        h1 = jnp.array([1.0, -1.0]) / jnp.sqrt(2.0)
    elif name == "db2":
        # Daubechies 2
        h0 = jnp.array([0.48296291, 0.8365163, 0.22414387, -0.12940952])
        h1 = jnp.array([-h0[3], h0[2], -h0[1], h0[0]])
    elif name == "db4":
        h0 = jnp.array([0.0322231, -0.01260397, -0.09921954, 0.2978578, 
                       0.80373875, 0.49761867, -0.02963553, -0.07576571])
        h1 = jnp.array([-h0[7], h0[6], -h0[5], h0[4], -h0[3], h0[2], -h0[1], h0[0]])
    elif name == "db6":
        h0 = jnp.array([0.03522629, -0.08544127, -0.13501102, 0.4598775, 
                       0.80689151, 0.33266548, -0.10144531, -0.04408825,
                       0.0223997, 0.04724845, -0.00238294, -0.01235692])
        h1 = jnp.array([-h0[11], h0[10], -h0[9], h0[8], -h0[7], h0[6], -h0[5], h0[4], -h0[3], h0[2], -h0[1], h0[0]])
    else:
        raise NotImplementedError(f"Wavelet {name} not supported. Use 'haar', 'db2', 'db4', or 'db6'.")
    return h0, h1

def subband_conv1d(x, kernel, stride=2, padding='SAME'):
    """Grouped 1D convolution for wavelet subbands."""
    C = x.shape[-1]
    k = kernel[:, None, None] # (K, 1, 1)
    k = jnp.tile(k, (1, 1, C)) # (K, 1, C)
    
    out = jax.lax.conv_general_dilated(
        x, k, window_strides=(stride,), padding=padding,
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=C
    )
    return out

def subband_conv2d(x, kernel_2d, stride=2, padding='SAME'):
    """Grouped 2D convolution for wavelet subbands."""
    C = x.shape[-1]
    k = kernel_2d[:, :, None, None] # (H, W, 1, 1)
    k = jnp.tile(k, (1, 1, 1, C)) # (H, W, 1, C)
    
    out = jax.lax.conv_general_dilated(
        x, k, window_strides=(stride, stride), padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C
    )
    return out

def dwt_step_2d(x, h0, h1):
    """Performs 1-level 2D Discrete Wavelet Transform."""
    cLL = subband_conv2d(x, jnp.outer(h0, h0), stride=2)
    cLH = subband_conv2d(x, jnp.outer(h0, h1), stride=2)
    cHL = subband_conv2d(x, jnp.outer(h1, h0), stride=2)
    cHH = subband_conv2d(x, jnp.outer(h1, h1), stride=2)
    return cLL, (cLH, cHL, cHH)

class WaveletConv1D(nn.Module):
    """
    Standard 1D Wavelet Convolution layer with multi-level support.
    """
    in_channels: int
    out_channels: int
    wavelet: str = "haar"
    levels: int = 1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h0, h1 = get_wavelet_filters(self.wavelet)
        # Synthesis filters for orthogonal wavelets are time-reversed
        g0, g1 = h0[::-1], h1[::-1]
        
        init_fn = nn.initializers.lecun_normal()
        
        # 1. Multi-level Decomposition
        details = []
        approx = x
        for i in range(self.levels):
            cA = subband_conv1d(approx, h0, stride=2)
            cD = subband_conv1d(approx, h1, stride=2)
            
            # Weight detail coefficients
            wD = self.param(f"wD_l{i}", init_fn, (self.in_channels, self.out_channels))
            yD = jnp.matmul(cD, wD)
            details.append(yD)
            approx = cA
            
        # Weight final approximation
        wA = self.param(f"wA", init_fn, (self.in_channels, self.out_channels))
        yA = jnp.matmul(approx, wA)
        
        # 2. Reconstruct
        def upsample1d(tensor):
            B, L, C = tensor.shape
            out = jnp.zeros((B, L * 2, C))
            return out.at[:, ::2, :].set(tensor)

        res = yA
        for i in reversed(range(self.levels)):
            zA = subband_conv1d(upsample1d(res), g0, stride=1)
            zD = subband_conv1d(upsample1d(details[i]), g1, stride=1)
            res = zA + zD
            
        return res

class WaveletConv2D(nn.Module):
    """
    Standard 2D Wavelet Convolution layer with multi-level support.
    """
    in_channels: int
    out_channels: int
    wavelet: str = "haar"
    levels: int = 1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 0. Handle Padding for non-power-of-2 dimensions
        orig_h, orig_w = x.shape[1:3]
        target_h = ((orig_h - 1) // (2**self.levels) + 1) * (2**self.levels)
        target_w = ((orig_w - 1) // (2**self.levels) + 1) * (2**self.levels)
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        
        if pad_h > 0 or pad_w > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')

        h0, h1 = get_wavelet_filters(self.wavelet)
        g0, g1 = h0[::-1], h1[::-1]
        
        # 1. Multi-level Decomposition
        res = x
        details = []
        for _ in range(self.levels):
            res, (lh, hl, hh) = dwt_step_2d(res, h0, h1)
            details.append((lh, hl, hh))

        # 2. Linear Transform in Wavelet Space (Hybrid Strategy)
        b, h, w, c = res.shape
        init_fn = nn.initializers.lecun_normal()
        
        # Always use mode-wise weighting for the coarsest Approximation (LL)
        # This is where the most 'global' structural information resides
        if h <= 64: # Increased threshold for more learnable spatial modes
            wLL = self.param("wLL", init_fn, (c, self.out_channels, h, w))
            yA = jnp.einsum("bhwi,iowh->bhwo", res, wLL)
        else: # Standard conv for extremely high res approximations
            yA = nn.Conv(self.out_channels, (1, 1), padding='SAME', kernel_init=init_fn)(res)
        
        # Weighting for all Details (LH, HL, HH at every level)
        transformed_details = []
        for i in range(self.levels):
            lh, hl, hh = details[i]
            li_h, li_w = lh.shape[1:3]
            
            # Hybrid: Mode-wise for most levels, 1x1 Conv for very fine levels
            # Threshold 64 allows 128x128 res models to use mode-wise weights at most levels
            if li_h <= 64: 
                wLH = self.param(f"wLH_l{i}", init_fn, (c, self.out_channels, li_h, li_w))
                wHL = self.param(f"wHL_l{i}", init_fn, (c, self.out_channels, li_h, li_w))
                wHH = self.param(f"wHH_l{i}", init_fn, (c, self.out_channels, li_h, li_w))
                
                yLH = jnp.einsum("bhwi,iowh->bhwo", lh, wLH)
                yHL = jnp.einsum("bhwi,iowh->bhwo", hl, wHL)
                yHH = jnp.einsum("bhwi,iowh->bhwo", hh, wHH)
            else:
                # Use standard convolution for finer levels to save memory/params
                yLH = nn.Conv(self.out_channels, (1, 1), padding='SAME', name=f"convLH_l{i}")(lh)
                yHL = nn.Conv(self.out_channels, (1, 1), padding='SAME', name=f"convHL_l{i}")(hl)
                yHH = nn.Conv(self.out_channels, (1, 1), padding='SAME', name=f"convHH_l{i}")(hh)
            
            transformed_details.append((yLH, yHL, yHH))

        # 3. Multi-level Reconstruction
        def upsample2d(tensor):
            B, H, W, C = tensor.shape
            out = jnp.zeros((B, H*2, W*2, C))
            return out.at[:, ::2, ::2, :].set(tensor)

        res = yA
        for i in reversed(range(self.levels)):
            yLH, yHL, yHH = transformed_details[i]
            
            zLL = subband_conv2d(upsample2d(res), jnp.outer(g0, g0), stride=1)
            zLH = subband_conv2d(upsample2d(yLH), jnp.outer(g0, g1), stride=1)
            zHL = subband_conv2d(upsample2d(yHL), jnp.outer(g1, g0), stride=1)
            zHH = subband_conv2d(upsample2d(yHH), jnp.outer(g1, g1), stride=1)
            res = zLL + zLH + zHL + zHH
            
        # 4. Crop back to original dimensions
        if pad_h > 0 or pad_w > 0:
            res = res[:, :orig_h, :orig_w, :]
            
        return res
