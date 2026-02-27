from typing import Tuple, Optional, List
import jax
import jax.numpy as jnp
from flax import linen as nn

def get_wavelet_filters(name: str):
    """
    Returns the decomposition filters (h0, h1) for a given wavelet.
    For reconstruction (g0, g1), they can be derived from the analysis filters.
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
        h1 = jnp.array([-h0[11], h0[10], -h0[9], h0[8], -h0[7], h0[6], 
                        -h0[5], h0[4], -h0[3], h0[2], -h0[1], h0[0]])
    else:
        raise NotImplementedError(
            f"Wavelet {name} not supported. Use 'haar', 'db2', 'db4', or 'db6'."
        )
    return h0, h1

def subband_conv1d(x, kernel, stride=2, padding='SAME'):
    """Grouped 1D convolution for wavelet subbands.
    
    Input x: (batch, length, channels) in NHC format.
    Kernel: 1D filter.
    """
    C = x.shape[-1]
    k = kernel[:, None, None]  # (K, 1, 1)
    k = jnp.tile(k, (1, 1, C))  # (K, 1, C)
    
    out = jax.lax.conv_general_dilated(
        x, k, window_strides=(stride,), padding=padding,
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=C
    )
    return out

def subband_conv2d(x, kernel_2d, stride=2, padding='SAME'):
    """Grouped 2D convolution for wavelet subbands.
    
    Input x: (batch, height, width, channels) in NHWC format.
    Kernel: 2D filter.
    """
    C = x.shape[-1]
    k = kernel_2d[:, :, None, None]  # (H, W, 1, 1)
    k = jnp.tile(k, (1, 1, 1, C))  # (H, W, 1, C)
    
    out = jax.lax.conv_general_dilated(
        x, k, window_strides=(stride, stride), padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C
    )
    return out

def dwt_step_1d(x, h0, h1):
    """Performs 1-level 1D Discrete Wavelet Transform.
    
    Args:
        x: Input signal (batch, length, channels).
        h0: Low-pass decomposition filter.
        h1: High-pass decomposition filter.
        
    Returns:
        cA: Approximation coefficients.
        cD: Detail coefficients.
    """
    cA = subband_conv1d(x, h0, stride=2)
    cD = subband_conv1d(x, h1, stride=2)
    return cA, cD

def dwt_step_2d(x, h0, h1):
    """Performs 1-level 2D Discrete Wavelet Transform.
    
    Args:
        x: Input signal (batch, height, width, channels).
        h0: Low-pass decomposition filter. 
        h1: High-pass decomposition filter.
        
    Returns:
        cLL: Approximation coefficients.
        (cLH, cHL, cHH): Detail coefficients tuple.
    """
    cLL = subband_conv2d(x, jnp.outer(h0, h0), stride=2)
    cLH = subband_conv2d(x, jnp.outer(h0, h1), stride=2)
    cHL = subband_conv2d(x, jnp.outer(h1, h0), stride=2)
    cHH = subband_conv2d(x, jnp.outer(h1, h1), stride=2)
    return cLL, (cLH, cHL, cHH)


class WaveletConv1D(nn.Module):
    """
    1D Wavelet Convolution layer following the reference WNO implementation.
    
    Architecture (from TapasTripura/WNO):
    1. Multi-level DWT decomposition.
    2. Apply learnable weights ONLY to the final-level approximation (cA)
       and the LAST-level detail coefficients (cD). All other detail 
       coefficients are zeroed out.
    3. Inverse DWT to reconstruct the signal.
    
    This matches the reference mul1d operation: einsum("bix,iox->box").
    
    Parameters
    ----------
    in_channels : int
        Input kernel dimension.
    out_channels : int
        Output kernel dimension.
    level : int
        Number of wavelet decomposition levels.
    size : int
        Signal length (used to determine wavelet mode sizes).
    wavelet : str
        Wavelet filter name (e.g. 'db4', 'db6', 'haar').
    """
    in_channels: int
    out_channels: int
    level: int = 1
    size: int = 1024
    wavelet: str = "db4"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, length, channels) — NHC format.
               Note: The reference uses channel-first (NCH), but SciREX uses 
               channel-last. We transpose internally as needed.
        Returns:
            Output tensor of same spatial shape: (batch, length, out_channels).
        """
        h0, h1 = get_wavelet_filters(self.wavelet)
        g0, g1 = h0[::-1], h1[::-1]
        
        # Determine effective level based on super-resolution
        input_size = x.shape[1]
        if input_size > self.size:
            factor = int(jnp.log2(input_size // self.size))
            effective_level = self.level + factor
        elif input_size < self.size:
            factor = int(jnp.log2(self.size // input_size))
            effective_level = self.level - factor
        else:
            effective_level = self.level
        
        # Compute the wavelet mode sizes using the training size
        # to determine learnable weight dimensions
        dummy_len = self.size
        for _ in range(self.level):
            dummy_len = (dummy_len + 1) // 2  # SAME padding output size
        modes1 = dummy_len
        
        # Initialize weight parameters matching reference:
        # weights1: for approximation coefficients (shape: in_ch x out_ch x modes)
        # weights2: for last-level detail coefficients (shape: in_ch x out_ch x modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        init_fn = lambda key, shape: scale * jax.random.uniform(key, shape)
        
        weights1 = self.param("weights1", init_fn, 
                              (self.in_channels, self.out_channels, modes1))
        weights2 = self.param("weights2", init_fn, 
                              (self.in_channels, self.out_channels, modes1))
        
        # 1. Multi-level Decomposition
        details = []
        approx = x
        for i in range(effective_level):
            approx, cD = dwt_step_1d(approx, h0, h1)
            details.append(cD)
        
        # 2. Apply learnable weights
        # Transpose to channel-first for einsum: (B, L, C) -> we use "blc,ico->blo"
        # Reference uses einsum("bix,iox->box") which is the same pattern
        out_ft = jnp.einsum("blc,col->blo", approx, 
                           weights1[:, :, :approx.shape[1]].transpose(1, 2, 0))
        
        # Actually, match reference exactly: einsum("bix,iox->box")
        # x is (B, C, L) in reference. Our x is (B, L, C).
        # So we do einsum("bli,iol->bol") to match semantics
        out_ft = jnp.einsum("bli,iol->bol", approx, 
                           weights1[:, :, :approx.shape[1]])
        
        # Weight the last-level detail coefficients only
        out_coeff = [jnp.zeros_like(cD) for cD in details]
        if len(details) > 0:
            last_detail = details[-1]
            out_coeff[-1] = jnp.einsum("bli,iol->bol", last_detail,
                                       weights2[:, :, :last_detail.shape[1]])
        
        # 3. Reconstruct via IDWT
        def upsample1d(tensor):
            B, L, C = tensor.shape
            out = jnp.zeros((B, L * 2, C))
            return out.at[:, ::2, :].set(tensor)
        
        res = out_ft
        for i in reversed(range(effective_level)):
            zA = subband_conv1d(upsample1d(res), g0, stride=1)
            zD = subband_conv1d(upsample1d(out_coeff[i]), g1, stride=1)
            res = zA + zD
            
        return res


class WaveletConv2D(nn.Module):
    """
    2D Wavelet Convolution layer following the reference WNO implementation.
    
    Architecture:
    1. Multi-level 2D DWT decomposition.
    2. Apply learnable weights ONLY to:
       - The final-level approximation (LL) coefficients.
       - The LAST-level detail coefficients (LH, HL, HH).
       All other detail coefficients are zeroed out.
    3. Inverse 2D DWT to reconstruct the signal.
    
    This matches the reference mul2d operation: einsum("bixy,ioxy->boxy").
    
    Parameters
    ----------
    in_channels : int
        Input kernel dimension.
    out_channels : int
        Output kernel dimension.
    level : int
        Number of wavelet decomposition levels.
    size : list of int
        Signal dimensions [height, width] for determining wavelet mode sizes.
    wavelet : str
        Wavelet filter name (e.g. 'db4', 'db6', 'haar').
    """
    in_channels: int
    out_channels: int
    level: int = 1
    size: Tuple[int, int] = (64, 64)
    wavelet: str = "db4"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, height, width, channels) — NHWC format.
        Returns:
            Output tensor: (batch, height, width, out_channels).
        """
        # 0. Handle Padding for non-power-of-2 dimensions
        orig_h, orig_w = x.shape[1:3]
        target_h = ((orig_h - 1) // (2**self.level) + 1) * (2**self.level)
        target_w = ((orig_w - 1) // (2**self.level) + 1) * (2**self.level)
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        
        if pad_h > 0 or pad_w > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), 
                       mode='symmetric')

        h0, h1 = get_wavelet_filters(self.wavelet)
        g0, g1 = h0[::-1], h1[::-1]
        
        # Determine effective level based on super-resolution
        input_h = x.shape[1]
        size_h = self.size[0] if isinstance(self.size, (list, tuple)) else self.size
        size_w = self.size[1] if isinstance(self.size, (list, tuple)) else self.size
        
        # Adjust level for super-resolution (matching reference)
        if input_h > size_h:
            factor = int(jnp.log2(input_h // size_h))
            effective_level = self.level + factor
        elif input_h < size_h:
            factor = int(jnp.log2(size_h // input_h))
            effective_level = self.level - factor
        else:
            effective_level = self.level
        
        # Compute wavelet mode sizes from training size
        dummy_h, dummy_w = size_h, size_w
        for _ in range(self.level):
            dummy_h = (dummy_h + 1) // 2
            dummy_w = (dummy_w + 1) // 2
        modes1 = dummy_h
        modes2 = dummy_w
        
        # Initialize weight parameters matching reference:
        # weights1: LL (approximation), weights2: LH, weights3: HL, weights4: HH
        # Shape: (in_channels, out_channels, modes_h, modes_w)
        scale = 1.0 / (self.in_channels * self.out_channels)
        init_fn = lambda key, shape: scale * jax.random.uniform(key, shape)
        
        weights1 = self.param("weights1", init_fn, 
                              (self.in_channels, self.out_channels, modes1, modes2))
        weights2 = self.param("weights2", init_fn, 
                              (self.in_channels, self.out_channels, modes1, modes2))
        weights3 = self.param("weights3", init_fn, 
                              (self.in_channels, self.out_channels, modes1, modes2))
        weights4 = self.param("weights4", init_fn, 
                              (self.in_channels, self.out_channels, modes1, modes2))

        # 1. Multi-level Decomposition
        approx = x
        details = []
        for _ in range(effective_level):
            approx, (lh, hl, hh) = dwt_step_2d(approx, h0, h1)
            details.append((lh, hl, hh))

        # 2. Apply learnable weights — ONLY to last-level coefficients
        # Reference einsum: "bixy,ioxy->boxy" 
        # In NHWC format (B, H, W, C), this becomes: "bhwi,iohw->bhwo"
        ah, aw = approx.shape[1], approx.shape[2]
        out_ft = jnp.einsum("bhwi,iohw->bhwo", approx,
                           weights1[:, :, :ah, :aw])
        
        # Zero out all detail coefficients, then weight only the last level
        out_coeff = [(jnp.zeros_like(lh), jnp.zeros_like(hl), jnp.zeros_like(hh)) 
                     for (lh, hl, hh) in details]
        
        if len(details) > 0:
            last_lh, last_hl, last_hh = details[-1]
            dh, dw = last_lh.shape[1], last_lh.shape[2]
            
            out_lh = jnp.einsum("bhwi,iohw->bhwo", last_lh,
                               weights2[:, :, :dh, :dw])
            out_hl = jnp.einsum("bhwi,iohw->bhwo", last_hl,
                               weights3[:, :, :dh, :dw])
            out_hh = jnp.einsum("bhwi,iohw->bhwo", last_hh,
                               weights4[:, :, :dh, :dw])
            out_coeff[-1] = (out_lh, out_hl, out_hh)

        # 3. Multi-level Reconstruction via IDWT
        def upsample2d(tensor):
            B, H, W, C = tensor.shape
            out = jnp.zeros((B, H*2, W*2, C))
            return out.at[:, ::2, ::2, :].set(tensor)

        res = out_ft
        for i in reversed(range(effective_level)):
            yLH, yHL, yHH = out_coeff[i]
            
            zLL = subband_conv2d(upsample2d(res), jnp.outer(g0, g0), stride=1)
            zLH = subband_conv2d(upsample2d(yLH), jnp.outer(g0, g1), stride=1)
            zHL = subband_conv2d(upsample2d(yHL), jnp.outer(g1, g0), stride=1)
            zHH = subband_conv2d(upsample2d(yHH), jnp.outer(g1, g1), stride=1)
            res = zLL + zLH + zHL + zHH
            
        # 4. Crop back to original dimensions
        if pad_h > 0 or pad_w > 0:
            res = res[:, :orig_h, :orig_w, :]
            
        return res
