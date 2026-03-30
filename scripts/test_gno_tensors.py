import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import jax
import jax.numpy as jnp
import numpy as np
from scirex.operators.layers.gno_block import GNOBlock

def test_gno_tensor():
    input_file = "gno_input_tensor/input_1774854686188.pt"
    output_file = "gno_output_tensor/output_1774854686188.pt"
    
    print(f"Loading {input_file}...")
    in_dict = torch.load(input_file, map_location='cpu')
    expected_out = torch.load(output_file, map_location='cpu')
    
    y_pt = in_dict['y']
    x_pt = in_dict['x']
    f_y_pt = in_dict['f_y']
    
    y = jnp.array(y_pt.numpy())
    x = jnp.array(x_pt.numpy())
    f_y = jnp.array(f_y_pt.numpy()) if f_y_pt is not None else None
    
    in_channels = f_y.shape[-1] if f_y is not None else 0
    # Expected out shape is [32768, 3] -> out_channels = 3
    out_channels = expected_out.shape[-1]
    
    # Initialize JAX GNOBlock
    # Using typical test parameters.
    model = GNOBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=3,
        radius=0.033, 
    )
    
    rng = jax.random.PRNGKey(0)
    
    print("\n--- Initializing JAX GNOBlock ---")
    variables = model.init(rng, y=y, x=x, f_y=f_y)
    
    print("--- Running Forward Pass ---")
    out = model.apply(variables, y=y, x=x, f_y=f_y)
    
    out_np = np.array(out)
    expected_np = expected_out.detach().cpu().numpy()
    
    print("\n================ Results ================")
    print(f"Input y (Source coords):     {y.shape}")
    print(f"Input x (Query coords):      {x.shape}")
    print(f"Input f_y (Features):        {f_y.shape if f_y is not None else None}")
    print(f"-----------------------------------------")
    print(f"GNOBlock Output Shape:       {out_np.shape}")
    print(f"Expected Output Shape:       {expected_np.shape}")
    
    if out_np.shape == expected_np.shape:
        print("Shapes match perfectly!")
    else:
        print("Shapes mismatch!")
        
    mse = np.mean((out_np - expected_np) ** 2)
    mae = np.mean(np.abs(out_np - expected_np))
    rel_l2 = np.linalg.norm(out_np - expected_np) / np.linalg.norm(expected_np)
    
    print(f"\nMean Squared Error (MSE):    {mse:.6f}")
    print(f"Mean Absolute Error (MAE):   {mae:.6f}")
    print(f"Relative L2 Error:           {rel_l2:.6f}")
    print("\n(Note: Non-zero MSE is expected because the JAX GNOBlock weights are randomly initialized.)")
    print("=========================================\n")

if __name__ == "__main__":
    test_gno_tensor()
