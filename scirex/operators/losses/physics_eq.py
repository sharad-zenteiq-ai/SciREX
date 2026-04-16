from scirex.operators.losses.math_module_FNO import first_order_derivative, second_order_derivative
import jax.numpy as jnp


def pde_loss_FNO(model,physics, params, v_L):



    """
    Computes the PDE loss based on the provided physics function and its parameters.
    
    Args:
        physics: A function that takes in the solution v and its derivatives, and returns the PDE residual.
        params: Parameters required by the physics function (e.g., coefficients).
        v_L: The output from the second last FNO block (shape: [batch_size, nx, ny, channels]).
        v_Lm1: The output from the last FNO block (shape: [batch_size, nx, ny, channels]).
        """
    if physics is None:
        raise ValueError("Physics system must be provided for PDE loss computation.")
    elif physics == 'Poisson2d':
        # Compute the second-order spatial derivatives (Laplacian)
        d2u = second_order_derivative(params, v_L)
        # Unpack the spatial branches
        d2u_dx2 = d2u[..., 0]  # The pure xx derivative
        d2u_dy2 = d2u[..., 1]  # The pure yy derivative
        return jnp.mean(jnp.square(d2u_dx2 + d2u_dy2))

       