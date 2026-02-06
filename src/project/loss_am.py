"""Loss functions for PINN training."""

import jax.numpy as jnp
from jax import grad, vmap

from .config import Config
from .model_am import forward


def data_loss(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]], sensor_data, cfg: Config
) -> jnp.ndarray:
    """MSE loss between predictions and sensor measurements.

    Args:
        nn_params: Network parameters (list of (w, b) tuples)
        sensor_data: Array of [x, y, t, T_measured] (N, 4)
        cfg: Configuration

    Returns:
        Mean squared error
    """
    x, y, t, cx, cy, T_true = (
        sensor_data[:, 0],
        sensor_data[:, 1],
        sensor_data[:, 2],
        sensor_data[:, 3],
        sensor_data[:, 4],
        sensor_data[:, 5],
    )

    T_predicted = forward(nn_params, x, y, t, cx, cy, cfg)
    data_loss_val = jnp.mean((T_predicted - T_true)**2)

    return data_loss_val


def ic_loss(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]],
    ic_points: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """Initial condition loss: T(x, y, 0) = T_outside.

    Args:
        nn_params: Network parameters (list of (w, b) tuples)
        ic_points: Array of [x, y, 0] points (N, 3)
        cfg: Configuration

    Returns:
        Mean squared IC error
    """
    x, y, cx, cy = ic_points[:, 0], ic_points[:, 1], ic_points[:, 3], ic_points[:, 4]

    T_predicted = forward(nn_params, x, y, 0, cx, cy, cfg)
    ic_loss_val = jnp.mean((T_predicted - cfg.T_outside)**2)

    return ic_loss_val


def physics_loss(pinn_params, interior_points, cfg: Config):
    """PDE residual loss at collocation points.

    Args:
        pinn_params: Full pinn_params dict with 'nn', 'log_alpha', 'log_power'
        interior_points: Array of [x, y, t] points (N, 3)
        cfg: Configuration

    Returns:
        Mean squared PDE residual
    """
    x, y, t, cx, cy = interior_points[:, 0], interior_points[:, 1], interior_points[:, 2], interior_points[:, 3], interior_points[:, 4]


    def _pde_residual_scalar(pinn_params, xi, yi, ti, cxi, cyi, cfg):
        """Compute PDE residual at a single point."""

        nn_params = pinn_params['nn']

        alpha = jnp.exp(pinn_params["log_alpha"])
        power = jnp.exp(pinn_params["log_power"])
        
        # Define function for predicted temperatures
        def T_fn(x, y, t):
            return forward(nn_params, x, y, t, cxi, cyi, cfg)

        # Find partial derivatives
        T_t  = grad(T_fn, 2)(xi, yi, ti)
        T_xx = grad(grad(T_fn, 0), 0)(xi, yi, ti)
        T_yy = grad(grad(T_fn, 1), 1)(xi, yi, ti)

        # Heat source
        source = power * cfg.is_source_am(xi, yi, cxi, cyi)

        return T_t - alpha * (T_xx + T_yy) - source
    
    # Vectorize the residuals
    residuals = vmap(
        lambda xi, yi, ti, cxi, cyi: _pde_residual_scalar(
            pinn_params, xi, yi, ti, cxi, cyi, cfg
    ))(x, y, t, cx, cy)

    # Evaluate the physics loss
    physics_loss_val = jnp.mean(residuals**2)

    return physics_loss_val


def bc_loss(pinn_params: dict, bc_points, cfg: Config) -> jnp.ndarray:
    """Robin boundary condition loss: -k * grad(T) . n = h * (T - T_out).

    Args:
        pinn_params: Full pinn_params dict with 'nn', 'log_k', 'log_h' keys
        bc_points: Array of [x, y, t, nx, ny] points (N, 5) where (nx, ny) is outward normal
        cfg: Configuration with k, h, T_outside

    Returns:
        Mean squared BC residual
    """
    x, y, t = bc_points[:, 0], bc_points[:, 1], bc_points[:, 2]
    nx, ny = bc_points[:, 3], bc_points[:, 4]
    cx, cy = bc_points[:, 5], bc_points[:, 6]

    def _bc_residual_scalar(pinn_params, x, y, t, nx, ny, cx, cy, cfg: Config):
        """Compute Robin BC residual: -k * grad(T) . n - h * (T - T_out) = 0.

        Args:
            pinn_params: Full pinn_params dict with 'nn', 'log_k', 'log_h' keys
            x, y, t: Point on boundary (scalars)
            nx, ny: Outward normal components (scalars)
            cfg: Configuration

        Returns:
            BC residual (scalar)
        """

        def T_fn(x, y, t):
            return forward(pinn_params["nn"], x, y, t, cx, cy, cfg)

        # Compute spatial gradients using automatic differentiation
        T_x = grad(T_fn, 0)(x, y, t)
        T_y = grad(T_fn, 1)(x, y, t)

        # Temperature at boundary point
        T = T_fn(x, y, t)

        # Robin BC: -k * (grad T . n) = h * (T - T_out)
        grad_T_dot_n = T_x * nx + T_y * ny
        k = jnp.exp(pinn_params["log_k"])
        h = jnp.exp(pinn_params["log_h"])
        residual = -k * grad_T_dot_n - h * (T - cfg.T_outside)

        return residual

    residuals = vmap(
        lambda xi, yi, ti, nxi, nyi, cxi, cyi: _bc_residual_scalar(
            pinn_params, xi, yi, ti, nxi, nyi, cxi, cyi, cfg
        )
    )(x, y, t, nx, ny, cx, cy)

    bc_loss_val = jnp.mean(residuals**2)

    return bc_loss_val
