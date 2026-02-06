"""Finite Difference Method solver for the 2D heat equation."""

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array, lax

from .config import Config


@jax.jit
def conjugate_method(A: Array, b: Array, x0: Array, max_iter=int, tol=1e-8) -> Array:
    """Use a conjugate method to solve the linear system."""

    # Convert to JAX arrays
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    x0 = jnp.asarray(x0).flatten()

    r0 = b - A @ x0  # First residual
    p0 = r0  # First conjugate direction

    # Return False if tolerance is reached
    def tol_cond(state):
        x, r, p, k = state

        return (jnp.linalg.norm(r) > tol) & (k < max_iter)

    # Each step in the while loop
    def cg_step(state):
        x, r, p, k = state

        alpha = jnp.dot(r, r) / jnp.dot(p, (A @ p))

        x_new = x + alpha * p  # Next vector "closer" to solution
        r_new = r - alpha * (A @ p)  # Next residual

        beta = jnp.dot(r_new, r_new) / jnp.dot(r, r)
        p_new = r_new + beta * p  # Next conjugate vector

        return (x_new, r_new, p_new, k + 1)


    # Initial state
    state0 = (x0, r0, p0, 0)

    # Final state
    x_final, r_final, p_final, k = lax.while_loop(tol_cond, cg_step, state0)
    
    return x_final


def solve_heat_equation(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2D heat equation using implicit Euler.

    Args:
        cfg: Configuration object

    Returns:
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        t: time points (nt,)
        T: temperature solution (nt, nx, ny)
    """
    # Create grids
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initialize temperature vector
    T = np.zeros((cfg.nt, cfg.nx, cfg.ny))

    # Fill first time step with T_outside
    T[0, :, :] = cfg.T_outside

    for i in range(0, cfg.nt-1):

        T_curr = T[i, :, :]
        t_next = t[i+1]

        # Matrix A in the linear system
        A = _build_matrix(cfg, dx, dy, dt)

        # Right-hand side
        b = _build_rhs(cfg, T_curr, X, Y, dx, dy, dt, t_next)

        # Update array T
        max_iter = cfg.nx * cfg.ny
        T[i+1, :, :] = conjugate_method(A, b, T_curr, max_iter).reshape(cfg.nx, cfg.ny)

    return x, y, t, T


def _build_matrix(cfg: Config, dx: float, dy: float, dt: float) -> np.ndarray:
    """Build the implicit Euler system matrix."""
    n = cfg.nx * cfg.ny
    A = np.zeros((n, n))

    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    def idx(i, j):
        return i * cfg.ny + j

    I, J = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")

    # Boundary masks
    left = I == 0
    right = I == cfg.nx - 1
    bottom = J == 0
    top = J == cfg.ny - 1

    # Diagonal entries
    diag = np.full((cfg.nx, cfg.ny), 1 + 2 * rx + 2 * ry)
    diag[left | right] -= rx
    diag[bottom | top] -= ry
    diag[left | right] += rx * cfg.h * dx / cfg.k
    diag[bottom | top] += ry * cfg.h * dy / cfg.k

    p = idx(I, J)
    A[p, p] = diag

    # Off-diagonals
    mask = ~left
    A[idx(I[mask], J[mask]), idx(I[mask] - 1, J[mask])] = -rx

    mask = ~right
    A[idx(I[mask], J[mask]), idx(I[mask] + 1, J[mask])] = -rx

    mask = ~bottom
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] - 1)] = -ry

    mask = ~top
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] + 1)] = -ry

    return A
 

def _build_rhs(cfg: Config, T_curr, X, Y, dx, dy, dt, t_next):
    """Build right-hand side for implicit system."""
    rhs = T_curr.copy()

    # Heat source
    q = np.array(cfg.heat_source(X, Y, t_next))
    rhs += dt * q

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs[0, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[-1, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[:, 0] += ry * (cfg.h * dy / cfg.k) * bc_term
    rhs[:, -1] += ry * (cfg.h * dy / cfg.k) * bc_term

    return rhs.flatten()
