"""Finite Difference Method solver for the 2D heat equation."""

import numpy as np

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config


def solve_heat_equation_34(
    cfg: Config, 
    source_locations: jnp.ndarray,
):
    """Solve the 2D heat equation using implicit Euler.

    Args:
        cfg: Configuration object

    Returns:
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        t: time points (nt,)
        T: temperature solution (nt, nx, ny)
    """
    # Bruker JAX for auto-diff senere

    # Create grids
    x = jnp.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = jnp.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = jnp.linspace(cfg.t_min, cfg.t_max, cfg.nt_pos)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Initialisering av temperaturvektoren
    T = jnp.zeros((cfg.nt_pos, cfg.nx, cfg.ny))

    # Fyller første tidssteg med T_outside
    T = T.at[0, :, :].set(cfg.T_outside)


    for i in range(cfg.nt_pos-1):

        T_curr = T[i, :, :]
        t_next = t[i+1]

        # Matrisen A i det lineære systemet
        A = _build_matrix_34(cfg, dx, dy, dt)

        # RHS
        b = _build_rhs_34(cfg, T_curr, X, Y, dx, dy, dt, t_next, source_locations)

        # Oppdaterer arrayet T
        T_next = jnp.linalg.solve(A, b).reshape(cfg.nx, cfg.ny) 
        T = T.at[i + 1, :, :].set(T_next) 

    return x, y, t, T


def _build_matrix_34(cfg: Config, dx, dy, dt):
    """Build the implicit Euler system matrix."""
    n = cfg.nx * cfg.ny
    A = jnp.zeros((n, n))

    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    def idx(i, j):
        return i * cfg.ny + j

    I, J = jnp.meshgrid(jnp.arange(cfg.nx), jnp.arange(cfg.ny), indexing="ij")

    # Boundary masks
    left = I == 0
    right = I == cfg.nx - 1
    bottom = J == 0
    top = J == cfg.ny - 1

    # Diagonal entries
    diag = jnp.full((cfg.nx, cfg.ny), 1 + 2 * rx + 2 * ry)
    diag = diag.at[left | right].add(-rx) 
    diag = diag.at[bottom | top].add(-ry) 
    diag = diag.at[left | right].add(rx * cfg.h * dx / cfg.k) 
    diag = diag.at[bottom | top].add(ry * cfg.h * dy / cfg.k)

    p = idx(I, J)
    A = A.at[p, p].set(diag) 

   

    # Off-diagonals
    mask = ~left
    A = A.at[idx(I[mask], J[mask]), idx(I[mask] - 1, J[mask])].set(-rx)

    mask = ~right
    A = A.at[idx(I[mask], J[mask]), idx(I[mask] + 1, J[mask])].set(-rx) 

    mask = ~bottom
    A = A.at[idx(I[mask], J[mask]), idx(I[mask], J[mask] - 1)].set(-ry) 

    mask = ~top
    A = A.at[idx(I[mask], J[mask]), idx(I[mask], J[mask] + 1)].set(-ry) 

    return A
 

def _build_rhs_34(cfg: Config, T_curr, X, Y, dx, dy, dt, t_next, new_source_locations):
    """Build right-hand side for implicit system."""
    rhs = T_curr 

    # Heat source
    q = jnp.array(cfg.heat_source_34_soft(X, Y, new_source_locations))
    rhs = rhs + dt * q

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs = rhs.at[0, :].add(rx * (cfg.h * dx / cfg.k) * bc_term) 
    rhs = rhs.at[-1, :].add(rx * (cfg.h * dx / cfg.k) * bc_term) 
    rhs = rhs.at[:, 0].add(ry * (cfg.h * dy / cfg.k) * bc_term) 
    rhs = rhs.at[:, -1].add(ry * (cfg.h * dy / cfg.k) * bc_term) 

    return rhs.flatten()
