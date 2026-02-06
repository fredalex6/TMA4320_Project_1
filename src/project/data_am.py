"""Data generation utilities."""

import jax.numpy as jnp
import numpy as np

from .config import Config
from .fdm_am import solve_heat_equation


def generate_training_data(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, jnp.ndarray]:
    """Generate training data from FDM solver.

    Args:
        cfg: Configuration

    Returns:
        x, y, t: Coordinate arrays
        T_fdm: FDM solution (nt, nx, ny)
        sensor_data: Sensor measurements [x, y, t, T_noisy]
    """

    # Create grids
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)
    cx = np.linspace(cfg.x_min, cfg.x_max, cfg.nmu)
    cy = np.linspace(cfg.y_min, cfg.y_max, cfg.nmu)

    # Gather all sensor data in one list
    all_sensor_data = []

    # Random generation of heat source location
    for k in range(cfg.num_rooms):

        cx = np.random.uniform(cfg.x_min, cfg.x_max)
        cy = np.random.uniform(cfg.y_min, cfg.y_max)

        # Use the numerical solver for T
        T_fdm = solve_heat_equation(cfg, cx, cy)[3]

        scenario_data = _generate_sensor_data(x, y, t, cx, cy, T_fdm, cfg)
        all_sensor_data.append(scenario_data)


    sensor_data_combined = np.vstack(all_sensor_data)

    return x, y, t, jnp.asarray(sensor_data_combined)


def _generate_sensor_data(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, cx:float, cy:float, T: np.ndarray, cfg: Config
) -> np.ndarray:
    """Generate noisy sensor measurements from FDM solution."""
    sensor_data = []

    for sx, sy in cfg.sensor_locations:
        # Find nearest grid point
        i = np.argmin(np.abs(x - sx))
        j = np.argmin(np.abs(y - sy))

        # Sample at specified rate
        dt = t[1] - t[0]
        for t_idx, time in enumerate(t):
            if time % cfg.sensor_rate < dt or t_idx == 0:
                temp = T[t_idx, i, j]
                temp_noisy = temp + np.random.normal(0, cfg.sensor_noise)

                sensor_data.append([x[i], y[j], time, cx, cy, temp_noisy])

    return np.array(sensor_data)
