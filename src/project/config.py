"""Configuration loader for PINN project."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import yaml


@dataclass
class Config:
    """Configuration for the PINN simulation."""

    # Domain
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float

    # Physics
    alpha: float
    k: float
    h: float
    T_outside: float
    varmekapasitet: float

    # Source
    source_locations: jnp.ndarray
    source_sizes: jnp.ndarray
    source_strength: float
    num_rooms: int

    # Grid
    nx: int
    ny: int
    nt: int
    nmu: int
    nt_pos: int

    # Sensors
    sensor_rate: float
    sensor_noise: float
    sensor_locations: jnp.ndarray

    # Training
    layer_sizes: list
    learning_rate: float
    num_epochs: int
    seed: int
    lambda_physics: float
    lambda_ic: float
    lambda_bc: float
    lambda_data: float
    # 3.4
    lambda_jevn_varme: float
    lambda_varmetap: float

    num_collocation: int
    num_ic: int
    num_bc: int

    def is_source(self, x, y):
        """Check if point(s) are inside any heat source."""
        # source_locations: (S, 2), source_sizes: (S,)
        cx = self.source_locations[:, 0]  # (S,)
        cy = self.source_locations[:, 1]  # (S,)
        sizes = self.source_sizes  # (S,)

        # Broadcast x, y against source centers
        # x, y can be scalars or arrays of any shape
        dx = jnp.abs(x - cx[:, None, None])  # (S, ...) broadcasts with x
        dy = jnp.abs(y - cy[:, None, None])  # (S, ...) broadcasts with y

        inside = (dx <= sizes[:, None, None]) & (dy <= sizes[:, None, None])
        return jnp.any(inside, axis=0)  # same shape as x, y

    def heat_source(self, x, y, t):
        """Heat source term at point (x, y, t)."""
        return jnp.where(self.is_source(x, y), self.source_strength, 0.0)
    

    # Create new identical functions for the amortised training
    def is_source_am(self, x, y, cx, cy):
        """Check if point(s) are inside any heat source."""
        # source_sizes: (S,)
        sizes = self.source_sizes  # (S,)

        # Broadcast x, y against source centers
        # x, y can be scalars or arrays of any shape
        dx = jnp.abs(x - cx)  # (S, ...) broadcasts with x
        dy = jnp.abs(y - cy)  # (S, ...) broadcasts with y

        inside = (dx <= sizes[:, None, None]) & (dy <= sizes[:, None, None])
        return jnp.any(inside, axis=0)  # same shape as x, y

    def heat_source_am(self, x, y, t, cx, cy):
        """Heat source term at point (x, y, t)."""
        return jnp.where(self.is_source_am(x, y, cx, cy), self.source_strength, 0.0)
    
    #####3.4
    

    def is_source_34(self, x, y, new_source_locations):
        """Check if point(s) are inside any heat source."""
        # source_locations:
        cx = new_source_locations[:, 0]  
        cy = new_source_locations[:, 1]  
        sizes = self.source_sizes  

        # Broadcast x, y against source centers
        # x, y can be scalars or arrays of any shape
        dx = jnp.abs(x - cx[:, None, None])  # (S, ...) broadcasts with x
        dy = jnp.abs(y - cy[:, None, None])  # (S, ...) broadcasts with y

        inside = (dx <= sizes[:, None, None]) & (dy <= sizes[:, None, None])
        return jnp.any(inside, axis=0)  # same shape as x, y

    def heat_source_34(self, x, y, new_source_locations):
        """Heat source term at point (x, y, t)."""
        return jnp.where(self.is_source_34(x, y, new_source_locations), self.source_strength, 0.0)

    def is_source_34_soft(self, x, y, new_source_locations):
        # cx, cy: (S,)
        cx = new_source_locations[:, 0]
        cy = new_source_locations[:, 1]
        sizes = jnp.asarray(self.source_sizes)

        # distance from source centers
        dx = jnp.abs(x - cx[:, None, None])
        dy = jnp.abs(y - cy[:, None, None])

        # Use fixed softness to avoid division by zero/infinity issues
        softness = 5.0

        # smooth mask: sigmoid approximation
        mask_x = 1 / (1 + jnp.exp(softness * (dx - sizes[:, None, None])))
        mask_y = 1 / (1 + jnp.exp(softness * (dy - sizes[:, None, None])))

        # combined mask
        mask = mask_x * mask_y

        # combine all sources
        return jnp.clip(jnp.sum(mask, axis=0), 0.0, 1.0)

    def heat_source_34_soft(self, x, y, new_source_locations):
        return jnp.asarray(self.source_strength) * self.is_source_34_soft(x, y, new_source_locations)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        # Domain
        x_min=data["domain"]["x_min"],
        x_max=data["domain"]["x_max"],
        y_min=data["domain"]["y_min"],
        y_max=data["domain"]["y_max"],
        t_min=data["time"]["t_min"],
        t_max=data["time"]["t_max"],
        # Physics
        alpha=data["physics"]["alpha"],
        k=data["physics"]["k"],
        h=data["physics"]["h"],
        T_outside=data["physics"]["T_outside"],
        varmekapasitet=data["physics"]["varmekapasitet"],  # make sure you add this too
        # Source
        source_locations=jnp.asarray(
            data["source"]["locations"],
        ),
        source_sizes=jnp.asarray(
            data["source"]["sizes"],
        ),
        source_strength=data["source"]["strength"],
        num_rooms=data["source"]["num_rooms"],
        # Grid
        nx=data["grid"]["nx"],
        ny=data["grid"]["ny"],
        nt=data["grid"]["nt"],
        nmu=data["grid"]["nmu"], # For amortization
        nt_pos = data["grid"]["nt_pos"], # 3.4
        # Sensors
        sensor_rate=data["sensors"]["measure_rate"],
        sensor_noise=data["sensors"]["noise_std"],
        sensor_locations=jnp.asarray(
            data["sensors"]["locations"],
        ),  # shape (n_sensors, 2)
        # Training
        layer_sizes=data["training"]["layer_sizes"],
        learning_rate=data["training"]["learning_rate"],
        num_epochs=data["training"]["num_epochs"],
        seed=data["training"]["seed"],
        lambda_physics=data["training"]["lambda_physics"],
        lambda_ic=data["training"]["lambda_ic"],
        lambda_bc=data["training"]["lambda_bc"],
        lambda_data=data["training"]["lambda_data"],
        # 3.4
        lambda_jevn_varme = data["training"]["lambda_jevn_varme"],
        lambda_varmetap = data["training"]["lambda_varmetap"],
        num_collocation=data["training"]["num_collocation"],
        num_ic=data["training"]["num_ic"],
        num_bc=data["training"]["num_bc"],
    )
