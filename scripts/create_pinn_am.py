"""Script for training and plotting the PINN model."""

import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with PINN...") 

    # Redefine layer structure, this is a bit arbitrary, but width of middle layers does need some size
    cfg.layer_sizes = [5, 50, 50, 50, 50, 1] 

    # Generate traning data
    x, y, t, sensor_data = generate_training_data(cfg)
    
    # Train PINN parameters
    pinn_params = train_pinn(sensor_data, cfg)

    print(f"Saving PINN params...")

    # Save params to file
    filename = './output/pinn/pinn.pkl'

    # Create file or overwrite
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    
    with open(filename, "wb") as f:
        pickle.dump(pinn_params, f)

    print(f"PINN saved to {filename}")


if __name__ == "__main__":
    main()
