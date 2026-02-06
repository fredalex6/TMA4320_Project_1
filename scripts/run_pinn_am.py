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

    print("Loading PINN parameters...")

    # Load params from file
    filename = './output/pinn/pinn.pkl'

    with open(filename, "rb") as f:
        pinn_params = pickle.load(f)

    print(f"PINN loaded from {filename}")

    # Create grid
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    # Choose an arbitrary heat source position
    test_cx = 0
    test_cy = 0

    print(f"Predikerer temperaturutviklingen for varmekilde ved: ({test_cx:.2f}, {test_cy:.2f})")

    # Predict temperature evolution
    T_PINN = predict_grid(pinn_params["nn"], x, y, t, test_cx, test_cy, cfg)

    print("\nGenerating PINN visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_PINN,
        save_path="output/pinn/pinn_am_snapshots.png",
    )
    create_animation(
        x, y, t, T_PINN, title="PINN Amortised", save_path="output/pinn/pinn_am_animation.gif"
    )

if __name__ == "__main__":
    main()
