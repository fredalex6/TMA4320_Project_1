"""Script for training and plotting the PINN model with amortised training."""

import pickle

import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    load_config,
    predict_grid,
    solve_heat_equation
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
    test_cx = 2
    test_cy = 2

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

    # Make error plot
    x, y, t, T_fdm = solve_heat_equation(cfg, test_cx, test_cy)
    T_err = T_PINN - T_fdm

    print("\nGenerating PINN Error visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_err,
        save_path="output/pinn/pinn_am_err_snapshots.png",
    )
    create_animation(
        x, y, t, T_err, title="PINN Amortised Error", save_path="output/pinn/pinn_am_err_animation.gif"
    )


if __name__ == "__main__":
    main()
