"""Script for training and plotting the PINN model."""

import os

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

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################

    print("Solving heat equation with PINN...") 

    # Generate traning data
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    
    # Train parameters and gather the loss history
    pinn_params, losses = train_pinn(sensor_data, cfg)

    # Predict temperature evolution
    T_PINN = predict_grid(pinn_params["nn"], x, y, t, cfg)

    print("\nGenerating PINN visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_PINN,
        save_path="output/pinn/pinn_snapshots.png",
    )
    create_animation(
        x, y, t, T_PINN, title="PINN", save_path="output/pinn/pinn_animation.gif"
    )


    # Create plots for the error
    T_err = T_PINN - T_fdm

    print("\nGenerating PINN Error visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_err,
        save_path="output/pinn/pinn_err_snapshots.png",
    )
    create_animation(
        x, y, t, T_err, title="PINN Error", save_path="output/pinn/pinn_err_animation.gif"
    )

    # Gather different losses
    data_loss = losses["data"]
    ic_loss = losses["ic"]
    physics_loss = losses["physics"]
    bc_loss = losses["bc"]
    total_loss = losses["total"]

    num_loss = np.arange(len(total_loss))

    plt.figure(figsize=(10, 5))

    # Plot the different losses 
    plt.plot(num_loss, data_loss, "r", label="Data loss")
    plt.plot(num_loss, ic_loss, "y", label="IC loss")
    plt.plot(num_loss, physics_loss, "b", label="Physics loss")
    plt.plot(num_loss, bc_loss, "g",   label="BC loss")

    # Plot the total loss
    plt.plot(num_loss, total_loss, "m", label="Total loss")

    plt.xlabel("Epoch")
    plt.title("Evolution of losses during training of PINN")
    plt.legend()
    plt.show()

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
