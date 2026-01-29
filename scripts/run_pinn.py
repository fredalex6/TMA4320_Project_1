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

    fig, axs = plt.subplots(3, 1, figsize=(10, 5))

    # Gather different losses
    data_loss = losses["data"]
    ic_loss = losses["ic"]
    physics_loss = losses["physics"]
    bc_loss = losses["bc"]
    total_loss = losses["total"]

    num_loss = np.arange(len(total_loss))

    # Plot the losses seperately 
    axs[0].plot(num_loss, data_loss, '-r')
    axs[0].set_title('Data loss')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(num_loss, ic_loss, '-g')
    axs[1].set_title('IC loss')
    axs[1].set_xlabel('Epoch')

    axs[2].plot(num_loss, physics_loss, '-b')
    axs[2].set_title('Physics loss')
    axs[2].set_xlabel('Epoch')

    axs[3].plot(num_loss, bc_loss, '-g')
    axs[3].set_title('BC loss')
    axs[3].set_xlabel('Epoch')

    axs[4].plot(num_loss, total_loss, '-b')
    axs[4].set_title('Total loss')
    axs[4].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
