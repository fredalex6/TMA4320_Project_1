"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    print("Solving heat equation with NN...") 

    # Gather training data
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    # Train parameters and gather the loss history
    nn_params, losses = train_nn(sensor_data, cfg)

    # Predict temperature evolution
    T_NN = predict_grid(nn_params,x,y,t,cfg)

    print("\nGenerating NN visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_NN,
        save_path="output/nn/nn_snapshots.png",
    )
    create_animation(
        x, y, t, T_NN, title="NN", save_path="output/nn/nn_animation.gif"
    )

    fig, axs = plt.subplots(3, 1, figsize=(10, 5))

    # Gather different losses
    data_loss = losses["data"]
    ic_loss = losses["ic"]
    total_loss = losses["total"]

    num_loss = np.arange(len(total_loss))

    # Plot the losses seperately 
    axs[0].plot(num_loss, data_loss, '-r')
    axs[0].set_title('Data loss')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(num_loss, ic_loss, '-g')
    axs[1].set_title('IC loss')
    axs[1].set_xlabel('Epoch')

    axs[2].plot(num_loss, total_loss, '-b')
    axs[2].set_title('Total loss')
    axs[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
