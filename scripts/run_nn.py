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

    # Create plots for the error
    T_err = T_NN - T_fdm

    print("\nGenerating NN Error visualizations...")

    plot_snapshots(
        x,
        y,
        t,
        T_err,
        save_path="output/nn/nn_err_snapshots.png",
    )
    create_animation(
        x, y, t, T_err, title="NN Error", save_path="output/nn/nn_err_animation.gif"
    )


    # Gather different losses
    data_loss = losses["data"]
    ic_loss = losses["ic"]
    total_loss = losses["total"]

    num_loss = np.arange(len(total_loss))

    plt.figure(figsize=(10, 8))

    # Plot the different losses 
    plt.plot(num_loss, data_loss, "r", label="Data loss")
    plt.plot(num_loss, ic_loss, "y", label="IC loss")

    # Plot the total loss
    plt.plot(num_loss, total_loss, "m", label="Total loss")

    plt.xlabel("Epoch")
    plt.title("Evolution of losses during training of NN")
    plt.legend()
    plt.show()

    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
