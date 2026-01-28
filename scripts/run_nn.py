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
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    nn_params, losses = train_nn(sensor_data, cfg)
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

    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
