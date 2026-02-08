"""Script for optimizing sorce_locations."""

import matplotlib.pyplot as plt
import jax.numpy as jnp

from project.optimize_pos import optimize_pos_fun

from project import load_config


def main():
    print("main kjører")
    cfg = load_config("config.yaml")

    print("solving optimized positions") 


    sorce_locations, losses = optimize_pos_fun(cfg)
    loss_func_step = losses["loss_func_step"]
    varmetap_step = losses["varme_tap"]
    loss_stabil_temp = losses["Loss_stabil_temp"]
    spread_penalty = losses["spread_penalty"]

    num_loss = jnp.arange(len(loss_func_step))

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # Plot the losses seperately 
    axs[0].plot(num_loss, varmetap_step, '-r')
    axs[0].set_title('varmetap (negative avg temp - want to minimize)')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(num_loss, loss_stabil_temp, '-g')
    axs[1].set_title('Loss_stabil_temp (variance)')
    axs[1].set_xlabel('Epoch')

    axs[2].plot(num_loss, spread_penalty, '-m')
    axs[2].set_title('Spread penalty (clustering penalty)')
    axs[2].set_xlabel('Epoch')

    axs[3].plot(num_loss, loss_func_step, '-b')
    axs[3].set_title('Total loss')
    axs[3].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

    print(sorce_locations)

if __name__ == "__main__":
    main()

