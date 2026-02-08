"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []} 

    # Define the loss function (without JIT, value_and_grad will handle it)
    def loss_func(nn_params, sensor_data, ic_epoch):
        data_loss_epoch = data_loss(nn_params, sensor_data, cfg)
        ic_loss_epoch = ic_loss(nn_params, ic_epoch, cfg)

        loss_func_epoch = cfg.lambda_data*data_loss_epoch + cfg.lambda_ic*ic_loss_epoch

        return loss_func_epoch, (data_loss_epoch, ic_loss_epoch)
    
    value_and_grad = jax.jit(jax.value_and_grad(loss_func, has_aux=True))
    
    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):

        # Sample IC points
        ic_epoch, key = sample_ic(key, cfg)

        # Estimate loss function for the sampled points and gradients
        (loss_func_epoch, aux), grads = value_and_grad(nn_params, sensor_data, ic_epoch)

        # Gather the different losses
        (data_loss_epoch, ic_loss_epoch) = aux


        losses["total"].append(loss_func_epoch)
        losses["ic"].append(ic_loss_epoch)
        losses["data"].append(data_loss_epoch)

        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)


    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config):

    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    # SoftAdapt parameters
    beta = 0.1
    epsilon = 1e-8
    n = 5 # buffer length
    m = 4 # number of losses

    # Start-weights
    alphas = jnp.ones(m) / m

    # Buffer for loss
    loss_buffer = jnp.zeros((n, m))
    buf_idx = 0
    buf_filled = False

    # Loss function (WLoss)
    def loss_func(pinn_params, sensor_data, ic_epoch, interior_epoch, bc_epoch, alphas):

        data_loss_epoch = data_loss(pinn_params["nn"], sensor_data, cfg)
        ic_loss_epoch = ic_loss(pinn_params["nn"], ic_epoch, cfg)
        phys_loss_epoch = physics_loss(pinn_params, interior_epoch, cfg)
        bc_loss_epoch = bc_loss(pinn_params, bc_epoch, cfg)

        losses_vec = jnp.array([data_loss_epoch, ic_loss_epoch, phys_loss_epoch, bc_loss_epoch])

        # Weighted loss for optimizer
        # Avoid WLoss being affected by gradient-descent
        WLoss = jnp.sum(jax.lax.stop_gradient(alphas) * losses_vec)

        return WLoss, losses_vec

    value_and_grad = jax.jit(jax.value_and_grad(loss_func, has_aux=True))

    # Training loop
    for epoch in tqdm(range(cfg.num_epochs)):

        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)

        (WLoss, losses_vec), grads = value_and_grad(
            pinn_params,
            sensor_data,
            ic_epoch,
            interior_epoch,
            bc_epoch,
            alphas,
        )

        # Update buffer
        loss_buffer = loss_buffer.at[buf_idx].set(losses_vec)
        buf_idx = (buf_idx + 1) % n
        buf_filled = buf_filled or (buf_idx == 0)

        # Use SoftAdapt when buffer is full 
        if buf_filled:

            # Reorder buffer so time is increasing
            L = jnp.roll(loss_buffer, -buf_idx, axis=0) # shape (n, m)

            # f_i = mean loss
            f = jnp.mean(L, axis=0)

            # s_i = rate of change (1st order)
            dL = jnp.diff(L, axis=0) # shape (n-1, m)
            s = jnp.mean(dL, axis=0)

            # Normalized SoftAdapt
            s = s / (jnp.sum(jnp.abs(s)) + epsilon)

            # Softmax with numerical stabilization
            s_hat = beta * (s - jnp.max(s))
            alpha_raw = jnp.exp(s_hat)
            alpha_raw = alpha_raw / (jnp.sum(alpha_raw) + epsilon)

            # Loss-weighted variant
            alphas = f * alpha_raw
            alphas = alphas / (jnp.sum(alphas) + epsilon)

 
        pinn_params, opt_state = adam_step(
            pinn_params, grads, opt_state, lr=cfg.learning_rate
        )

        # Log true loss
        TLoss = jnp.sum(losses_vec)

        losses["total"].append(TLoss)
        losses["data"].append(losses_vec[0])
        losses["ic"].append(losses_vec[1])
        losses["physics"].append(losses_vec[2])
        losses["bc"].append(losses_vec[3])

    return pinn_params, alphas, {k: jnp.array(v) for k, v in losses.items()}
