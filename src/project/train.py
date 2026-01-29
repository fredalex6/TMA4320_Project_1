"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
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

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    # Define the loss function and compile with JIT
    @jax.jit
    def loss_func(nn_params,sensor_data, ic_epoch):
        return cfg.lambda_data*data_loss(nn_params,sensor_data, cfg) + cfg.lambda_ic*ic_loss(nn_params,ic_epoch, cfg)
    
    # Compile value_and_grad with JIT
    value_and_grad = jax.jit(jax.value_and_grad(loss_func))
    
    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):

        # Sample IC points
        ic_epoch, key = sample_ic(key, cfg)

        # Estimate loss function for the sampled points and gradients
        loss_func_epoch, grads = value_and_grad(nn_params,sensor_data, ic_epoch)

        # Estimate the different losses
        data_loss_epoch = data_loss(nn_params,sensor_data, cfg)
        ic_loss_epoch = ic_loss(nn_params,ic_epoch, cfg)

        # Update the nn_params and losses dictionary
        losses["total"].append(loss_func_epoch)
        losses["ic"].append(ic_loss_epoch)
        losses["data"].append(data_loss_epoch)

        # Find nest iteration of parameters
        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################


    # Define the loss function and compile with JIT
    @jax.jit
    def loss_func(pinn_params, sensor_data, ic_epoch, interior_epoch, bc_epoch): 
        data_loss_epoch = cfg.lambda_data * data_loss(pinn_params["nn"], sensor_data, cfg)
        ic_loss_epoch = cfg.lambda_ic * ic_loss(pinn_params["nn"], ic_epoch, cfg)
        physics_loss_epoch = cfg.lambda_physics * physics_loss(pinn_params, interior_epoch, cfg)
        bc_loss_epoch = cfg.lambda_bc * bc_loss(pinn_params, bc_epoch, cfg)

        return data_loss_epoch + ic_loss_epoch + physics_loss_epoch + bc_loss_epoch

    # Compile value_and_grad with JIT
    value_and_grad = jax.jit(jax.value_and_grad(loss_func)) 

    for _ in tqdm(range(cfg.num_epochs), desc="Training PINN"): 

        # Sample points
        interior_epoch, key = sample_interior(key, cfg) 
        ic_epoch, key = sample_ic(key, cfg) 
        bc_epoch, key = sample_bc(key, cfg) 

        # Estimate total loss and gradients
        loss_func_epoch, grads = value_and_grad(
            pinn_params, sensor_data, ic_epoch, interior_epoch, bc_epoch
        )

        # Estimate each loss 
        data_loss_epoch = data_loss(pinn_params["nn"], sensor_data, cfg) 
        ic_loss_epoch = ic_loss(pinn_params["nn"], ic_epoch, cfg) 
        physics_loss_epoch = physics_loss(pinn_params, interior_epoch, cfg) 
        bc_loss_epoch = bc_loss(pinn_params, bc_epoch, cfg)

        # Log the different losses
        losses["total"].append(loss_func_epoch) 
        losses["data"].append(data_loss_epoch) 
        losses["physics"].append(physics_loss_epoch) 
        losses["ic"].append(ic_loss_epoch) 
        losses["bc"].append(bc_loss_epoch)

        # Estimate next parameters
        pinn_params, opt_state = adam_step(
            pinn_params, grads, opt_state, lr=cfg.learning_rate
        )

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
