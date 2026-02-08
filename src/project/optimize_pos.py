import jax
import jax.numpy as jnp
from tqdm import tqdm

from .optim import adam_step, init_adam
from .loss_34 import varmetap, loss_stabil_temp, source_spread_penalty
from .fdm_34 import solve_heat_equation_34



def optimize_pos_fun(cfg):
    #@jax.jit
    print("Kjører optimize")

    #solve heat ...
    def total_loss_34(source_locations, cfg):
        #Må se på hvordan man bruker jax her
        x, y, t, temp = solve_heat_equation_34(cfg,source_locations)
        loss_stabil_temp_val = loss_stabil_temp(temp)
        varmetap_val = varmetap(x, y, temp, cfg)
        spread_penalty = source_spread_penalty(source_locations)
        

        loss_func_source = (cfg.lambda_varmetap*varmetap_val + 
                          cfg.lambda_jevn_varme*loss_stabil_temp_val +
                          0.5*spread_penalty)  # Weight for spread penalty

        return loss_func_source, (varmetap_val, loss_stabil_temp_val, spread_penalty)


    #Lager source locations som kan forandres på
    source_locations = cfg.source_locations[:]
    source_locations = jnp.asarray(source_locations)
    #Adam state start
    adam_state = init_adam(source_locations)


    #Iterations, time_index, losses
    N = 150
    time_inx = cfg.nt_pos
    losses = {"loss_func_step": [], "varme_tap": [], "Loss_stabil_temp": [], "spread_penalty": []}  

    #compile value and grad
    value_and_grad = jax.value_and_grad(total_loss_34, has_aux=True)


    for _ in tqdm(range(N),desc="Optimizing position parameters" ):
        # Estimate loss function and gradients
        #with respect to source_locations
        (loss_func_step, aux), grads = value_and_grad(source_locations,cfg)
        (varmetap_step, loss_stabil_temp_val, spread_penalty_val) = aux

        # Update the pos params and losses dictionary
        losses["loss_func_step"].append(loss_func_step)
        losses["varme_tap"].append(varmetap_step)
        losses["Loss_stabil_temp"].append(loss_stabil_temp_val)
        losses["spread_penalty"].append(spread_penalty_val)

        source_locations, adam_state = adam_step(source_locations, grads, adam_state, lr=cfg.learning_rate)
        
        # make sure within room boundaries
        source_locations = jnp.clip(source_locations, 
                                   jnp.array([[cfg.x_min + 0.5, cfg.y_min + 0.5]]),
                                   jnp.array([[cfg.x_max - 0.5, cfg.y_max - 0.5]]))
    for i in range(4):
        print(source_locations[i][0])
        print(source_locations[i][1])


    return source_locations, {k: jnp.array(v) for k, v in losses.items()}




