import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

#@jax.jit
def varmetap(x, y, temp, cfg):
    """Heat loss metric: temperature drop from start to end.
    Smaller drop = better heat retention.
    
    Fixed: Correct indexing for temp shape (nt, nx, ny)
    """
    # Average temperature at start and end
    T_initial = jnp.mean(temp[0, :, :])    
    T_final = jnp.mean(temp[-1, :, :])     
    
    # Heat loss 
    varmetap_val = T_initial - T_final
    
    return varmetap_val

def source_spread_penalty(source_locations):
    """Penalize sources being too close to each other.
    Encourages spatial distribution of heat sources.
    """
    n_sources = source_locations.shape[0]
    total_dist = 0.0
    
    for i in range(n_sources):
        for j in range(i+1, n_sources):
            dist = jnp.sqrt(jnp.sum((source_locations[i] - source_locations[j])**2))
            # Penalty increases as sources get closer (inverse distance)
            total_dist += 1.0 / (dist + 0.1)  # +0.1 to avoid division by zero
    
    return total_dist

#@jax.jit
def loss_stabil_temp(Temp):
    """Temperature variance at final time step.
    Lower variance = more uniform temperature = better stability.
    
    Fixed: Only evaluate at final time step, correct indexing
    """
    # Temp shape: (nt, nx, ny)
    Temp_final = Temp[-1, :, :]  # Last time step: shape (nx, ny)
    Temp_avg = jnp.mean(Temp_final)
    
    # Variance at final time
    loss_stabil_temp = jnp.mean((Temp_final - Temp_avg)**2)
    
    return loss_stabil_temp

#cfg.lambda_jevn_varme*jax.mean((temp-Temp_avg)**2) + cfg.lambda_varmetap*