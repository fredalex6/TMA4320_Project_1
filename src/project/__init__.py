from .config import Config, load_config
from .data import generate_training_data
from .fdm import solve_heat_equation
from .loss import data_loss, ic_loss, physics_loss
from .model import forward, init_nn_params, init_pinn_params, predict_grid
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior
from .train import train_nn, train_pinn

# Import for Adaptive Loss Weighting
#from .train_woptim import train_nn, train_pinn



# These ones should be imported when using
"""from .config import Config, load_config
from .data_am import generate_training_data
from .fdm_am import solve_heat_equation
from .loss_am import data_loss, ic_loss, physics_loss
from .model_am import forward, init_nn_params, init_pinn_params, predict_grid
from .optim import adam_step, init_adam
from .sampling_am import sample_bc, sample_ic, sample_interior
from .train_am import train_nn, train_pinn"""

__all__ = [
    "Config",
    "load_config",
    "solve_heat_equation",
    "init_nn_params",
    "init_pinn_params",
    "forward",
    "predict_grid",
    "data_loss",
    "physics_loss",
    "ic_loss",
    "init_adam",
    "adam_step",
    "train_nn",
    "train_pinn",
    "generate_training_data",
    "sample_interior",
    "sample_ic",
    "sample_bc",
]
