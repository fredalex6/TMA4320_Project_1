
import jax.numpy as jnp


from project import (
    generate_training_data,
    load_config,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with PINN...") 

    # Generate traning data
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    
    # Train parameters with PINN and gather the loss history
    pinn_params, losses = train_pinn(sensor_data, cfg)

    # Gather the parameters
    alpha_pinn = jnp.exp(pinn_params["log_alpha"])
    k_pinn = jnp.exp(pinn_params['log_k'])
    h_pinn = jnp.exp(pinn_params['log_h'])
    P_pinn = jnp.exp(pinn_params['log_power'])

    # Estimate the relative errors
    alpha_rel = jnp.abs((alpha_pinn - cfg.alpha)/cfg.alpha)
    k_rel = jnp.abs((k_pinn - cfg.k)/cfg.k)
    h_rel = jnp.abs((h_pinn - cfg.h)/cfg.h)
    P_rel = jnp.abs((P_pinn - cfg.source_strength)/cfg.source_strength)


    print("Estimated parameters")
    print(f"alpha = {alpha_pinn}, k = {k_pinn}, h = {h_pinn} og P = {P_pinn}")

    print("Relative error")
    print(f"alpha_rel = {alpha_rel}, k_rel = {k_rel}, h_rel = {h_rel} og P_rel = {P_rel}")


if __name__ == "__main__":
    main()