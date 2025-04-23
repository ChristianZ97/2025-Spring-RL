"""
optimize_ddpg.py - Script for optimizing DDPG hyperparameters using Bayesian Optimization.

This script uses scikit-optimize to perform Bayesian Optimization on the DDPG
algorithm's hyperparameters to maximize performance on reinforcement learning tasks.
"""

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
# import wandb
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import the DDPG training function
from ddpg import train, env_name, random_seed, device
# from ddpg_cheetah import train, env_name, random_seed, device

import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import gc

from itertools import count
counter = count(start=0)

# Define the hyperparameter search space
search_space = [
    Real(0.95, 0.99, name='gamma'),                       
    Real(0.005, 0.02, name='tau'),                        
    Real(0.1, 0.4, name='noise_scale'),                   
    Real(1e-4, 1e-3, name='lr_a', prior='log-uniform'),   
    Real(1e-4, 1e-3, name='lr_c', prior='log-uniform'),   
    Integer(1, 4, name='updates_per_step'),              
]

# Define the objective function for Bayesian Optimization
@use_named_args(search_space)
def objective(gamma, tau, noise_scale, lr_a, lr_c, updates_per_step):
    """
    Objective function for Bayesian Optimization.
    Runs DDPG with given hyperparameters and returns negative reward for minimization.
    """
    print(f"\nTrying parameters: gamma={gamma:.6f}, tau={tau:.6f}, noise_scale={noise_scale:.6f}, lr_a={lr_a:.6f}, lr_c={lr_c:.6f}, updates_per_step={updates_per_step}")
    env = gym.make(env_name)
    
    # Set random seeds for reproducibility
    global counter
    bo_step = next(counter)
    bo_seed = random_seed + bo_step
    
    env.seed(bo_seed)
    torch.manual_seed(bo_seed)
    np.random.seed(bo_seed)
    random.seed(bo_seed)
    
    start_time = time.time()
    writer = SummaryWriter(f"./tb_record_pendulum/{bo_step}")

    results = train(
        env=env,
        num_episodes=75, # Use fewer episodes for optimization to save time
        gamma=gamma,
        tau=tau,
        noise_scale=noise_scale,
        lr_a=lr_a,
        lr_c=lr_c,
        render=False,   # No rendering during optimization
        save_model=False,  # Don't save models during optimization
        writer=writer
    )
    
    duration = time.time() - start_time
    final_rewards = results['last_rewards']
    
    final_mean = np.mean(final_rewards[-10:])
    stability = -np.std(final_rewards[-10:]) 
    score = final_mean + 0.1 * stability

    print(f"Training done in {duration:.1f}s | Mean reward: {final_mean:.2f} | Std: {-stability:.2f} | Score: {score:.2f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    
    return -score

def run_optimization(n_calls=20, n_random_starts=5, output_dir='optimization_results'):
    """
    Run the Bayesian Optimization process to find optimal DDPG hyperparameters.
    """
    print("Starting Bayesian Optimization process...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Run Bayesian Optimization
    result = gp_minimize(
        objective,                # Objective function to minimize
        search_space,             # Hyperparameter search space
        n_calls=n_calls,          # Total number of evaluations
        n_random_starts=n_random_starts,  # Initial random evaluations
        verbose=True,             # Print progress
        random_state=random_seed, # Random seed
        n_jobs=1
    )
    
    # Extract best hyperparameters
    best_gamma, best_tau, best_noise_scale, best_lr_a, best_lr_c, best_updates_per_step = result.x
    
    print("\nOptimization completed!")
    print("Best hyperparameters:")
    for name, value in zip(['gamma', 'tau', 'noise_scale', 'lr_a', 'lr_c', 'updates_per_step'], result.x):
        print(f"{name}: {value}")
    
    print(f"Best reward: {-result.fun:.2f}")
    
    # Save optimization history
    with open(f"{output_dir}/optimization_history.txt", "w") as f:
        f.write("Iteration, Objective Value, Parameters\n")
        for i, (value, params) in enumerate(zip(result.func_vals, result.x_iters)):
            param_str = ", ".join([f"{name}={value}" for name, value in 
                                  zip(['gamma', 'tau', 'noise_scale', 'lr_a', 'lr_c', 'updates_per_step'], params)])
            seed = random_seed + i
            f.write(f"{i}, {-value:.4f}, {param_str}, seed={seed}\n")
    
    # Create visualization plots
    plt.figure(figsize=(10, 6))
    plot_convergence(result)
    plt.savefig(f'{output_dir}/convergence.png')
    plt.close()
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_idx = int(np.argmin(result.func_vals))
    best_seed = random_seed + best_idx
    final_env = gym.make(env_name)
    final_env.seed(best_seed)
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    random.seed(best_seed)

    final_results = train(
        env=final_env,
        num_episodes=500,
        gamma=best_gamma,
        tau=best_tau,
        noise_scale=best_noise_scale,
        lr_a=best_lr_a,
        lr_c=best_lr_c,
        updates_per_step=best_updates_per_step,
        render=True,       # Render the environment
        save_model=True    # Save the final model
    )
    
    # Save best hyperparameters to file
    with open(f"{output_dir}/best_hyperparameters.txt", "w") as f:
        f.write(f"gamma = {best_gamma}\n")
        f.write(f"tau = {best_tau}\n")
        f.write(f"noise_scale = {best_noise_scale}\n")
        f.write(f"lr_a = {best_lr_a}\n")
        f.write(f"lr_c = {best_lr_c}\n")
        f.write(f"updates_per_step = {best_updates_per_step}\n")
        f.write(f"random_seed = {best_seed}\n")
        f.write(f"Best reward: {-result.fun:.2f}\n")
    
    return result, final_results

if __name__ == '__main__':
    # Run optimization with 30 total evaluations, 10 random
    result, final_model = run_optimization(n_calls=100, n_random_starts=20)
    print("Optimization and visualization completed!")
