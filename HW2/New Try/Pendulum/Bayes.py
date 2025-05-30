"""
optimize_ddpg.py - Script for optimizing DDPG hyperparameters using Bayesian Optimization.

This script uses scikit-optimize to perform Bayesian Optimization on the DDPG
algorithm's hyperparameters to maximize performance on reinforcement learning tasks.
"""

import sys
import gymnasium as gym
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

import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import gc

from itertools import count
counter = count(start=0)

from main import random_seed, env_name, main
from utils import set_seed_and_env, set_seed

# Define the hyperparameter search space
search_space = [
    Real(0.9996, 0.99999, name='gamma'),
    Real(0.02, 0.03, name='tau'),
    Real(1.4, 1.6, name='noise_scale'),
    Real(5e-4, 1.5e-3, name='lr_a'),
    Real(2.5e-3, 3.5e-3, name='lr_c'),
    Categorical([32, 64, 128, 256, 512], name='batch_size')
]

# Define the objective function for Bayesian Optimization
@use_named_args(search_space)
def objective(gamma, tau, noise_scale, lr_a, lr_c, batch_size):
    """
    Objective function for Bayesian Optimization.
    Runs DDPG with given hyperparameters and returns negative reward for minimization.
    """
    print(f"\nTrying parameters: gamma={gamma:.3e} tau={tau:.3e} noise_scale={noise_scale:.3e} lr_a={lr_a:.3e} lr_c={lr_c:.3e} batch_size={batch_size}")
    
    global counter
    bo_step = next(counter)
    bo_seed = random_seed + bo_step

    env = set_seed_and_env(bo_seed, env_name)

    start_time = time.time()
    writer = SummaryWriter(f"./tb_record_pendulum/seed={bo_seed}")

    results = main(
        env=env,
        gamma=gamma,
        tau=tau,
        noise_scale=noise_scale,
        lr_a=lr_a,
        lr_c=lr_c,
        batch_size=batch_size,
        num_episodes=500, # Use fewer episodes for optimization to save time
        save_model=False,  # Don't save models during optimization
        writer=writer
    )
    
    duration = time.time() - start_time

    final_rewards = results['ewma_reward']
    recent_rewards = final_rewards[-100:]

    x = np.arange(len(recent_rewards))
    momentum, _ = np.polyfit(x, recent_rewards, 1)

    final_mean = np.mean(recent_rewards)
    score = final_mean + 0.3 * momentum

    print(f"Training done in {duration:.1f}s | Mean reward: {final_mean:.2f} | Momentum: {momentum:.2f} | Score: {score:.2f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    
    return -score

def run_optimization(n_calls=20, n_random_starts=5, output_dir='optimization_results'):
    """
    Run the Bayesian Optimization process to find optimal DDPG hyperparameters.
    """
    print("\nStarting Bayesian Optimization process...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    set_seed(random_seed)
    
    # Run Bayesian Optimization
    result = gp_minimize(
        objective,                # Objective function to minimize
        search_space,             # Hyperparameter search space
        n_calls=n_calls,          # Total number of evaluations
        n_random_starts=n_random_starts,  # Initial random evaluations
        verbose=True,             # Print progress
        random_state=random_seed, # Random seed
        n_jobs=-1
    )
    
    # Extract best hyperparameters
    best_gamma, best_tau, best_noise_scale, best_lr_a, best_lr_c, best_batch_size = result.x
    
    print("\nOptimization completed!")
    print("Best hyperparameters:")
    for name, value in zip(['gamma', 'tau', 'noise_scale', 'lr_a', 'lr_c', 'batch_size'], result.x):
        print(f"{name}: {value}")
    
    print(f"Best reward: {-result.fun:.2f}")
    
    # Save optimization history
    with open(f"{output_dir}/optimization_history.txt", "w") as f:
        f.write("Iteration, Objective Value, Parameters\n")
        for i, (value, params) in enumerate(zip(result.func_vals, result.x_iters)):
            param_str = ", ".join([f"{name}={value}" for name, value in 
                                  zip(['gamma', 'tau', 'noise_scale', 'lr_a', 'lr_c', 'batch_size'], params)])
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
    
    final_env = set_seed_and_env(best_seed, env_name)
    final_results = main(
        env=final_env,
        gamma=best_gamma,
        tau=best_tau,
        noise_scale=best_noise_scale,
        lr_a=best_lr_a,
        lr_c=best_lr_c,
        batch_size=best_batch_size,
        num_episodes=4000,
        render=False,
        save_model=True
    )
    
    # Save best hyperparameters to file
    with open(f"{output_dir}/best_hyperparameters.txt", "w") as f:
        f.write(f"gamma = {best_gamma}\n")
        f.write(f"tau = {best_tau}\n")
        f.write(f"noise_scale = {best_noise_scale}\n")
        f.write(f"lr_a = {best_lr_a}\n")
        f.write(f"lr_c = {best_lr_c}\n")
        f.write(f"batch_size = {best_batch_size}\n")
        f.write(f"random_seed = {best_seed}\n")
        f.write(f"Best reward: {-result.fun:.2f}\n")
    
    return result, final_results

if __name__ == '__main__':
    # Run optimization with 30 total evaluations, 10 random
    result, final_model = run_optimization(n_calls=200, n_random_starts=50)
    print("Optimization and visualization completed!\n")
