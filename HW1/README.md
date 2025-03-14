# D4RL Setup on M2 Mac (x86 via Rosetta)

This guide explains how to set up D4RL with MuJoCo in an x86_64 Conda environment on an M2 Mac.

## Create x86_64 Conda Environment

```bash
# Create a new Conda environment with x86_64 architecture
CONDA_SUBDIR=osx-64 conda create -n d4rl-x86 python=3.8 -y

# Activate the environment
conda activate d4rl-x86

# Set the environment to permanently use x86_64 architecture
conda config --env --set subdir osx-64

# Verify the architecture is x86_64
python -c "import platform; print(platform.machine())"
```

## Install MuJoCo

```bash
# Create MuJoCo directory
mkdir -p ~/.mujoco
cd ~/.mujoco

# Download MuJoCo 2.1.0 x86_64 version
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz

# Extract to MuJoCo directory
tar -xzf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco
```

## Set Environment Variables

Add these lines to your `~/.zshrc` (or equivalent):

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

Then run:
```bash
source ~/.zshrc
```

## Install D4RL

```bash
pip install gym d4rl
```

## Troubleshooting

When running `python d4rl_sanity_check.py`, the following error occurred:
```
RuntimeError: Could not find supported GCC executable.
```

This error suggests GCC is needed for MuJoCo compilation. Install it with:
```bash
brew install gcc
```
