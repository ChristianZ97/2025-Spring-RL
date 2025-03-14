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
conda activate d4rl-x86
```

## Install D4RL

```bash
pip install gym
conda install -c conda-forge pybullet -y
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

## Compiler Setup (Required for MuJoCo)

For MuJoCo to work properly, you'll need to address the GCC requirement:

### Option 1: Install Latest GCC

```bash
# Install latest GCC version
brew install gcc

# Check installed version
which gcc
gcc --version

# Add to your ~/.zshrc file
export CC=/opt/homebrew/bin/gcc-14
export CXX=/opt/homebrew/bin/g++-14

# Apply changes
source ~/.zshrc
conda activate d4rl-x86
```

### Option 2: Modify MuJoCo Source (Recommended for M2 Macs)

Since installing GCC for x86_64 on M2 Macs is problematic, you can modify the MuJoCo source code:

```bash
# Find the builder.py file
find ~/miniconda3/envs/d4rl-x86 -name "builder.py" | grep mujoco_py

# Edit the file using VS Code
code [PATH_TO_BUILDER_PY]
```

Find the code around line 333 that checks for GCC and modify it to use clang instead:

```python
# Original code:
# if c_compiler is None:
#     raise RuntimeError(
#         "Could not find supported GCC executable.\n\n"
#         "HINT: On OS X, install GCC 9.x with `brew install gcc@9`. "
#         "or `port install gcc9`.")

# Modified code:
if c_compiler is None:
    print("Warning: Using default compiler instead of GCC")
    c_compiler = "clang"  # Use macOS default clang compiler
```

Save the file and try running your code again.

## Testing Your Installation

To test your installation, simply run:

```bash
python d4rl_sanity_check.py
```

If it runs without errors, your D4RL setup is working correctly.
