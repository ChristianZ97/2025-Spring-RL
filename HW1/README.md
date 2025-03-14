# D4RL Setup on M2 Mac (x86 via Rosetta)

This guide explains how to set up D4RL with MuJoCo in an x86_64 Conda environment on an M2 Mac.

## Create x86_64 Conda Environment

```bash
CONDA_SUBDIR=osx-64 conda create -n d4rl-x86 python=3.8 -y
conda activate d4rl-x86
conda config --env --set subdir osx-64
python -c "import platform; print(platform.machine())"
```

## Install MuJoCo

```bash
mkdir -p ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
tar -xzf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco
```

## Set Environment Variables

Add these lines to your `~/.zshrc` (or equivalent):

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

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
code $(find ~/miniconda3/envs/d4rl-x86 -name "builder.py" | grep mujoco_py)
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

## Cython.Compiler.Errors.CompileError
```bash
pip uninstall cython -y
pip install Cython==3.0.0a10
```

## distutils.errors.CompileError: command '/opt/homebrew/bin/gcc-14' failed with exit code 1
```bash
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c menpo glfw3 -y

export CPATH=$CONDA_PREFIX/include
source ~/.zshrc
conda activate d4rl-x86
```
