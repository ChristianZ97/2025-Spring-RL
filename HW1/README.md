# D4RL Setup Guide

## Environment Setup

### Create conda environment
```bash
conda create -n d4rl python=3.7 -y # for Windows 10
conda create -n d4rl python=3.8 -y # for Mac M2
conda activate d4rl
```

## MuJoCo Installation

### Install MuJoCo on Windows 10
```bash
mkdir -p ~/.mujoco
curl -L -O https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip
unzip mujoco210-windows-x86_64.zip -d ~/.mujoco
```

### Install MuJoCo on M2 Mac
- First, download and install MuJoCo.app for MacOS (the .dmg file) from [official release](https://github.com/google-deepmind/mujoco/releases).
- Second, copy the MuJoCo.app into /Applications/ folder, then

```bash
mkdir -p $HOME/.mujoco/mujoco210/bin
mkdir -p $HOME/.mujoco/mujoco210/include

# Create symbolic links to MuJoCo files
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include/
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib

# Install GLFW and links
brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin/
```

## MuJoCo environment variables setup

### For Windows 10
```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
export PATH="$HOME/.mujoco/mujoco210/bin:$PATH"
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Verify MuJoCo installation
~/.mujoco/mujoco210/bin/simulate ~/.mujoco/mujoco210/model/humanoid.xml
```

### For Mac M2 
```bash
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export PATH="$HOME/.mujoco/mujoco210/bin:$PATH"
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

## Python Dependencies

### Install mujoco-py
```bash
# Install mujoco-py
git clone https://github.com/openai/mujoco-py.git && cd mujoco-py

# For M2 Mac, you'll need GCC 11
brew install gcc@11  # If not already installed
export CC=/opt/homebrew/bin/gcc-11

pip install -e .
pip install -U cython==3.0.0a10
```

### Install d4rl
```bash
pip install absl-py matplotlib
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

# For Windows 10, you may need to fix the PATH Conflict between Git Bash and Conda.
export PATH="$HOME/.mujoco/mujoco210/bin:$PATH"
```

### Verify installations
```bash
python -c "import mujoco_py; print(mujoco_py.__version__)"
python -c "import d4rl; print(d4rl.__version__)"
```

Run sanity check to ensure everything is working:
```bash
python d4rl_sanity_check.py
```

## Windows-Specific Error

### Fix for maze2d-umaze-v1 error in Windows 10
```
Exception: Failed to load XML file
```

#### Fix 1: Modify `locomotion/maze_env.py`
```python
# Find:
_, file_path = tempfile.mkstemp(text=True, suffix='.xml')
tree.write(file_path)

# Replace with:
fd, file_path = tempfile.mkstemp(text=True, suffix='.xml')
tree.write(file_path)
os.close(fd)
```

#### Fix 2: Modify `pointmaze/dynamic_mjc.py`
```python
# Find:
with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as f:
    self.root.write(f)
    f.seek(0)

# Replace with:
with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=False) as f:
    self.root.write(f)
    f.seek(0)
    f.close()
```

#### Quick Fix Commands
```bash
# Get D4RL path
D4RL_PATH=$(python -c "import d4rl; import os; print(os.path.dirname(d4rl.__file__))")

# Backup original files
cp $D4RL_PATH/locomotion/maze_env.py $D4RL_PATH/locomotion/maze_env.py.bak
cp $D4RL_PATH/pointmaze/dynamic_mjc.py $D4RL_PATH/pointmaze/dynamic_mjc.py.bak

# Fix 1: maze_env.py
sed -i 's/_, file_path = tempfile.mkstemp(text=True, suffix='"'"'\.xml'"'"')/fd, file_path = tempfile.mkstemp(text=True, suffix='"'"'\.xml'"'"')/' $D4RL_PATH/locomotion/maze_env.py
sed -i 's/\([[:space:]]*\)tree.write(file_path)/\1tree.write(file_path)\n\1os.close(fd)/' $D4RL_PATH/locomotion/maze_env.py

# Fix 2: dynamic_mjc.py
sed -i 's/with tempfile.NamedTemporaryFile(mode='"'"'w+'"'"', suffix='"'"'\.xml'"'"', delete=True) as f:/with tempfile.NamedTemporaryFile(mode='"'"'w+'"'"', suffix='"'"'\.xml'"'"', delete=False) as f:/' $D4RL_PATH/pointmaze/dynamic_mjc.py
sed -i 's/\([[:space:]]*\)f.seek(0)/\1f.seek(0)\n\1f.close()/' $D4RL_PATH/pointmaze/dynamic_mjc.py
```

## Troubleshooting

1. **Missing Dependencies**:
   - Windows: Ensure Visual C++ redistributables are installed
   - Mac: Ensure XCode command line tools are installed (`xcode-select --install`)

2. **Environment Variable Issues**:
   - Add environment variables to your shell's configuration file (.bashrc, .zshrc) for persistence

3. **Compiler Errors on Mac M2**:
   - Make sure to use GCC 11 for M1/M2 compatibility
   - If you encounter GLFW errors, verify the symbolic link to libglfw.3.dylib is correct

4. **Import Errors**:
   - Double-check that D4RL_SUPPRESS_IMPORT_ERROR is set
   - Verify that mujoco-py is installed correctly

5. **Windows File Handling Errors**:
   - If you encounter XML file loading issues, apply the fixes for maze_env.py and dynamic_mjc.py

## References
- [D4RL Installation on Zhihu](https://zhuanlan.zhihu.com/p/434073300) - Original Windows installation guide
- [Installation Guide for M1 Mac](https://github.com/openai/mujoco-py/issues/682) - Solutions for Apple Silicon
- [Maze2D and AntMaze Issue on Windows](https://github.com/Farama-Foundation/D4RL/pull/148/commits/dcb5695f1d8919301f7c92a9c710c86d048e64fa) - Fixes for XML file handling in Windows
