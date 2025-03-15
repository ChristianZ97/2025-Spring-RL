# D4RL Setup

```bash
# Create conda environment
conda create -n d4rl python=3.7 -y
conda activate d4rl

# Install MuJoCo
mkdir -p ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
tar -xzf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Add these exports to your .bashrc or .zshrc for persistence
source ~/.bashrc

# Install mujoco-py
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
pip install -U cython==3.0.0a10

# Verify installation
python -c "import mujoco_py; print(mujoco_py.__version__)"

# Install D4RL
pip install absl-py
pip install matplotlib
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

# Install Gym
git clone https://github.com/openai/gym.git
cd gym
python setup.py install

# Run
python d4rl_sanity_check.py

Exception: Failed to load XML file: C:\Users\chris\AppData\Local\Temp\tmp8xwm6brj.xml. mj_loadXML error: b'XML parse error at line 0, column 0:\nFailed to open file\n'
```
