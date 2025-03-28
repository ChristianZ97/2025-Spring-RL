# D4RL Setup Guide

## Table of Contents
- [Environment Setup](#environment-setup)
- [MuJoCo Installation](#mujoco-installation)
  - [Windows 10](#install-mujoco-on-windows-10)
  - [M2 Mac](#install-mujoco-on-m2-mac)
- [Environment Variables](#mujoco-environment-variables-setup)
  - [Windows 10](#for-windows-10)
  - [M2 Mac](#for-mac-m2)
- [Python Dependencies](#python-dependencies)
  - [Install mujoco-py](#install-mujoco-py)
  - [Install d4rl](#install-d4rl)
  - [Verification](#verify-installations)
  - [Additional Packages](#other-installations)
- [Windows-Specific Fixes](#windows-specific-error)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Environment Setup

### Create conda environment
```bash
# For Windows 10
conda create -n d4rl python=3.7 -y 

# For M2 Mac
conda create -n d4rl python=3.8 -y 

# Activate environment
conda activate d4rl
```

## MuJoCo Installation

### Install MuJoCo on Windows 10

**Git Bash**
```bash
mkdir -p ~/.mujoco
curl -L -O https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip
unzip mujoco210-windows-x86_64.zip -d ~/.mujoco
```

**PowerShell**
```powershell
New-Item -ItemType Directory -Force $env:USERPROFILE\.mujoco
Invoke-WebRequest -Uri "https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip" -OutFile "$env:USERPROFILE\mujoco210-windows-x86_64.zip"
Expand-Archive -Path "$env:USERPROFILE\mujoco210-windows-x86_64.zip" -DestinationPath "$env:USERPROFILE\.mujoco" -Force
```

**CMD**
```cmd
mkdir %USERPROFILE%\.mujoco
curl -L -o %USERPROFILE%\mujoco210-windows-x86_64.zip https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip
powershell -Command "Expand-Archive -Path '%USERPROFILE%\mujoco210-windows-x86_64.zip' -DestinationPath '%USERPROFILE%\.mujoco' -Force"
```

### Install MuJoCo on M2 Mac

1. Download and install MuJoCo.app for MacOS (the .dmg file) from [official release](https://github.com/google-deepmind/mujoco/releases)
2. Copy the MuJoCo.app into /Applications/ folder
3. Run the following commands:

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

## MuJoCo Environment Variables Setup

### For Windows 10

**Git Bash**
```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
export PATH="$HOME/.mujoco/mujoco210/bin:$PATH"
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Verify MuJoCo installation
~/.mujoco/mujoco210/bin/simulate ~/.mujoco/mujoco210/model/humanoid.xml
```

**PowerShell**
```powershell
# Set environment variables
$mujoco_bin = "$env:USERPROFILE\.mujoco\mujoco210\bin"
if ($env:LD_LIBRARY_PATH) {
    $env:LD_LIBRARY_PATH = "$mujoco_bin;$env:LD_LIBRARY_PATH"
} else {
    $env:LD_LIBRARY_PATH = "$mujoco_bin"
}

$env:MUJOCO_KEY_PATH = "$env:USERPROFILE\.mujoco"
$env:PATH = "$mujoco_bin;$env:PATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"

# Verify MuJoCo installation
Start-Process -NoNewWindow -FilePath "$env:USERPROFILE\.mujoco\mujoco210\bin\simulate.exe" -ArgumentList "$env:USERPROFILE\.mujoco\mujoco210\model\humanoid.xml"
```

**CMD**
```cmd
:: Set environment variables
if defined LD_LIBRARY_PATH (
  set "LD_LIBRARY_PATH=%USERPROFILE%\.mujoco\mujoco210\bin;%LD_LIBRARY_PATH%"
) else (
  set "LD_LIBRARY_PATH=%USERPROFILE%\.mujoco\mujoco210\bin"
)

set "MUJOCO_KEY_PATH=%USERPROFILE%\.mujoco"
set "PATH=%USERPROFILE%\.mujoco\mujoco210\bin;%PATH%"
set "D4RL_SUPPRESS_IMPORT_ERROR=1"

:: Verify MuJoCo installation
"%USERPROFILE%\.mujoco\mujoco210\bin\simulate.exe" "%USERPROFILE%\.mujoco\mujoco210\model\humanoid.xml"
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
# For M2 Mac, you'll need GCC 11
brew install gcc@11  # If not already installed
export CC=/opt/homebrew/bin/gcc-11

# To avoid cython compiler error
pip install -U cython==3.0.0a10

# Option 1: Install mujoco-py directly (may fail)
pip install git+https://github.com/openai/mujoco-py.git

# Option 2: Clone and install
git clone https://github.com/openai/mujoco-py.git && cd mujoco-py
pip install -e .
```

### Install d4rl
```bash
pip install absl-py matplotlib
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

### Verify installations
```bash
python -c "import mujoco_py; print(mujoco_py.__version__)"
python -c "import d4rl; print(d4rl.__version__)"

# Run sanity check
python d4rl_sanity_check.py
```

### PATH conflict fix (for Windows 10)
**Git Bash**
```bash
export PATH="$HOME/.mujoco/mujoco210/bin:$PATH"
```

**PowerShell**
```powershell
[System.Environment]::SetEnvironmentVariable("PATH", "$env:USERPROFILE\.mujoco\mujoco210\bin;$env:PATH", [System.EnvironmentVariableTarget]::User)
```

**CMD**
```cmd
setx PATH "%USERPROFILE%\.mujoco\mujoco210\bin;%PATH%"
```

### Other installations
```bash
# Make sure CUDA is available
conda install -c nvidia cuda-toolkit -y

# Install PyTorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install TensorBoard
conda install tensorboard -y
pip install chardet  # Fix error on Windows 10

# Fix for LunarLander on M2 Mac
conda install -c conda-forge box2d-py

# Fix for LunarLander on Windows 10
conda install swig -y
pip install box2d-py
pip install gym[box2d]
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

#### Automated Fix

**Identify D4RL path first**

**Git Bash**
```bash
D4RL_PATH=$(python -c "import d4rl; import os; print(os.path.dirname(d4rl.__file__))")
```

**PowerShell**
```powershell
$env:D4RL_PATH = python -c "import d4rl; import os; print(os.path.dirname(d4rl.__file__))"
```

**CMD**
```cmd
for /f "delims=" %%i in ('python -c "import d4rl; import os; print(os.path.dirname(d4rl.__file__))"') do set "D4RL_PATH=%%i"
```

**Backup original files**

**Git Bash**
```bash
cp $D4RL_PATH/locomotion/maze_env.py $D4RL_PATH/locomotion/maze_env.py.bak
cp $D4RL_PATH/pointmaze/dynamic_mjc.py $D4RL_PATH/pointmaze/dynamic_mjc.py.bak
```

**PowerShell**
```powershell
Copy-Item "$env:D4RL_PATH\locomotion\maze_env.py" -Destination "$env:D4RL_PATH\locomotion\maze_env.py.bak"
Copy-Item "$env:D4RL_PATH\pointmaze\dynamic_mjc.py" -Destination "$env:D4RL_PATH\pointmaze\dynamic_mjc.py.bak"
```

**CMD**
```cmd
copy "%D4RL_PATH%\locomotion\maze_env.py" "%D4RL_PATH%\locomotion\maze_env.py.bak"
copy "%D4RL_PATH%\pointmaze\dynamic_mjc.py" "%D4RL_PATH%\pointmaze\dynamic_mjc.py.bak"
```

**Apply fixes**

**Git Bash**
```bash
# Fix 1: maze_env.py
sed -i 's/_, file_path = tempfile.mkstemp(text=True, suffix='"'"'\.xml'"'"')/fd, file_path = tempfile.mkstemp(text=True, suffix='"'"'\.xml'"'"')/' $D4RL_PATH/locomotion/maze_env.py
sed -i 's/\([[:space:]]*\)tree.write(file_path)/\1tree.write(file_path)\n\1os.close(fd)/' $D4RL_PATH/locomotion/maze_env.py

# Fix 2: dynamic_mjc.py
sed -i 's/with tempfile.NamedTemporaryFile(mode='"'"'w+'"'"', suffix='"'"'\.xml'"'"', delete=True) as f:/with tempfile.NamedTemporaryFile(mode='"'"'w+'"'"', suffix='"'"'\.xml'"'"', delete=False) as f:/' $D4RL_PATH/pointmaze/dynamic_mjc.py
sed -i 's/\([[:space:]]*\)f.seek(0)/\1f.seek(0)\n\1f.close()/' $D4RL_PATH/pointmaze/dynamic_mjc.py
```

**PowerShell**
```powershell
# Fix 1: maze_env.py
(Get-Content "$env:D4RL_PATH\locomotion\maze_env.py") | 
    ForEach-Object {$_ -replace '_, file_path = tempfile\.mkstemp\(text=True, suffix=''\.xml''\)', 'fd, file_path = tempfile.mkstemp(text=True, suffix=''.xml'')'} |
    ForEach-Object {$_ -replace '([ \t]*)tree\.write\(file_path\)', '$1tree.write(file_path)\n$1os.close(fd)'} |
    Set-Content "$env:D4RL_PATH\locomotion\maze_env.py"

# Fix 2: dynamic_mjc.py
(Get-Content "$env:D4RL_PATH\pointmaze\dynamic_mjc.py") |
    ForEach-Object {$_ -replace 'with tempfile\.NamedTemporaryFile\(mode=''w\+'', suffix=''\.xml'', delete=True\)', 'with tempfile.NamedTemporaryFile(mode=''w+'', suffix=''.xml'', delete=False)'} |
    ForEach-Object {$_ -replace '([ \t]*)f\.seek\(0\)', '$1f.seek(0)\n$1f.close()'} |
    Set-Content "$env:D4RL_PATH\pointmaze\dynamic_mjc.py"
```

**CMD**
```cmd
:: It's recommended to use PowerShell for these text manipulations
:: This will execute the PowerShell script from CMD
powershell -Command ^
"(Get-Content '%D4RL_PATH%\locomotion\maze_env.py') | ForEach-Object {$_ -replace '_, file_path = tempfile\.mkstemp\(text=True, suffix=''\.xml''\)', 'fd, file_path = tempfile.mkstemp(text=True, suffix=''.xml'')'} | ForEach-Object {$_ -replace '([ \t]*)tree\.write\(file_path\)', '$1tree.write(file_path)`n$1os.close(fd)'} | Set-Content '%D4RL_PATH%\locomotion\maze_env.py'"

powershell -Command ^
"(Get-Content '%D4RL_PATH%\pointmaze\dynamic_mjc.py') | ForEach-Object {$_ -replace 'with tempfile\.NamedTemporaryFile\(mode=''w\+'', suffix=''\.xml'', delete=True\)', 'with tempfile.NamedTemporaryFile(mode=''w+'', suffix=''.xml'', delete=False)'} | ForEach-Object {$_ -replace '([ \t]*)f\.seek\(0\)', '$1f.seek(0)`n$1f.close()'} | Set-Content '%D4RL_PATH%\pointmaze\dynamic_mjc.py'"
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
- [Cython Compiler Error](https://github.com/openai/mujoco-py/issues/786) or [this one](https://github.com/openai/mujoco-py/issues/773) - Issue about the compiled error
