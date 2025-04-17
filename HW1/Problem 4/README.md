# REINFORCE with Baseline and GAE

## Environment Setup

### Option 1: Using Conda (Tested on Mac M2)

```bash
# Create and activate a new conda environment
conda create -n gae python=3.12 -y
conda activate gae

# Install dependencies
conda install gymnasium -y
pip install torch torchvision torchaudio
conda install tensorboard -y
pip install box2d pygame

conda create -n gae python=3.7 -y
# WSL2 with python=3.7 follows the d4rl env
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
conda install tensorboard -y
pip install box2d-py box2d pygame chardet
```

### Option 2: Using pip with requirements.txt

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the Code

```bash
python reinforce_gae_orth.py
```

## Using TensorBoard

Training metrics are automatically saved to `./tb_record_gae` directory. To view them:

```bash
# Navigate to the project root directory
cd 2025-Spring-RL/HW1/Problem\ 4

# Start TensorBoard
conda activate gae
tensorboard --logdir=./tb_record_gae

# Access in browser
# Open http://localhost:6006 in your web browser
```
