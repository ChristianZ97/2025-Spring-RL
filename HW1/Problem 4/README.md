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
```

### Option 2: Using pip with requirements.txt

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the Code

```bash
python reinforce_gae_orth_init.py
```
