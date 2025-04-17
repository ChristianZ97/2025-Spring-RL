# -*- coding: utf-8 -*-
"""
setup.py for your RL project

Hardware Configuration:
- GPU: NVIDIA GeForce RTX 4060
- CUDA Version: 12.7
- Operating System: WSL (Windows Subsystem for Linux) on Windows 11
"""

from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    description="Reinforcement Learning project with GAE and Orthogonal Initialization",
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1+cu117',
        'torchvision==0.14.1+cu117',
        'torchaudio==0.13.1',
        'tensorboard==2.10.0',
        'box2d-py',
        'box2d',
        'pygame',
        'chardet',
    ],
    python_requires='>=3.7, <3.8',
)
