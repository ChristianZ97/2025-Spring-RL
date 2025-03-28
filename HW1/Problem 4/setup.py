from setuptools import setup, find_packages

setup(
    name="reinforce_gae",
    version="0.1.0",
    description="REINFORCE with baseline and GAE implementation",
    author="Student",
    author_email="student@university.edu",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "box2d-py>=2.3.5",
        "pygame>=2.1.0",
        "tensorboard>=2.12.0",
        "protobuf==3.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)