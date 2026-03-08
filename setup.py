"""
setup.py
Risk-Aware MARL for Cloudburst Disaster Response

Installs the `src` package so all internal imports resolve cleanly
regardless of the working directory.

Usage:
    pip install -e .          # editable install (recommended for development)
    pip install .             # standard install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
_here = Path(__file__).parent
long_description = (_here / "README.md").read_text(encoding="utf-8")

setup(
    name="risk-aware-marl-cloudburst",
    version="1.0.0",
    description=(
        "Constrained Multi-Agent Reinforcement Learning for Coordinated "
        "Cloudburst Disaster Response"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The Authors",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "torch==2.1.0",
        "torchvision==0.16.0",
        "timm==0.9.12",
        "stable-baselines3==2.2.1",
        "gymnasium==0.29.1",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "h5py",
        "netCDF4",
        "tensorboard",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
)
