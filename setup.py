from setuptools import setup, find_packages

setup(
    name="agentic-weather-rl",
    version="1.0.0",
    description="Risk-Aware Multi-Agent Reinforcement Learning for Cloudburst Disaster Response",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ali Karma",
    url="https://github.com/aliakarma/agentic-weather-rl",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23",
        "matplotlib>=3.5",
        "scipy>=1.9",
    ],
    extras_require={
        "gpu": ["torch>=2.0"],
        "dev": ["pytest>=7.0", "jupyter>=1.0", "tensorboard>=2.10"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "multi-agent reinforcement learning",
        "disaster response",
        "safe reinforcement learning",
        "Lagrangian optimization",
        "weather",
        "emergency management",
    ],
)
