"""
CatalystGNN: A Graph Neural Network Package for Catalyst Property Prediction

This package provides tools for featurizing molecular and crystal structures
into graphs and predicting catalytic properties using pre-trained GNN models.

Author: Isaac U. Adeyeye
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="catalystgnn",
    version="0.1.0",
    author="Isaac U. Adeyeye",
    author_email="isaac.adeyeye@example.com",
    description="Graph Neural Networks for Catalyst Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isaacuwana/CatalystGNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "catalystgnn=catalystgnn.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "catalystgnn": ["models/*.pth", "data/*.json"],
    },
)