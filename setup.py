"""
QuantiaMagica - ADC Behavioral Event-Driven Simulator

Setup script for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantiamagica",
    version="1.0.0",
    author="KonataLin",
    author_email="2424441676@qq.com",
    description="ADC Behavioral Event-Driven Simulator with Bukkit-style API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KonataLin/QuantiaMagica",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords=[
        "adc",
        "analog-to-digital",
        "converter",
        "simulation",
        "sar",
        "pipeline",
        "event-driven",
        "behavioral",
    ],
    project_urls={
        "Bug Reports": "https://github.com/KonataLin/QuantiaMagica/issues",
        "Source": "https://github.com/KonataLin/QuantiaMagica",
        "Documentation": "https://github.com/KonataLin/QuantiaMagica/docs",
    },
)
