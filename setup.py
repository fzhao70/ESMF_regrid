"""Setup script for ESMF_regrid package."""

from setuptools import setup, find_packages
import os


def read_file(filename):
    """Read file contents."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


# Read README
readme = read_file("README.md") if os.path.exists("README.md") else ""

# Read requirements
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="esmf-regrid",
    version="1.0.0",
    author="ESMF_regrid Contributors",
    description="A Python implementation of NCL's ESMF_regrid function",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/esmf_regrid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
