from setuptools import setup, find_packages

setup(
    name="arcchallenge",
    version="1.0.0",
    description="ARC Challenge Solver",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
    ],
    python_requires=">=3.8",
)
