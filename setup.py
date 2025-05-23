from setuptools import find_packages, setup


setup(
    name="fastmlx",
    version="0.1.0",
    description="Lightweight MLX-based deep learning framework",
    packages=find_packages(include=["fastmlx*"]),
    install_requires=[
        "mlx",
        "mlx-data",
        "mlx-lm",
    ],
    python_requires=">=3.11",
)

