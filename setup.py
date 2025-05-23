from pathlib import Path
from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_file = Path(__file__).parent / "requirements.txt"
    return [line.strip() for line in req_file.read_text().splitlines() if line.strip()]


setup(
    name="fastmlx",
    version="0.1.0",
    description="Lightweight MLX-based deep learning framework",
    packages=find_packages(include=["fastmlx*"]),
    install_requires=read_requirements(),
    python_requires=">=3.12",
)

