# setup.py
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

math_extras = [
    "transformers>=4.44.0,<4.46.0",
    "trl>=0.9.6,<0.10.0",
    "accelerate>=0.33.0,<0.34.0",
    "datasets>=2.18.0,<3.0.0",
    "sentencepiece>=0.1.99,<0.2.0",
    "sympy>=1.12,<1.13",
    "peft>=0.14.0,<0.15.0",
]

setup(
    name="ppo_from_scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"math": math_extras},
    python_requires=">=3.8",
)
