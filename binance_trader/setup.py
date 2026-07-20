from setuptools import setup, find_packages

setup(
    name="binance_trader",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
)
