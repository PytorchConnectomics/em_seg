"""Package installation"""

from setuptools import setup, find_packages


def setup_package():
    """Required packages"""
    __version__ = "0.1"
    url = "https://github.com/PytorchConnectomics/seglib"
    setup(
        name="erl",
        description="Useful functions for segmentation",
        version=__version__,
        url=url,
        license="MIT",
        author="Donglai Wei",
        install_requires=["scipy", "numpy", "h5py", "imageio", "argparse"],
        packages=find_packages(),
    )


if __name__ == "__main__":
    # pip install --editable .
    setup_package()
