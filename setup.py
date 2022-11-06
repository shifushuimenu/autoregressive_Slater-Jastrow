"""
Autoregressive Slater-Jastrow VMC in orbital space.
"""
import setuptools
from distutils.core import setup 

def setup_autoregressive_SJVMC():
    setup(
        name='autoregressive_SJVMC',
        maintainer='S.H.',
        maintainer_email='@gmail.com',
        packages=['autoregressive_SJVMC'],
        install_requires=[
            "scipy",
            "numpy",
            "torch",
            "h5py"
            ]
    )

if __name__ == "__main__":
    setup_autoregressive_SJVMC()
