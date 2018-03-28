from setuptools import find_packages
from distutils.core import setup

setup(
    name='pygrn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'atari-py',
        'certifi',
        'flake8',
        'gym',
        'h5py',
        'ipython',
        'Keras',
        'keras-rl',
        'matplotlib',
        'numpy',
        'Pillow',
        'pybullet',
        'pytest',
        'PyYAML',
        'scipy',
        'sklearn',
        'tensorflow',
        'tqdm'],
    license='',
    long_description=open('README.md').read(),
)
