import sys

from setuptools import setup, find_packages

if sys.version_info[: 2] < (3, 6):
    raise RuntimeError('Python version >= 3.6 required')

MAJOR = 0
MINOR = 8
MICRO = 1
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'

PACKAGES = ['tcontrol'] + [f"tcontrol.{i}" for i in find_packages('tcontrol')]

setup(
    name='TinyControl',
    version=VERSION,
    author='TinyControl Developers',
    packages=PACKAGES,
    license='BSD 3-clause',
    install_requires=['sympy>=1.0', 'numpy>=1.12.0', 'matplotlib>=2.0.0', 'scipy>=1.1.0'],
    tests_requires=['nose2>=0.7.4'],
    extra_requires={
        'doc': ['sphinx>=1.6.6', 'sphinx-rtd-theme>=0.4.0']
    }
)
