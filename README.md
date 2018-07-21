# TinyControl
TinyControl is a python lib for automatic control system analysis,
but now only a few features have been implemented.

## Features
+ Support transfer function and state space models.
+ Time response
    + impulse
    + step
    + ramp
    + any input
+ Frequency response
    + nyquist
    + bode
+ Control system analysis
    + Controllability
    + Observability
    + ...

__NOTICE__ 
+ Transfer Function and State Space are designed for linear time-invariant system.
+ Two response methods, _time response_ and _frequency response_ are only support the SISO system now.

## Dependency
+ [Python 3](https://www.python.org)
+ [Numpy](http://www.numpy.org)
+ [Sympy](http://www.sympy.org)
+ [Matplotlib](https://matplotlib.org)
+ [nose2](https://github.com/nose-devs/nose2) (optional for test)
+ [Sphinx](http://www.sphinx-doc.org) (optional for building docs)
+ [sphinx-rtd-theme](https://github.com/rtfd/sphinx_rtd_theme) (optional sphinx theme)

## Installation
    python -m setup.py install

## Usage
    >>> import tcontrol as tc
    >>> system = tc.tf([1], [1, 1])
    >>> print(system)
       1
    -----
    s + 1
    >>> tc.tf2ss(system)
    A:
    [[-1.]]

    B:
    [[1.]]

    C:
    [[1.]]

    D:
    [[0.]]

## Contributions
Your contributions are welcome anytime.
