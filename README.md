# TinyControl
A Python lib for automatic control system analysis.

__IMPORTANT__ 

This project is quite __UNSTABLE__, __Do Not Use in Any Serious Work!__

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
    + controllability
    + observability
    + pole placement
    + Lyapunov stability

__NOTICE__ 
+ Two classes, TransferFunction and StateSpace, are designed for linear time-invariant system.
+ Two response methods, _time response_ and _frequency response_ are only support the SISO system now.

## Dependency
+ [Python 3.6+](https://www.python.org)
+ [Numpy](https://www.numpy.org)
+ [Scipy](https://scipy.org/)
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
    A:     B:
      [-1.]  [1.]
    C:     D:
      [ 1.]  [0.]


## License
This project is under the BSD-3-Clause License. For more information, see the file
[LICENSE](https://github.com/akizunx/TinyControl/blob/master/LICENSE).

## Contributions
Your contributions are welcome anytime.
