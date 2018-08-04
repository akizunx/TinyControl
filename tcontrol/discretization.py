from functools import singledispatch
import numbers
from typing import Tuple
import math

from .transferfunction import TransferFunction, ss2tf, _tf_to_symbol
from .statespace import StateSpace, tf2ss
import numpy as np
import sympy as sym

__all__ = ['c2d']


def _zoh(sys_: StateSpace, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    AT = sys_.A * sample_time
    AT_K = [np.eye(sys_.A.shape[0]), AT]
    for i in range(18):
        AT_K.append(AT_K[-1] * AT)

    G = 0
    for k in range(20):
        G += AT_K[k] / math.factorial(k)

    H = 0
    for k in range(20):
        H += AT_K[k] / math.factorial(k + 1)
    H *= sample_time
    H = H * sys_.B
    return G, H, sys_.C.copy(), sys_.D.copy()


def _tustin(sys_: TransferFunction, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    gs, *_ = _tf_to_symbol(sys_.num, sys_.den)
    s, z = sym.symbols('s z')
    tustin = (z - 1) / (z + 1) * 2 / sample_time
    gz = gs.replace(s, tustin)
    gz = sym.cancel(gz)
    num, den = gz.as_numer_denom()
    num = sym.Poly(num, z).all_coeffs()
    den = sym.Poly(den, z).all_coeffs()

    # convert from sympy numbers to numpy float
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return num, den


def _matched(sys_: TransferFunction, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    poles = sys_.pole()
    zeros = sys_.zero()
    num = np.poly(np.exp(zeros * sample_time))
    den = np.poly(np.exp(poles * sample_time))
    nump = np.poly1d(num)
    denp = np.poly1d(den)
    ds = np.polyval(sys_.num, 0) / np.polyval(sys_.den, 0)
    d1z = nump(1) / denp(1)
    dz_gain = ds / d1z
    return num * dz_gain, den


_methods = {'matched': _matched, 'Tustin': _tustin, 'tustin': _tustin,
            'zoh': _zoh}


@singledispatch
def c2d(sys_, sample_time, method='zoh'):
    """
    Convert a continuous system to a discrete system.

    :param sys_: the system to be transformed
    :type sys_: TransferFunction | StateSpace
    :param sample_time: sample time
    :type sample_time: numbers.Real
    :param method: method used in transformation(default: zoh)
    :type method: str
    :return: a discrete system
    :rtype: TransferFunction | StateSpace

    :raises TypeError: Raised when the type of sys_ is wrong

    :Example:

        >>> system = TransferFunction([1], [1, 1])
        >>> c2d(system, 0.5, 'matched')
             0.393469340287367
        -------------------------
        1.0*z - 0.606530659712633
        sample time:0.5s
    """
    raise TypeError(f'TransferFunction or StateSpace expected, got{type(sys_)}')


@c2d.register(TransferFunction)
def f(sys_, sample_time, method='zoh'):
    if method == 'zoh':
        sys_ = tf2ss(sys_)
    f = _methods[method]
    r = f(sys_, sample_time)
    if method == 'zoh':
        A, B, C, D = r
        return ss2tf(StateSpace(A, B, C, D, dt=sample_time))
    else:
        num, den = r
        return TransferFunction(num, den, dt=sample_time)


@c2d.register(StateSpace)
def f(sys_, sample_time, method='zoh'):
    f = _methods[method]
    A, B, C, D = f(sys_, sample_time)
    return StateSpace(A, B, C, D, dt=sample_time)
