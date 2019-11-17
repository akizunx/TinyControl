import math
import numbers
from functools import singledispatch
from typing import Tuple

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

from .exception import WrongSampleTime, UnknownDiscretizationMethod
from .model_conversion import *
from .statespace import StateSpace
from .transferfunction import TransferFunction

__all__ = ['c2d']


def _zoh(sys_: StateSpace, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    AT = sys_.A * sample_time
    AT_K = [np.eye(sys_.A.shape[0]), AT]
    for i in range(18):
        AT_K.append(AT_K[-1] @ AT)

    G = np.zeros_like(AT)
    for k in range(20):
        G += AT_K[k] / math.factorial(k)

    H = np.zeros_like(AT)
    for k in range(20):
        H += AT_K[k] / math.factorial(k + 1)
    H *= sample_time
    H = H @ sys_.B
    return G, H, sys_.C.copy(), sys_.D.copy()


def _foh(sys_: StateSpace, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    n = sys_.A.shape[0]
    p = sys_.inputs
    u = np.zeros((2, n + p + 1))
    u[0, -1] = 1 / sample_time
    F = np.block([[sys_.A, sys_.B, np.zeros((n, 1))],[u]])
    J = expm(F * sample_time)
    G = J[0: n, 0: n]
    Gamma1 = J[0: n, n: n + p]
    Gamma2 = J[0: n, n + p:]
    H = Gamma1 + G @ Gamma2 - Gamma2
    D = sys_.D + sys_.C @ Gamma2
    return G, H, sys_.C, D


def _tustin(sys_: StateSpace, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    alpha = 2 / sample_time
    eye = np.eye(sys_.A.shape[0])
    P = eye - 1 / alpha * sys_.A
    Q = eye + 1 / alpha * sys_.A
    P_I = inv(P)
    A = P_I @ Q
    B = P_I @ sys_.B
    C = 2 / alpha * sys_.C @ P_I
    D = sys_.D + sys_.C @ B / alpha
    return A, B, C, D


def _matched(sys_: TransferFunction, sample_time: numbers.Real) -> Tuple[np.ndarray, ...]:
    poles = sys_.pole()
    zeros = sys_.zero()
    num = np.poly(np.exp(zeros * sample_time))
    den = np.poly(np.exp(poles * sample_time))
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    root_number_delta = np.roots(den).shape[0] - np.roots(num).shape[0]
    while root_number_delta > 0:
        num = np.polymul(num, np.array([1, 1]))
        root_number_delta -= 1
    nump = np.poly1d(num)
    denp = np.poly1d(den)
    ds = np.polyval(sys_.num, 0) / np.polyval(sys_.den, 0)
    d1z = nump(1) / denp(1)
    dz_gain = ds / d1z
    return num * dz_gain, den


_methods = {'matched': _matched, 'Tustin': _tustin, 'tustin': _tustin,
            'bilinear': _tustin, 'zoh': _zoh, 'foh': _foh}


@singledispatch
def c2d(sys_, sample_time, method='zoh'):
    """
    Convert a continuous system to a discrete system.

    :param sys_: the system to be transformed
    :type sys_: TransferFunction | StateSpace
    :param sample_time: sample time
    :type sample_time: numbers.Real
    :param method: method used in transformation(default: zoh)
    method list:
        matched
        tustin
        bilinear
        zoh
        foh
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
def fn(sys_, sample_time, method='zoh'):
    if sample_time <= 0:
        raise WrongSampleTime(f'The sample time must be larger than 0. got {sample_time}')

    try:
        f = _methods[method]
    except KeyError as e:
        raise UnknownDiscretizationMethod from e

    if method != 'matched':
        A, B, C, D = f(tf2ss(sys_), sample_time)
        return ss2tf(StateSpace(A, B, C, D, dt=sample_time))
    else:
        num, den = f(sys_, sample_time)
        return TransferFunction(num, den, dt=sample_time)


@c2d.register(StateSpace)
def fn(sys_, sample_time, method='zoh'):
    if sample_time <= 0:
        raise WrongSampleTime(f'The sample time must be larger than 0. got {sample_time}')

    try:
        f = _methods[method]
    except KeyError as e:
        raise UnknownDiscretizationMethod from e

    A, B, C, D = f(sys_, sample_time)
    return StateSpace(A, B, C, D, dt=sample_time)
