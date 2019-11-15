from functools import singledispatch
import numbers
import warnings
from typing import Union, Tuple

import numpy as np
from numpy.linalg import eigvals

from .discretization import c2d
from .model_conversion import tf2ss
from .plot_utility import plot_response_curve
from .statespace import StateSpace
from .transferfunction import TransferFunction

__all__ = ['impulse', 'step', 'ramp', 'any_input']


# convert system to state space then get result
def _any_input(sys_, t, input_signal, init_cond=None):
    """
    Accept any input signal, then calculate the response of the system.

    :param sys_: the system
    :type sys_: TransferFunction | StateSpace
    :param t: time
    :type t: np.ndarray
    :param input_signal: input signal accepted by the system
    :type input_signal: np.ndarray
    :param init_cond: initial condition of the system
    :type init_cond: None | numbers.Real | np.ndarray

    :return: system output and time array
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises TypeError: when give wrong types of arguments
    :raises ValueError: raised when t, input_signal or init_cond has a wrong shape
    :raises NotImplementedError: when system is a MIMO system

    .. note:: This is internal api.
    """
    # convert transfer function or continuous system to discrete system
    dt = t[1] - t[0]
    if dt > 1 and sys_.is_ctime:
        warnings.warn("Large sample time will lead to low accuracy.",
                      stacklevel=3)

    if sys_.is_ctime:
        d_sys_ = c2d(sys_, dt)
    else:
        if _is_dt_validated(sys_, dt):
            d_sys_ = sys_
        else:
            raise ValueError('The step of time vector didn\'t match the sample time of '
                             'the system.')

    u = input_signal

    if init_cond is None:
        init_cond = np.zeros((d_sys_.A.shape[0], 1))
    else:
        # check the shape of init_cond
        if init_cond.shape[0] != d_sys_.A.shape[0] or init_cond.shape[1] != 1:
            raise ValueError("wrong dimension of init condition")

    x = _cal_x(d_sys_.A, d_sys_.B, t.size, init_cond, u)
    y = _cal_y(d_sys_.C, d_sys_.D, x.shape[1], x, u)

    if sys_.is_siso:
        return y.reshape(-1), t
    else:
        y = [np.asarray(_).reshape(-1) for _ in y]
    return np.array(y), t


def _is_dt_validated(sys_: Union[TransferFunction, StateSpace],
                     dt: Union[int, float]) -> bool:
    if abs(dt - sys_.dt) <= 1e-7:
        return True
    else:
        return False


def _cal_x(G, H, n, x_0, u):
    """
    calculate x step by step
    """
    x = np.empty((G.shape[0], n))
    x[:, 0: 1] = x_0
    for i in range(n - 1):
        x_k = G @ x[:, i: i + 1] + H @ u[:, i: i + 1]
        x[:, i + 1: i + 2] = x_k
    return x


def _cal_y(C, D, n, x, u):
    """
    calculate system output
    """
    y = np.empty((C.shape[0], n))
    for i in range(n):
        y_k = C @ x[:, i: i + 1] + D @ u[:, i: i + 1]
        y[:, i: i + 1] = y_k
    return y


def step(sys_, t=None, *, plot=True):
    """
    step response of the system

    .. seealso:: any_input
    """
    if isinstance(sys_, TransferFunction):
        sys_ = tf2ss(sys_)

    if t is None:
        t = _setup_time_vector(sys_)

    u = np.ones(t.shape, dtype=int)
    y, t = any_input(sys_, t, u, plot=False)
    if plot:
        plot_response_curve(y, t, "step response", sys_.is_ctime)
    return y, t


def impulse(sys_, t=None, *, plot=True, **kwargs):
    """
    impulse response of the system

    .. seealso:: any_input
    """
    if isinstance(sys_, TransferFunction):
        sys_ = tf2ss(sys_)

    if t is None:
        t = _setup_time_vector(sys_)

    u = np.zeros(t.shape)
    x0 = kwargs.get('x0')
    K = kwargs.get('K', 1)
    if not sys_.is_ctime:
        u[0] = 1
    else:
        x0 = sys_.B * K if x0 is None else x0 + sys_.B * K

    y, t = any_input(sys_, t, u, x0, plot=False)
    if plot:
        plot_response_curve(y, t, "impulse response", sys_.is_ctime)
    return y, t


def ramp(sys_, t=None, *, plot=True):
    """
    ramp response of the system

    .. seealso:: any_input
    """
    if isinstance(sys_, TransferFunction):
        sys_ = tf2ss(sys_)

    if t is None:
        t = _setup_time_vector(sys_)

    u = t
    y, t = any_input(sys_, t, u, plot=False)
    if plot:
        plot_response_curve(y, t, "impulse response", sys_.is_ctime)
    return y, t


def any_input(sys_, t, input_signal=0, init_cond=None, *, plot=True):
    """
    Accept any input signal, then calculate the response of the system.

    :param sys_: the system
    :type sys_: TransferFunction | StateSpace
    :param t: time
    :type t: array_like
    :param input_signal: input signal accepted by the system
    :type input_signal: numbers.Real | np.ndarray
    :param init_cond: initial condition of the system
    :type init_cond: None | numbers.Real | np.ndarray
    :param plot: If plot is True, it will show the response curve.
    :type plot: bool

    :return: system output and time array
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if isinstance(sys_, TransferFunction):
        sys_ = tf2ss(sys_)

    u = _setup_input_signal(input_signal, t, sys_.inputs)
    y, t = _any_input(sys_, t, u, init_cond)
    if plot:
        plot_response_curve(y, t, "response", sys_.is_ctime)
    return y, t


@singledispatch
def _setup_input_signal(input_signal, t, input_number):
    raise TypeError


@_setup_input_signal.register(np.ndarray)
def fn(input_signal, t, input_number):
    input_signal = np.atleast_2d(input_signal)
    if input_signal.shape[1] != t.size:
        raise ValueError('The input signal length doesn\'t match the time series.')

    if input_number > input_signal.shape[0] == 1:
        return np.repeat(input_signal, input_number, axis=0)
    elif input_number == input_signal.shape[0]:
        return input_signal
    else:
        raise ValueError('The input signal doesn\'t match the input channel number.')


@_setup_input_signal.register(numbers.Real)
def fn(input_signal, t, input_number):
    input_signal = np.zeros((1, t.size)) + input_signal
    return np.repeat(input_signal, input_number, axis=0)


def _setup_time_vector(sys_: StateSpace, n=1000):
    ev = eigvals(sys_.A)
    tc = 1 / np.min(np.abs(ev)) * 2
    if tc == np.inf:
        tc = 1
    if sys_.is_ctime:
        return np.linspace(0, 10 * tc, n)
    else:
        return np.arange(0, 10 * sys_.dt + 1, sys_.dt)
