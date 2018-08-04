from functools import partial, singledispatch
import warnings
import numbers
from typing import Union, Tuple, List

from .transferfunction import TransferFunction, _tf_to_symbol
from .statespace import StateSpace, tf2ss
from .plot_utility import _plot_response_curve
from .discretization import c2d
import numpy as np
import sympy as sym

__all__ = ['impulse', 'step', 'ramp', 'any_input']


def __get_cs(sys_, input_signal):
    """

    :param sys_:
    :type sys_: TransferFunction
    :param input_signal:
    :type input_signal: str
    :return:
    :rtype: sym.Add
    """
    from sympy.parsing.sympy_parser import parse_expr

    s = sym.Symbol('s')
    t = sym.Symbol('t')

    input_expr = sym.simplify(parse_expr(input_signal))

    signal_table = {'step': 1/s, 'impulse': 1, '0': 0, 'ramp': 1/s**2,
                    'user': sym.laplace_transform(input_expr, t, s)[0]}
    gs, *_ = _tf_to_symbol(sys_.num, sys_.den)
    cs = gs*signal_table.get(input_signal, signal_table["user"])
    return cs


def __ilaplace(expr):
    """
    Use:
        conduct the ilaplace transform

    :param expr: the expression
    :type expr: sympy.Add
    :return:
    :rtype:
    """
    from sympy.integrals.transforms import inverse_laplace_transform
    s = sym.Symbol('s')
    t = sym.Symbol('t')
    cs = expr.apart(s)

    tmp = sum(cs.args)
    if expr.equals(tmp):
        polys = [sym.nsimplify(i, tolerance=0.001, rational=True) for i in tmp.args]
        ilaplace_p = partial(inverse_laplace_transform, s=s, t=t)
        ct = 0
        for i in polys:
            i = ilaplace_p(i)
            ct += i
    else:
        cs = sym.nsimplify(expr, tolerance=0.001, rational=True)
        ct = inverse_laplace_transform(cs, s, t)

    if ct.has(sym.Heaviside):
        ct = ct.replace(sym.Heaviside, sym.numbers.One)
    if ct.has(sym.InverseLaplaceTransform):
        ct = ct.replace(sym.InverseLaplaceTransform, sym.numbers.Zero)

    return ct


# convert system to state space then get result
def _any_input(sys_, t, input_signal=0, init_cond=None):
    """
    Accept any input signal, then calculate the response of the system.

    :param sys_: the system
    :type sys_: TransferFunction | StateSpace
    :param t: time
    :type t: np.ndarray
    :param input_signal: input signal accepted by the system
    :type input_signal: numbers.Real | np.ndarray
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
    if dt > 0.02 and sys_.is_ctime:
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

    # check the input_signal validity
    if d_sys_.is_siso:
        u = _setup_control_signal(input_signal, t)
    else:
        raise NotImplementedError("not support MIMO system right now")  # TODO: finish it

    if init_cond is None:
        init_cond = np.mat(np.zeros((d_sys_.A.shape[0], 1)))
    else:
        # check the shape of init_cond
        if init_cond.shape[0] != d_sys_.A.shape[0] or init_cond.shape[1] != 1:
            raise ValueError("wrong dimension of init condition")

    x = _cal_x(d_sys_.A, d_sys_.B, len(t[1:]), init_cond, u)
    y = _cal_y(d_sys_.C, d_sys_.D, len(x), x, u)

    if sys_.is_siso:
        y = np.asarray(y).reshape(-1)
    else:
        y = [np.asarray(_).reshape(-1) for _ in y]
    return np.array(y), t


def _is_dt_validated(sys_: Union[TransferFunction, StateSpace],
                     dt: Union[int, float]) -> bool:
    if abs(dt - sys_.dt) <= 1e-7:
        return True
    else:
        return False


@singledispatch
def _setup_control_signal(input_signal, t):
    raise TypeError("Wrong type is given.")


@_setup_control_signal.register(np.ndarray)
def f(input_signal, t):
    if input_signal.shape == t.shape:
        u = input_signal
    else:
        raise ValueError("The input signal should have the same shape with t.")
    return u


@_setup_control_signal.register(list)
def f(input_signal, t):
    u = np.array(input_signal)
    if u.shape != t.shape:
        raise ValueError("The input signal should have the same shape with t")
    return u


def _cal_x(G, H, n, x_0, u):
    """
    calculate x step by step
    """
    x = [x_0]
    for i in range(n):
        x_k = G*x[i] + H*u[i]
        x.append(x_k)
    return x


def _cal_y(C, D, n, x, u):
    """
    calculate system output
    """
    y = []
    for i in range(n):
        y_k = C*x[i] + D*u[i]
        y.append(y_k)
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
    y, t = _any_input(sys_, t, u)
    if plot:
        _plot_response_curve(y, t, "step response", sys_.is_ctime)
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

    y, t = _any_input(sys_, t, u, x0)
    if plot:
        _plot_response_curve(y, t, "impulse response", sys_.is_ctime)
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
    y, t = _any_input(sys_, t, u)
    if plot:
        _plot_response_curve(y, t, "impulse response", sys_.is_ctime)
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

    y, t = _any_input(sys_, t, input_signal, init_cond)
    if plot:
        _plot_response_curve(y, t, "response", sys_.is_ctime)
    return y, t


def _setup_time_vector(sys_: StateSpace, n: int=1000):
    eigvals = np.linalg.eigvals(sys_.A)
    tc = 1 / np.min(np.abs(eigvals)) * 2
    if tc == np.inf:
        tc = 1
    if sys_.is_ctime:
        return np.linspace(0, 10 * tc, n)
    else:
        return np.arange(0, 10 * sys_.dt + 1, sys_.dt)
