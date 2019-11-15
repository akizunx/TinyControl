from functools import singledispatch

from tcontrol.plot_utility import plot_bode, plot_nyquist
from tcontrol.transferfunction import TransferFunction, LinearTimeInvariant
from .statespace import StateSpace
import numpy as np

__all__ = ["nyquist", "bode"]


def nyquist(sys_, omega=None, *, plot=True):
    """
    Use:
        Draw the nyquist plot of the system

    :param sys_: the transfer function of the system
    :type sys_: TransferFunction
    :param omega: values of angular frequency
    :type omega: np.ndarray
    :param plot:
    :type plot: bool
    :return:
    :rtype: (np.ndarray, np.ndarray)
    """
    if not isinstance(sys_, TransferFunction) and isinstance(sys_, LinearTimeInvariant):
        raise NotImplementedError

    if omega is None:
        omega = np.linspace(0, 100, 10000)
    omega = 1j * omega

    num = np.poly1d(sys_.num)
    den = np.poly1d(sys_.den)

    result = num(omega) / den(omega)

    if plot:
        plot_nyquist(result)

    return result, omega


def bode(sys_, omega=None, *, plot=True):
    """
    Use:
        draw the bode plot

    :param sys_:
    :type sys_: TransferFunction
    :param omega: values of angular frequency
    :type omega: np.ndarray
    :param plot:
    :type plot: bool
    :return:
    :rtype: (np.ndarray, np.ndarray)
    """
    if not isinstance(sys_, TransferFunction):
        if isinstance(sys_, LinearTimeInvariant):
            raise NotImplementedError

    if omega is None:
        omega = np.logspace(-2, 2)
    omega = omega * 1j

    num = np.poly1d(sys_.num)
    den = np.poly1d(sys_.den)

    A = 20 * np.log10(np.abs(num(omega) / den(omega)))

    deg = np.zeros(omega.shape)
    for i in sys_.zero():
        p = np.poly1d([1, -i])
        deg = deg + np.angle(p(omega), deg=True)
    for i in sys_.pole():
        p = np.poly1d([1, -i])
        deg = deg - np.angle(p(omega), deg=True)
    phi = np.asarray(deg, dtype=float)

    if plot:
        plot_bode(A, omega, phi)

    return A, phi, omega


@singledispatch
def evalfr(system, frequency):
    msg = f'expected TransferFunction or StateSpace, got{type(system)}'
    raise TypeError(msg)


@evalfr.register(TransferFunction)
def _tf_evalfr(system, frequency):
    return system.evalfr(frequency)


@evalfr.register(StateSpace)
def _ss_evalfr(system, frequency):
    return system.evalfr(frequency)
