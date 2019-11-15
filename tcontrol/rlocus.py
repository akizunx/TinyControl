from tcontrol.statespace import StateSpace
from tcontrol.plot_utility import plot_rlocus
from .model_conversion import *
import numpy as np

__all__ = ["rlocus"]


def rlocus(sys_, kvect=None, *, plot=True, **kwargs):
    """
    Use:
        the root locus of the system

    Example:
        >>> import tcontrol as tc
        >>> import numpy as np
        >>> system = tc.tf([1], [1, 2, 1])
        >>> tc.rlocus(system, np.linspace(0, 5, 1000))

    :param sys_: the transfer function of the system
    :type sys_: SISO | LTI
    :param kvect: k from 0 to inf
    :type kvect: np.ndarray
    :param plot: if plot is true it will draw the picture
    :type plot: bool
    :return: roots of the den and kvect
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    if not sys_.is_siso:
        raise NotImplementedError('rlocus is only for TransferFunction now.')
    if isinstance(sys_, StateSpace):
        system = ss2tf(sys_)
    else:
        system = sys_

    if kvect is None:
        nump = np.poly1d(system.num)
        denp = np.poly1d(system.den)

        d = _cal_multi_roots(nump, denp)
        k = -denp(d) / nump(d)
        k = k[np.where(k >= 0)]

        if k.dtype == np.complex:
            k = k[np.where(k.imag == 0)]
            k = np.real(k)
        k = np.sort(k)
        kvect = _setup_kvect(k)

    roots = _cal_roots(system, kvect)
    roots = _sort_roots(roots)

    if plot:
        plot_rlocus(kwargs, roots, system)

    return roots, kvect


def _cal_roots(sys_, kvect):
    nump = np.poly1d(sys_.num)
    denp = np.poly1d(sys_.den)
    p = denp + kvect[0] * nump
    order = p.order
    roots = np.zeros((len(kvect), order), dtype=np.complex)
    for i, k in enumerate(kvect):
        p_ = denp + k * nump
        roots[i] = p_.roots
    return roots


def _sort_roots(roots):
    """
    This is modified from _RLSortRoots in python-control.
    Reference: https://github.com/python-control/python-control
    """
    sorted_roots = np.zeros_like(roots)
    sorted_roots[0] = roots[0]
    pre_row = roots[0]
    for i, row in enumerate(roots[1:, :], 1):
        available = list(range(pre_row.shape[0]))
        for element in row:
            distance = np.abs(element - pre_row[available])
            min_index = available.pop(distance.argmin())
            sorted_roots[i, min_index] = element
        pre_row = sorted_roots[i, :]
    return sorted_roots.T


def _cal_multi_roots(nump, denp):
    p = denp * np.polyder(nump) - np.polyder(denp) * nump
    return p.roots


def _setup_kvect(k):
    if len(k):
        kvect = np.linspace(0, k[0], 50)
        for i in range(1, len(k)):
            kvect = np.append(kvect, np.linspace(k[i - 1], k[i], 50))
        kvect = np.append(kvect, np.linspace(k[-1], k[-1] * 50, 100))
        if kvect[-1] < 10:
            kvect = np.append(kvect, np.linspace(kvect[-1], kvect[-1] + 10, 10))
    else:
        kvect = np.linspace(0, 100, 10000)

    return kvect


