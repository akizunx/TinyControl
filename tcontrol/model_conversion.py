from .transferfunction import TransferFunction
from .statespace import StateSpace

import numpy as np
from numpy.polynomial.polynomial import polycompanion

__all__ = ['tf2ss', 'ss2tf']


def tf2ss(sys_):
    """
    Convert transfer function model to state space model.

    :param sys_: the system
    :type sys_: TransferFunction

    :return: corresponded transfer function model
    :rtype: StateSpace
    """
    try:
        nump = np.poly1d(sys_.num)
        denp = np.poly1d(sys_.den)
    except AttributeError as e:
        raise TypeError(f'TransferFunction expected got {type(sys_)}') from e

    if not sys_.is_siso:
        raise ValueError('tf2ss only for siso system right now.')

    dt = sys_.dt
    if denp[denp.order] != 1:
        nump /= denp[denp.order]
        denp /= denp[denp.order]
    q, r = nump / denp
    num = r.coeffs
    bn = q.coeffs
    den = denp.coeffs
    D = np.atleast_2d(bn)

    if sys_.is_gain:
        A = np.array([[0]])
        B = np.array([[0]])
        C = np.array([[0]])
    else:
        # generate matrix A
        n = den.shape[0] - 1
        A = polycompanion(den[::-1]).T

        B = np.zeros((n, 1))
        B[-1] = 1

        C = np.zeros((1, n))
        C[0, 0: num.shape[0]] = num[::-1]

    return StateSpace(A, B, C, D, dt=dt)


def ss2tf(sys_):
    """
    Covert state space model to transfer function model.
    Only for SISO now

    a(s) = det(sI - A)
    R_{n-1} = I
    R_{n-2} = R_{n-1} * A + a_{n-1} * I
    E_{n-1} = C * R_{n-1} * B

    G(s) = (E_{n-1}s^{n-1} + E_{n-2}s^{n-2} + ... + E_0) / a(s) + D

    :param sys_: system
    :type sys_: StateSpace
    :return: corresponded transfer function model
    :rtype: TransferFunction
    """

    n = sys_.A.shape[0]
    p = sys_.B.shape[1]
    q = sys_.C.shape[0]

    cp = np.poly(sys_.A)  # characteristic polynomial
    I = np.eye(n)
    R = I
    E = np.empty((n, q * p))

    for i in range(n):
        E[i] = (sys_.C @ R @ sys_.B).ravel('C')
        R = R @ sys_.A + cp[i + 1] * I

    # every row in E presents coefficients of each numerator polynomial
    E = E.T
    # make sure every element in D correspond to the very E's row
    D = sys_.D.flatten('C').T
    if sys_.is_siso:
        num = np.polyadd(E[0], cp * D[0])
        return TransferFunction(num, cp, dt=sys_.dt)
    else:
        # return mimo as a list of TransferFunction temporarily
        ret = []
        r = []
        for i in range(q * p):
            num = np.polyadd(E[i], cp * D[i])
            r.append(TransferFunction(num, cp, dt=sys_.dt))
            if (i + 1) % p == 0:
                ret.append(r)
                r = []
        return ret
