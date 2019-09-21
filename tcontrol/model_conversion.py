from .transferfunction import TransferFunction
from .statespace import StateSpace

import numpy as np

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
        raise TypeError("TransferFunction expected got {0}".format(type(sys_))) from e
    dt = sys_.dt

    if denp[denp.order] != 1:
        nump /= denp[denp.order]
        denp /= denp[denp.order]
    q, r = nump/denp
    num = r.coeffs
    bn = q.coeffs
    den = denp.coeffs
    D = np.matrix(bn)

    if sys_.is_gain:
        A = np.mat([[0]])
        B = np.mat([[0]])
        C = np.mat([[0]])
        return StateSpace(A, B, C, D, dt=dt)
    else:
        # generate matrix A
        A = np.zeros((1, den.shape[0] - 2))
        A = np.concatenate((A, np.eye(den.shape[0] - 2)), axis=0)
        den = den[1:]
        den = np.mat(den[::-1])
        A = np.concatenate((A, -den.T), axis=1)

        B = np.zeros(A.shape[0])
        B[A.shape[0] - num.shape[0]:] = num
        B = np.mat(B[::-1]).T
        c = np.zeros(A.shape[0])
        c[-1] = 1
        C = np.mat(c)
        return StateSpace(A.T, C.T, B.T, D, dt=dt)


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

    cp = np.poly(sys_.A) # characteristic polynomial
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
