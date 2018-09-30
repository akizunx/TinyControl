from .transferfunction import TransferFunction
from .statespace import StateSpace
from .exception import *

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
    Covert state space model to  transfer function model.

    :param sys_: system
    :type sys_: StateSpace
    :return: corresponded transfer function model
    :rtype: TransferFunction
    """
    from scipy.linalg import eigvals

    n = sys_.A.shape[0]
    p = sys_.B.shape[1]
    q = sys_.C.shape[0]

    poles = np.linalg.eigvals(sys_.A)
    den = np.poly(poles)

    tfs = []
    for i in range(p):
        tmp = []
        for j in range(q):
            b = sys_.B[:, i]
            c = sys_.C[j, :]
            d = sys_.D[i: i + 1, j: j + 1]

            M = np.concatenate((np.concatenate((sys_.A, -c)),
                                np.concatenate((b, -d))), axis=1)
            N = np.zeros_like(M)
            N[0: n, 0: n] = np.eye(n)
            zeros = eigvals(M, N)
            zeros = zeros[zeros != np.inf]
            num = np.poly(zeros)
            if not isinstance(num, np.ndarray):
                num = np.array([num])

            s = np.append(poles[poles.imag == 0], zeros[zeros.imag == 0])
            if s.size:
                s = np.real(np.max(s) + 1)
            else:
                s = 1
            u = c * (s * np.eye(n) - sys_.A).I * b + d
            v = np.polyval(den, s) / np.polyval(num, s)
            k = u * v

            num = num.reshape(-1)
            k = np.asarray(k).reshape(-1)
            tmp.append(TransferFunction(k * num, den, dt=sys_.dt))
        tfs.append(tmp)

    if sys_.is_siso:
        return tfs[0][0]
    else:
        return tfs
