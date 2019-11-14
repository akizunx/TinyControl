"""
This module implements full state feedback
"""
from typing import List

from .canonical import ctrb_mat, ctrb_indices, ctrb_trans_mat
import numpy as np
from numpy.linalg import inv, matrix_rank
from numpy.polynomial.polynomial import polycompanion

__all__ = ['place']


def place(A, B, poles) -> np.ndarray:
    """
    Configure system poles by state feedback.

    The feedback matrix K is calculated by A - BK.

    :param A: the state matrix
    :type A: np.ndarray | List
    :param B: the input matrix
    :type B: np.ndarray | List
    :param poles: the expected poles
    :type poles: np.ndarray | List
    :return: state feedback matrix K
    :rtype: : np.ndarray | List
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    poles = np.atleast_1d(poles)
    try:
        Q = ctrb_mat(A, B)
    except ValueError as e:
        raise ValueError('wrong shape of input matrices') from e
    else:
        if matrix_rank(Q) != A.shape[0]:
            raise ValueError('FSF failed, since {A, B} is not controllable.')

    if B.shape[1] == 1:
        return _si_place(A, B, poles)
    else:
        return _mi_place(A, B, poles)


def _si_place(A, B, poles):
    k = np.poly(poles) - np.poly(A)
    k = k[1:]
    T = ctrb_trans_mat(A, B)
    return np.matmul(k[::-1], T)


def _mi_place(A, B, poles):
    n = A.shape[0]
    p = B.shape[1]
    T = ctrb_trans_mat(A, B)
    Ac = T @ A @ inv(T)
    Bc = T @ B

    Ac_tilde = polycompanion(np.flip(np.poly(poles))).T

    r = -1
    indices = ctrb_indices(A, B)
    Ar = np.empty((indices.shape[0], n))
    Br = np.empty((p, p))
    Ar_tilde = np.empty(Ar.shape)
    for i, mu in enumerate(indices):
        r += mu
        Ar[i] = Ac[r]
        Br[i] = Bc[r]
        Ar_tilde[i] = Ac_tilde[r]

    K = inv(Br) @ (Ar - Ar_tilde)
    return K @ T
