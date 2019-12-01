from copy import deepcopy

import numpy as np
import sympy as sym

__all__ = ['conv', 'deconv', 'poly', 'roots']


def conv(*args):
    args = [np.array(i) for i in args]
    r = np.array([1])
    for i in args:
        r = np.convolve(i, r)
    return r


def deconv(u, v):
    return np.polydiv(u, v)


def poly(p):
    return np.poly(p)


def roots(p):
    """

    :param p:
    :type p: numpy.ndarray
    :return:
    :rtype: numpy.ndarray
    """
    return np.roots(p)


def _swap_rows(M, i, j):
    M[i, :], M[j, :] = deepcopy(M[j, :]), deepcopy(M[i, :])
    return M


def _swap_cols(M, i, j):
    M[:, i], M[:, j] = deepcopy(M[:, j]), deepcopy(M[:, i])
    return M


def _simplify_symbol_expr(func):
    def wrapper(*args, **kwargs):
        r = func(*args, **kwargs)
        if isinstance(r, sym.Matrix):
            r.simplify()
        return r

    return wrapper


@_simplify_symbol_expr
def _mul_row(M, i, k):
    M[i, :] = k * M[i, :]
    return M


@_simplify_symbol_expr
def _mul_col(M, i, k):
    M[:, i] = k * M[:, i]
    return M


@_simplify_symbol_expr
def _add_row(M, i, j, k):
    M[i, :] = M[i, :] + k * M[j, :]
    return M


@_simplify_symbol_expr
def _add_col(M, i, j, k):
    M[:, i] = M[:, i] + k * M[:, j]
    return M


def poly_smith_form(M):
    pass
