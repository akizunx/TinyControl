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
    M[i, :], M[j, :] = M[j, :].copy(), M[i, :].copy()
    return M


def _swap_cols(M, i, j):
    M[:, i], M[:, j] = M[:, j].copy(), M[:, i].copy()
    return M


def _simplify_symbol_expr(func):
    def wrapper(*args, **kwargs):
        r = func(*args, **kwargs)
        if isinstance(r[0, 0], sym.Expr):
            if not isinstance(r[0, 0], sym.Poly):
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
    M[i, :] = M[i, :] + M[j, :] * k
    return M


@_simplify_symbol_expr
def _add_col(M, i, j, k):
    M[:, i] = M[:, i] + M[:, j] * k
    return M


def _elem2sympy_poly(M):
    m, n = M.shape
    s = sym.symbols('s')
    N = sym.zeros(*M.shape)
    for i in range(m):
        for j in range(n):
            N[i, j] = M[i, j].as_poly(s)
    return N


def _min_degree_index(M: sym.Matrix, i=0):
    row = M[i, :].reshape(1, M.shape[1])
    col = M[:, i].reshape(M.shape[0], 1)
    a = [x.degree() for x in row]
    b = [x.degree() for x in col]
    minimum = np.min([x for x in a + b if x != -sym.oo])
    min_ind = (i, i)
    try:
        min_ind = (i, a.index(minimum))
    except ValueError:
        min_ind = (b.index(minimum), i)
    finally:
        return min_ind


def poly_smith_form(M: sym.Matrix):
    M = _elem2sympy_poly(M)
    N = sym.zeros(*M.shape)
    diag = _poly_smith_form(M)
    for i, x in enumerate(diag):
        N[i, i] = x
    return N


def _poly_smith_form(M: sym.Matrix):
    m, n = M.shape
    if M.is_zero:
        return []

    while True:
        i, j = _min_degree_index(M)
        if i < j:
            _swap_cols(M, i, j)
        elif i > j:
            _swap_rows(M, i, j)

        p = M[0, 0]
        qr_table = dict()
        for i in range(m):
            q, r = M[i, 0].div(p)
            if r == 0:
                continue
            else:
                qr_table.update({(i, 0, q, r): r.degree()})
        for j in range(n):
            q, r = M[0, j].div(p)
            if r == 0:
                continue
            else:
                qr_table.update({(0, j, q, r): r.degree()})

        if not qr_table:
            break
        else:
            i, j, q, r = min(qr_table, key=qr_table.get)
            if i < j:
                _add_col(M, j, i, -q)
            elif i > j:
                _add_row(M, j, i, -q)

    for i in range(1, m):
        q, r = M[i, 0].div(p)
        _add_row(M, i, 0, -q)
    for j in range(1, n):
        q, r = M[0, j].div(p)
        _add_col(M, j, 0, -q)
    coeff = M[0, 0].all_coeffs()[0]
    p, _ = M[0, 0].div(coeff)

    return [p] + _poly_smith_form(M[1:, 1:])
