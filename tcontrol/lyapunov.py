import numpy as np
from numpy.linalg import inv
from scipy.linalg import schur, hessenberg
from scipy.linalg import solve_sylvester, LinAlgError

from .statespace import config


def lyapunov(A, Q=None):
    """
    Solve the equation A.T * X + X * A = -Q
    default Q is set to I

    :param A: system matrix
    :type A: np.ndarray | np.matrix | List[List]
    :param Q: matrix
    :type Q: np.ndarray | np.matrix

    :return: the matrix X if there is a solution
    :rtype: np.ndarray | None
    """
    A = np.array(A)
    if Q is None:
        Q = np.eye(A.shape[0])
    try:
        X = solve_sylvester(A.T, A, -Q)
        if config['use_numpy_matrix']:
            return np.mat(X)
        else:
            return X
    except LinAlgError:
        return None


def discrete_lyapunov(A, Q=None):
    A = np.array(A)
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)

    if n < 3:
        return _mini_dlyap(A.T, A, Q)

    X = np.zeros(A.shape)
    h, q = hessenberg(A, True)
    t, z, *_ = schur(h)
    w = q @ z
    Q = w.T @ Q @ w

    partition = _partition_mat(t)
    for row in partition:
        for p in row:
            Q[p] += _prev_r(X, t)[p]
            i, j = p
            X[p] = _mini_dlyap(t[i, i], t[j, j], Q[p])
    X = w @ X @ w.T

    if config['use_numpy_matrix']:
        return np.mat(X)
    else:
        return X


def _partition_mat(M: np.ndarray):
    n = M.shape[0]
    i = 0
    diagnose_indices = []
    while i < n:
        if i + 1 == n or M[i + 1, i] == 0:
            part = (slice(i, i + 1), slice(i, i + 1))
            i += 1
        else:
            part = (slice(i, i + 2), slice(i, i + 2))
            i += 2
        diagnose_indices.append(part)

    indices = [[] for _ in diagnose_indices]
    for i, (m, _) in enumerate(diagnose_indices):
        indices[i].append((m, m))
        for j, (n, _) in enumerate(diagnose_indices[i + 1:], i + 1):
            indices[i].append((m, n))
            indices[j].append((n, m))
    return indices


def _prev_r(X, A):
    return A.T @ X @ A


def _mini_dlyap(Ak: np.ndarray, Al: np.ndarray, Q):
    """
    solve :math:`Ak^T X Al - X = -Q`

    :param Ak:
    :type Ak: np.ndarray
    :param Al:
    :type Al: np.ndarray
    :param Q:
    :type Q: np.ndarray | None
    :return:
    :rtype:
    """
    Q = -Q
    if Ak.shape == Al.shape == (1, 1):
        return Q / (Ak[0, 0] * Al[0, 0] - 1)
    elif Ak.shape == Al.shape == (2, 2):
        a11, a12, a21, a22 = Ak.flatten()
        b11, b12, b21, b22 = Al.flatten()
        U = [[a11 * b11 - 1, a21 * b11, b21 * a11, a21 * b21],
             [b11 * a12, b11 * a22 - 1, a12 * b21, a22 * b21],
             [a11 * b12, a21 * b12, a11 * b22 - 1, b22 * a21],
             [a12 * b12, b12 * a22, a12 * b22, a22 * b22 - 1]]

        V = np.reshape(Q, (4, 1), 'F')
        return np.reshape(inv(U) @ V, (2, 2), 'F')
    else:
        if Ak.size == 4:
            U = Ak.T * Al[0, 0] - np.eye(2)
            return inv(U) @ Q
        else:
            U = Ak[0, 0] * Al - np.eye(2)
            return Q @ inv(U)
