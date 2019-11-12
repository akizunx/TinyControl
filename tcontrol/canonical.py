import numpy as np
from numpy.linalg import matrix_power, matrix_rank, inv

__all__ = ['ctrb_mat', 'ctrb_index', 'ctrb_indices', 'ctrb_trans_mat',
           'obsv_mat', 'obsv_indices', 'obsv_index']


def _check_ab(A: np.ndarray, B: np.ndarray):
    if A.shape[0] != A.shape[1]:
        raise ValueError('matrix A should be square.')
    if B.shape[0] != A.shape[1]:
        raise ValueError('matrix B should have the same row number as matrix A')


def _check_ac(A: np.ndarray, C: np.ndarray):
    if A.shape[0] != A.shape[1]:
        raise ValueError('matrix A should be square.')
    if C.shape[1] != A.shape[1]:
        raise ValueError('matrix C should have the same column number as matrix A')


def ctrb_mat(A: np.ndarray, B: np.ndarray):
    _check_ab(A, B)

    n = A.shape[0]
    p = B.shape[1]

    q = np.empty((n, n * p))
    for i in range(n):
        q[:, i * p: i * p + p] = matrix_power(A, i) @ B

    return q


def obsv_mat(A: np.ndarray, C: np.ndarray):
    _check_ac(A, C)
    return ctrb_mat(A.T, C.T).T


def _adjust_qc_order(Qc):
    p = Qc.shape[1] // Qc.shape[0]

    Qc_ = Qc[:, 0::p]
    for i in range(1, p):
        q = Qc[:, i::p]
        Qc_ = np.concatenate((Qc_, q), axis=1)

    return Qc_


def ctrb_indices(A: np.ndarray, B: np.ndarray):
    try:
        Qc = ctrb_mat(A, B)
    except ValueError as e:
        raise ValueError('wrong shape of input matrices') from e

    mat_indices = np.zeros(Qc.shape[1], dtype=np.bool)
    rank = 0
    n = A.shape[0]
    for i in range(Qc.shape[1]):
        mat_indices[i] = True
        r = matrix_rank(Qc[:, mat_indices])
        if r <= rank:
            mat_indices[i] = False
        else:
            rank = r

        if rank == n:
            break

    controllability_indices = np.zeros(n)
    p = B.shape[1]
    for i in range(p):
        controllability_indices[i] = np.sum(mat_indices[i: n: p])

    return np.trim_zeros(controllability_indices).astype(np.int)


def _adjust_qo_order(Qo):
    return _adjust_qc_order(Qo.T).T


def obsv_indices(A: np.ndarray, C: np.ndarray):
    return ctrb_indices(A.T, C.T)


def ctrb_index(A: np.ndarray, B: np.ndarray):
    return np.max(ctrb_indices(A, B))


def obsv_index(A: np.ndarray, C: np.ndarray):
    return np.max(obsv_indices(A, C))


def ctrb_trans_mat(A: np.ndarray, B: np.ndarray):
    _check_ab(A, B)
    if B.shape[1] == 1:
        T = np.empty(A.shape)
        Qc = inv(ctrb_mat(A, B))
        p = Qc[-1]
        for i in range(A.shape[0]):
            T[i] = p @ matrix_power(A, i)
        return T
    else:
        return luenberger_ctrb_trans_mat(A, B)


def luenberger_ctrb_trans_mat(A: np.ndarray, B: np.ndarray):
    Qc = ctrb_mat(A, B)
    indices = ctrb_indices(A, B)
    Qc = _adjust_qc_order(Qc)

    n = A.shape[0]
    Q = Qc[:, : n][:, : indices[0]]
    for i, ind in enumerate(indices[1:], 1):
        Q = np.concatenate((Q, Qc[:, i * n: i * n + n][:, : ind]), axis=1)
    Q_I = inv(Q)

    k = 0
    T = np.empty(A.shape)
    for ind in indices:
        e = Q_I[k: k + ind, :][-1]
        for i in range(ind):
            T[k + i] = e @ matrix_power(A, i)
        k += ind
    return T
