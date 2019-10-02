import unittest

from ..lyapunov import *
from numpy.random import randint, seed
from scipy.linalg import schur


class TestLyapunov(unittest.TestCase):

    def test_lyapunov(self):
        A = [[0, 1], [-2, -3]]
        self.assertTrue(np.allclose(lyapunov(A), [[1.25, 0.25], [0.25, 0.25]]))
        A = [[-1, 1], [2, -3]]
        self.assertTrue(np.allclose(lyapunov(A), [[1.75, 0.625], [0.625, 0.375]]))

    def test_discrete_lyapunov(self):
        A = [[-0.2, -0.2, 0.4], [0.5, 0, 1], [0, -0.4, -0.5]]
        B = [[1.595998, 0.5665776, 0.00224935],
             [0.5665776, 3.02730, -0.6620739],
             [0.00224934, -0.662074, 1.62605]]
        self.assertTrue(np.allclose(discrete_lyapunov(A), B))

    def test_partition_mat(self):
        seed(234252)
        M, *_ = schur(randint(0, 5, (4, 4)))
        print(M)
        p = _partition_mat(M)
        for i in p:
            print(i)
        print(M[p[-1][-1]])

    def test_mini_dlyap(self):

        A = np.array([[-0.2, -0.2, 0.4], [0.5, 0, 1], [0, -0.4, -0.5]])
        Q = np.eye(3)
        X1 = _solve_discrete_lyapunov(A, Q)
        X2 = discrete_lyapunov(A, Q)
        print(X1 - X2)
        A = randint(1, 10, (5, 5))
        Q = np.eye(5)
        X1 = _solve_discrete_lyapunov(A, Q)
        X2 = discrete_lyapunov(A, Q)
        print(X1 - X2)
