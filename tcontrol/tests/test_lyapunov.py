import unittest

from ..lyapunov import *
import numpy as np
from numpy.random import randint, rand


class TestLyapunov(unittest.TestCase):
    def test_lyapunov(self):
        A = [[0, 1], [-2, -3]]
        self.assertTrue(np.allclose(lyapunov(A), [[1.25, 0.25], [0.25, 0.25]]))
        A = [[-1, 1], [2, -3]]
        self.assertTrue(np.allclose(lyapunov(A), [[1.75, 0.625], [0.625, 0.375]]))

    def test_discrete_lyapunov(self):
        A = np.array([[-0.2, -0.2, 0.4], [0.5, 0, 1], [0, -0.4, -0.5]])
        Q = np.eye(3)
        X = discrete_lyapunov(A)
        self.assertTrue(np.allclose(A.T @ X @ A - X + Q, np.zeros((3, 3))))

        A = randint(0, 5, (6, 6)) + rand(6, 6)
        Q = np.eye(6)
        X = discrete_lyapunov(A)
        self.assertTrue(np.allclose(A.T @ X @ A - X + Q, np.zeros((6, 6))))
