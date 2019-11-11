import unittest

from ..canonical import *
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestCanonical(unittest.TestCase):
    def setUp(self) -> None:
        self.A = np.array([[1, 2, 1], [0, -2, 2], [-1, 5, 0]])
        self.C = np.array([[1, 1, -0.5], [2, 0, 1]])

    def test_ctrb_mat(self):
        A = np.array([[0, 1, 0], [0, 0, 1], [-2, -4, -3]])
        B = np.array([[1, 0], [0, 1], [-1, 1]])
        M = np.array([[1, 0, 0, 1, -1, 1],
                      [0, 1, -1, 1, 1, -7],
                      [-1, 1, 1, -7, 1, 15]])
        assert_array_equal(ctrb_mat(A, B), M)

    def test_ctrb_indices(self):
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 3, -1]])
        B = np.array([[0, 1], [1, 0], [0, 0]])
        assert_array_equal(ctrb_indices(A, B), [2, 1])

    def test_ctrb_trans_mat(self):
        # si
        A = np.diag([1, 2])
        B = np.array([[-1], [2]])
        T = ctrb_trans_mat(A, B)
        assert_array_equal(T @ A @ np.linalg.inv(T), np.array([[0, 1], [-2, 3]]))

        # mi
        A = np.array([[-1, -4, -2], [0, 6, -1], [1, 7, -1]])
        B = np.array([[2, 0], [0, 0], [1, 1]])
        T = np.array([[0, -1, 0], [0, -6, 1], [-0.5, 3, 1]])
        assert_array_almost_equal(T, ctrb_trans_mat(A, B))
