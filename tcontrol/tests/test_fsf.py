import unittest

from ..fsf import *
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestFSF(unittest.TestCase):
    def setUp(self) -> None:
        self.A = np.array([[0, 1], [-2, -3]])
        self.b = np.array([[0], [1]])
        self.poles = np.array([-3, -4])

    def test_place(self):
        # si
        k = place(self.A, self.b, self.poles)
        assert_array_equal(k, [10, 4])

        # mi
        A = np.array([[0, 1, 0], [-1, -2, 0], [0, 0, -4]])
        B = np.array([[0, -1], [1, 0], [0, 1]])
        poles = np.array([-1, -3, -4])
        K = place(A, B, poles)
        assert_array_almost_equal(np.roots(np.poly(A - B @ K)), [-4, -3, -1])

        # bad input
        self.assertRaises(ValueError, place, self.A, self.b.T, self.poles)
        self.assertRaises(ValueError, place, self.A, np.array([[0], [0]]), self.poles)
