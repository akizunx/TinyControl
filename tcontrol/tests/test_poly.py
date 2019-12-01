from unittest import TestCase
from tcontrol.poly import *
from ..poly import _swap_rows, poly_smith_form, _swap_cols, \
    _mul_col, _mul_row, _add_col, _add_row
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import sympy as sym


class TestPoly(TestCase):
    def setUp(self) -> None:
        s = sym.symbols('s')
        self.s = sym.symbols('s')
        self.tfm = sym.Matrix([[s, s ** 2 + 1], [1, s - 1]])

    def test_conv(self):
        ret = conv([1, -1], [1, 1])
        self.assertEqual(all(ret == np.array([1, 0, -1])), True)
        ret = conv([1, 1], [1, 1], [1, 1])
        self.assertEqual(all(ret == np.array([1, 3, 3, 1])), True)

    def test_poly(self):
        ret = poly([1, -1])
        self.assertEqual(all(ret == np.array([1, 0, -1])), True)
        ret = poly([1j, -1j, -1])
        self.assertEqual(all(ret == np.array([1, 1, 1, 1])), True)

    def test_roots(self):
        ret = roots(np.array([1, 0, -1]))
        self.skipTest('roots is from numpy roots')
        self.assertEqual(all(ret == np.array([1, -1])), True)

    def test__swap_rows(self):
        a = np.array([[1, 2], [3, 4]])
        assert_array_equal(_swap_rows(a, 0, 1), np.array([[3, 4], [1, 2]]))


    def test__swap_cols(self):
        a = np.array([[1, 2], [3, 4]])
        assert_array_equal(_swap_cols(a, 0, 1), np.array([[2, 1], [4, 3]]))

    def test__mul_row(self):
        a = np.array([[1, 2], [3, 4]])
        assert_array_equal(_mul_row(a, 0, -1), np.array([[-1, -2], [3, 4]]))

    def test__mul_col(self):
        a = np.array([[1, 2], [3, 4]])
        assert_array_equal(_mul_col(a, 0, -1), np.array([[-1, 2], [-3, 4]]))

        self.assertTrue(_mul_col(self.tfm, 0, -self.s) == sym.Matrix(
            [[-self.s ** 2, self.s ** 2 + 1], [-self.s, self.s - 1]]))

    def test__add_row(self):
        a = np.array([[1, 2], [2, 4]])
        assert_array_equal(_add_row(a, 0, 1, -0.5), np.array([[0, 0], [2, 4]]))

        b = _add_row(self.tfm, 0, 1, -self.s)
        # b.simplify()
        self.assertTrue(b == sym.Matrix([[0, self.s + 1], [1, self.s - 1]]))

    def test__add_col(self):
        a = np.array([[1, 2], [2, 4]])
        assert_array_equal(_add_col(a, 0, 1, -0.5), np.array([[0, 2], [0, 4]]))
