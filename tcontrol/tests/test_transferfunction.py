from unittest import TestCase

from tcontrol.transferfunction import TransferFunction, tf, zpk
from .tools.test_utility import assert_tf_equal, assert_array_almost_equal
from ..exception import *
import numpy as np


class TestTransferFunction(TestCase):
    def setUp(self):
        self.s1 = TransferFunction([1, 1], [1, 0, 1])
        self.s2 = TransferFunction([1, 0, 1], [1, 0, 0, 1])
        self.s3 = TransferFunction([1], [1, 0])
        self.s4 = TransferFunction([2, 0], [1, 4, 3])
        self.s5 = TransferFunction([1, 4, 3], [1, 4, 5, 0])
        self.s6 = TransferFunction([3, 4, 3], [1, 4, 3, 0])
        self.s7 = TransferFunction([2, 0], [1, 4, 3, 0])
        self.s8 = TransferFunction([1], [1, 2, 1])
        self.s9 = TransferFunction([1, 2, 3, 0], [1, 2, 2, 2, 1])

        self.s = TransferFunction([1, 1], [1, 0, 1])
        self.neg_s = TransferFunction([-1, -1], [1, 0, 1])

    def test___init__(self):
        self.assertNotEqual(id(self.s1), id(self.s2))
        self.assertNotEqual(id(self.s1.num), id(self.s2.num))
        self.assertNotEqual(id(self.s1.den), id(self.s2.den))

    def test___neg__(self):
        assert_tf_equal(-self.s, self.neg_s)

    def test_parallel(self):
        sys_ = tf([5, 24, 34, 24, 9], [1, 8, 22, 24, 9, 0])
        assert_tf_equal(self.s3.parallel(self.s4, self.s4), sys_)
        assert_tf_equal(self.s3.parallel(self.s3), tf([2], [1, 0]))

    def test__cascade(self):
        sys_ = tf([4, 0, 0], [1, 8, 22, 24, 9, 0])
        assert_tf_equal(self.s3.cascade(self.s4, self.s4), sys_)

    def test_feedback(self):
        assert_tf_equal(self.s3.feedback(self.s4), self.s5)

    def test___add__(self):
        assert_tf_equal(self.s3 + self.s4, self.s6)
        assert_tf_equal(self.s3 + 1, tf([1, 1], [1, 0]))

    def test___mul__(self):
        assert_tf_equal(self.s3 * self.s4, self.s7)
        assert_tf_equal(self.s3 * 0.5, tf([0.5], [1, 0]))

    def test___sub__(self):
        assert_tf_equal(self.s1 - self.s8, self.s9)

    def test_dc_gain(self):
        self.assertEqual(tf([1, 2], [1, 1, 0]).dc_gain, float('inf'))

    def test_pole(self):
        s = TransferFunction([1, 2], [1, 2, 1])
        assert_array_almost_equal(s.pole(), [-1, -1])
        s = TransferFunction([1, 2], [1, 1, -1, -1])
        assert_array_almost_equal(np.sort(s.pole()), [-1, -1, 1])

    def test_zero(self):
        s = TransferFunction([1, 2], [1, 2, 1])
        self.assertEqual((s.zero() == np.roots([1, 2])).all(), True)

    def test_tf(self):
        assert_tf_equal(tf([1], [1, 0]), TransferFunction([1], [1, 0]))
        assert_tf_equal(tf(TransferFunction([1], [1, 0])), tf([1], [1, 0]))
        assert_tf_equal(tf(num=[1], den=[1, 0]), tf([1], [1, 0]))

    def test_zpk(self):
        s1 = TransferFunction([5, 5], [1, 0, -4])
        s2 = zpk([-1], [-2, 2], 5)
        assert_tf_equal(s1, s2)

    def test_bad_input(self):
        self.assertRaises(WrongNumberOfArguments, tf, *[[1], 2, 3, 4])
        self.assertRaises(TypeError, tf, [1, 3, 4, 5])
        self.assertRaises(WrongNumberOfArguments, tf, **{})
        self.assertRaises(WrongNumberOfArguments, tf, **{'num': [1]})
        self.assertRaises(WrongNumberOfArguments, tf, **{'den': [1]})
        self.assertRaises(WrongNumberOfArguments, tf, **{'num': [1], 'dt': 0.1})
