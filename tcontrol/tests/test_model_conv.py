from unittest import TestCase

from ..model_conversion import *
from ..statespace import ss
from ..transferfunction import tf
import numpy as np


class TestModelConversion(TestCase):
    def setUp(self):
        self.ss = ss([[1, 1], [-2, -5]], [[1], [1]], [[0, 3]], 0)
        self.tf = tf([3, -9], [1, 4, -3])

    def test_ss2tf(self):
        self.assertTrue(ss2tf(self.ss) == self.tf)
