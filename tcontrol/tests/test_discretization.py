from unittest import TestCase

from tcontrol.discretization import c2d
from tcontrol.transferfunction import tf
import numpy as np


class TestDiscretization(TestCase):
    def setUp(self):
        self.s1 = tf([1], [1, 0, 1])
        self.zoh = tf([0.4597, 0.4597], [1, 1.0806, 1], dt=1)

    def test_c2d_zoh(self):
        d_sys = c2d(self.s1, 1, 'zoh')
        self.assertLessEqual(np.max(np.abs(d_sys.num - self.zoh.num)), 1e-4)

    def test_c2d_tustin(self):
        d_sys = c2d(self.s1, 1, 'tustin')
        error = np.abs(d_sys.num - np.array([1, 2, 1]))
        self.assertLessEqual(np.max(error), 1e-4)

    def test_c2d_matched(self):
        d_sys = c2d(self.s1, 1, 'matched')
        error = np.abs(d_sys.num - np.array([0.2298, 0.4597, 0.2298]))
        self.assertLessEqual(np.max(error), 1e-4)
