from unittest import TestCase

import tcontrol as tc
import numpy as np


class TestTimeResponse(TestCase):
    def setUp(self):
        self.tf_ = tc.tf([5, 25, 30], [1, 6, 10, 8])
        self.A = np.array([[0, 1, 0], [0, 0, 1], [-8, -10, -6]])
        self.B = np.array([[0], [0], [1]])
        self.C = np.array([[30, 25, 5]])
        self.ss_ = tc.ss(self.A.T, self.C.T, self.B.T, 0)

        self.tf1 = tc.tf([1], [1, 1])
        self.step_y = np.array([0, 0.3934, 0.6321, 0.7769, 0.8647, 0.9179, 0.9502, 0.9698,
                                0.9816, 0.9888, 0.9932, 0.9959, 0.9975, 0.9984, 0.9991,
                                0.9994, 0.9996, 0.9997, 0.9998, 0.9999])
        self.impulse_y = np.array([1.0000, 0.6065, 0.3679, 0.2231, 0.1353, 0.0821, 0.0498,
                                   0.0302, 0.0183, 0.0111, 0.0067, 0.0041, 0.0025, 0.0015,
                                   0.0009, 0.0006, 0.0003, 0.0002, 0.0001, 0])
        self.t = np.arange(0, 10, 0.5)

    def test_step(self):
        y, _ = tc.step(self.tf1, self.t, plot=False)
        r = np.max(np.abs(y - self.step_y))
        self.assertLess(r, 1e-4)

        y1, t1 = tc.step(self.tf_, plot=False)
        y2, t2 = tc.step(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5)

    def test_ramp(self):
        y1, t1 = tc.ramp(self.tf_, plot=False)
        y2, t2 = tc.ramp(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5)

    def test_impulse(self):
        y, _ = tc.impulse(self.tf1, self.t, plot=False)
        r = np.max(np.abs(y - self.impulse_y))
        self.assertLess(r, 1e-4)

        y1, t1 = tc.impulse(self.tf_, plot=False)
        y2, t2 = tc.impulse(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5)

    def test_any_input(self):
        t = np.linspace(0, 10, 1000)
        u = np.linspace(0, 10, 1000) + 1
        y1, t1 = tc.any_input(self.tf_, t, u, plot=False)
        y2, t2 = tc.any_input(self.ss_, t, u, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5)

        # test discrete time situation
        d_sys = tc.c2d(tc.tf([1], [1, 1]), 1, 'Tustin')
        u = np.ones((10,), dtype=int)
        y, t = tc.any_input(d_sys, np.arange(0, 10, 1), u, plot=False)
        error = np.abs(
            y - [0.3333, 0.7778, 0.9259, 0.9753, 0.9918, 0.9973, 0.9991, 0.9997, 0.9999,
                 1])
        self.assertTrue(np.all(np.less_equal(error, 1e-4)))
