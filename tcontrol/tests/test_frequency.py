from unittest import TestCase
from tcontrol import frequency
import tcontrol as tc


class TestFrequency(TestCase):
    def test_nyquist(self):
        frequency.nyquist(tc.tf([0.5], [1, 2, 1, 0.5]), plot=False)

    def test_bode(self):
        frequency.bode(tc.zpk([], [0, -1, -2], 2), plot=False)
        frequency.bode(tc.tf([1], [1, 1]), plot=False)

    def test_evalfr(self):
        tf = tc.tf([1, -1], [1, 1, 1])
        frequency.evalfr(tf, 1 + 1j) - (0.23077 + 0.15385j)
        ss = tc.ss([[1, 2], [4, 0]], [[0], [1]], [[1, 1]], 0)
        frequency.evalfr(ss, 1 + 1j)
