from unittest import TestCase

from tcontrol.discretization import c2d
from tcontrol.transferfunction import tf


class TestDiscretization(TestCase):
    def setUp(self):
        self.s1 = tf([1, 1], [1, 0, 1])


    def test_c2d(self):
        msg = 'The c2d has been already tested in the test of transferfunction and' \
              'statespace.'
        self.skipTest(msg)
