import tcontrol
from matplotlib import pyplot as plt
from unittest import TestCase


class TestTimeResponse(TestCase):
    def test_step(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(1)
        tcontrol.step(system)

    def test_ramp(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(2)
        tcontrol.ramp(system)

    def test_impulse(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(3)
        tcontrol.impulse(system)

    def test_any_input(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(4)
        tcontrol.any_input(system, None, "t + 1")