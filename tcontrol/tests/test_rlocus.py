import pathlib
from unittest import TestCase

from ..transferfunction import tf
from tcontrol.rlocus import rlocus
import numpy as np

path = pathlib.Path('.')
if 'tests' not in str(path.cwd()):
    path = (path / 'tcontrol/tests/rlocus_result.npz').resolve()
else:
    path = (path / 'rlocus_result.npz').resolve()

class TestRlocus(TestCase):
    def test_rlocus(self):
        self.skipTest('fail in some environments')
        system = tf([0.5, 1], [0.5, 1, 1])
        r0, _ = rlocus(system, xlim=[-5, 0.5], plot=False)
        r1, _ = rlocus(tf([1], [1, 4, 3, 0]), plot=False)
        r2, _ = rlocus(tf([1], [1, 2, 1, 0]), plot=False)
        r3, _ = rlocus(tf([1, 3], [1, 6, 8, 0]), plot=False)
        r4, _ = rlocus(tf([1, 2], [1, 2, 1, -1]), plot=False)
        # np.savez('rlocus_result', r0=r0, r1=r1, r2=r2, r3=r3, r4=r4)

        t = np.load(path)
        self.assertTrue(np.array_equal(r0, t['r0']))
        self.assertTrue(np.array_equal(r1, t['r1']))
        self.assertTrue(np.array_equal(r2, t['r2']))
        self.assertTrue(np.array_equal(r3, t['r3']))
        self.assertTrue(np.array_equal(r4, t['r4']))
