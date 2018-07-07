from unittest import TestCase

import numpy as np

from tcontrol.pzmap import pzmap
from tcontrol.tests.data_generator import SYSTEMS
from tcontrol.transferfunction import tf


class TestPzmap(TestCase):
    def test_pzmap(self):
        sys_ = tf([1, 2, 2], [1, 2, 1, 1])
        p, z = pzmap(sys_, plot=False)
        self.assertTrue(all(np.equal(p, np.roots(sys_.den))))
        self.assertTrue(all(np.equal(z, np.roots(sys_.num))))

    def test_pzmap_random_data(self):
        sys_ = [tf(*i) for i in SYSTEMS]
        for i in sys_:
            p, z = pzmap(i, plot=False)
            self.assertTrue(all(np.equal(p, np.roots(i.den))))
            self.assertTrue(all(np.equal(z, np.roots(i.num))))

    def test_pzmap_discrete_time(self):
        system =  tf([1], [1, 1], 1)
        pzmap(system, plot=False)
