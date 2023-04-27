from unittest import TestCase
import numpy as np

from pfs.ga.binaries import OrbitSampler

class BinariesTest(TestCase):
    def test_observe(self):
        s = OrbitSampler()
        bin = s.sample(size=100)
        self.assertEqual(bin.theta0.shape, (100,))

        obs = bin.observe(0)
        self.assertIsNotNone(obs.t)
        self.assertEqual(obs.v_los.shape, (100,))
        self.assertEqual(obs.v_los_err.shape, (100,))

        obs = bin.observe(100, s=np.s_[:10])
        self.assertIsNotNone(obs.t)
        self.assertEqual(obs.v_los.shape, (10,))
        self.assertEqual(obs.v_los_err.shape, (10,))