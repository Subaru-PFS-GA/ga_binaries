from unittest import TestCase
import numpy as np

from pfs.ga.binaries import OrbitSampler

class BinariesTest(TestCase):
    def test_calculate_observables(self):
        s = OrbitSampler()
        orb = s.sample(size=100)
        self.assertEqual(orb.theta0.shape, (100,))

        t = 1
        orb.calculate_observables(t)
        self.assertIsNotNone(orb.t)
        self.assertEqual(orb.v_los.shape, (100,))

        t = 1
        orb.calculate_observables(t, s=np.s_[:10])
        self.assertIsNotNone(orb.t)
        self.assertEqual(orb.v_los.shape, (10,))

        t = np.linspace(0, 100, 10)
        orb.calculate_observables(t)
        self.assertIsNotNone(orb.t)
        self.assertEqual(orb.v_los.shape, (10, 100))

    def test_generate_dv_los_index(self):
        s = OrbitSampler()
        orb = s.sample(size=10000)
        self.assertEqual(orb.theta0.shape, (10000,))

        t = np.array([0, 200, 400])
        orb.calculate_observables(t)

        orb.generate_dv_los_index()

    def test_search_dv_los_index(self):
        s = OrbitSampler()
        orb = s.sample(size=1000000)

        t = np.array([0, 120, 360])
        orb.calculate_observables(t)
        orb.generate_dv_los_index()

        dv = np.array([100, 50])
        cov = np.array([[40, -10], [-10, 80]])
        orb.search_dv_los_index(dv, cov)