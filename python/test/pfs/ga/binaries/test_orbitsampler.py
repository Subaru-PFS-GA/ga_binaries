from unittest import TestCase

from pfs.ga.binaries import OrbitSampler

class OrbitSamplerTest(TestCase):
    def test_sample(self):
        s = OrbitSampler()
        orb = s.sample(size=1)
        self.assertEqual(orb.theta0.shape, (1,))

        orb = s.sample(size=100)
        self.assertEqual(orb.theta0.shape, (100,))

        orb = s.sample(size=(10, 10))
        self.assertEqual(orb.theta0.shape, (10, 10))