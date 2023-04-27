from unittest import TestCase

from pfs.ga.binaries import OrbitSampler

class OrbitSamplerTest(TestCase):
    def test_sample(self):
        s = OrbitSampler()
        bin = s.sample(size=1)
        self.assertEqual(bin.theta0.shape, (1,))

        bin = s.sample(size=100)
        self.assertEqual(bin.theta0.shape, (100,))