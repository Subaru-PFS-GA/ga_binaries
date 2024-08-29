from unittest import TestCase

from pfs.ga.binaries import LuminosityFunction

class OrbitSamplerTest(TestCase):
    def test_sample(self):
        l = LuminosityFunction()

        M = l.sample(size=100)
        self.assertEqual(M.shape, (100,))