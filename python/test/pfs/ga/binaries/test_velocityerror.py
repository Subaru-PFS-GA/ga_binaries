from unittest import TestCase
import numpy as np

from pfs.ga.binaries import VelocityError

class VelocityErrorTest(TestCase):
    def test_eval(self):
        M = np.linspace(19, 23.5, 100)

        v_los_err = VelocityError().eval(20)
        self.assertEqual(np.shape(v_los_err), ())
        
        v_los_err = VelocityError().eval(M)
        self.assertEqual(v_los_err.shape, (100,))
        
    def test_sample(self):
        M = np.linspace(19, 23.5, 100)

        v_los_err = VelocityError().sample(20)
        self.assertEqual(np.shape(v_los_err), ())
        
        v_los_err = VelocityError().sample(M)
        self.assertEqual(v_los_err.shape, (100,))

        v_los_err = VelocityError().sample(M, size=(10,))
        self.assertEqual(v_los_err.shape, (100, 10))