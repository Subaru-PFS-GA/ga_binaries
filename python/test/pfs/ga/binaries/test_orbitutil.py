from unittest import TestCase
import numpy as np

from pfs.ga.binaries import OrbitUtil

class OrbitUtilTest(TestCase):
    def test_calculate_gamma(self):
        gamma0 = np.array([0, 1, 2])
        P = np.array([0, 1, 2])
        
        t = np.array(1)
        gamma = OrbitUtil.calculate_gamma(gamma0, P, t)
        self.assertEqual(gamma.shape, (3,))

        t = np.array([0, 1, 2])
        gamma = OrbitUtil.calculate_gamma(gamma0, P, t)
        self.assertEqual(gamma.shape, (3, 3))

    def test_calculate_theta_series(self):
        gamma = np.array([0, 1, 2])
        e = np.array([0, 0.1, 0.2])
        
        theta = OrbitUtil.calculate_theta_series(gamma, e)
        self.assertEqual(theta.shape, (3,))
        
    def test_calculate_theta_iterative(self):
        gamma = np.array([0, 1, 2])
        e = np.array([0, 0.1, 0.2])
        
        theta = OrbitUtil.calculate_theta_iterative(gamma, e)
        self.assertEqual(theta.shape, (3,))

    def test_calculate_v_los(self):
        a = np.array([0, 1, 2])
        P = np.array([0, 1, 2])
        e = np.array([0, 0.1, 0.2])
        i = np.array([0, 0.1, 0.2])
        theta = np.array([0, 1, 2])
        omega = np.array([0, 0.1, 0.2])
        
        v_los = OrbitUtil.calculate_v_los(a, P, e, i, theta, omega)
        self.assertEqual(v_los.shape, (3,))
    
    def test_calculate_major_axis_from_period(self):
        P = np.array([0.5, 1, 2])
        m1 = np.array([0.8, 0.8, 0.8])
        m2 = np.array([0.1, 0.2, 0.3]) * m1
        
        a = OrbitUtil.calculate_major_axis_from_period(P, m1, m2)
        self.assertEqual(a.shape, (3,))

    def test_calculate_period_from_major_axis(self):
        a = np.array([0.5, 1, 2])
        m1 = np.array([0.8, 0.8, 0.8])
        m2 = np.array([0.1, 0.2, 0.3]) * m1
        
        P = OrbitUtil.calculate_period_from_major_axis(a, m1, m2)
        self.assertEqual(P.shape, (3,))

    def test_calculate_eccentric_anomaly_from_mean(self):
        e = np.array([0, 0.1, 0.2])
        gamma = np.array([0, 1, 2])
        
        E = OrbitUtil.calculate_eccentric_anomaly_from_mean(e, gamma)
        self.assertEqual(E.shape, (3,))

    def test_calculate_true_anomaly_from_eccentric(self):
        e = np.array([0, 0.1, 0.2])
        E = np.array([0, 1, 2])
        
        theta = OrbitUtil.calculate_true_anomaly_from_eccentric(e, E)
        self.assertEqual(theta.shape, (3,))

    def test_convert_AU_per_year_to_km_per_sec(self):
        v = np.array([0, 1, 2])
        
        v = OrbitUtil.convert_AU_per_year_to_km_per_sec(v)
        self.assertEqual(v.shape, (3,))