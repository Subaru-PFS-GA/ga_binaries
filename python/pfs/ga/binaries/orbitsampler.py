import numpy as np
import scipy

from .orbitutil import OrbitUtil
from .orbits import Orbits

class OrbitSampler():
    """
    Implements function to draw samples of binary orbital elements in
    a convenient representation.
    """

    def __init__(self):
        self.m1 = 0.8               # Primary mass

    def _draw_mass_ratio_q(self, output_size=1, lower_bound=0.1, upper_bound=1.0, mu=0.23, sigma=0.42):
        """
        Mass ratio (q, unitless) follows a normal distribution with mean=0.23, stdev=0.42
        """
        
        a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        r = scipy.stats.truncnorm.rvs(a=a, b=b, loc=mu, scale=sigma, size=output_size)
        return np.atleast_1d(r)
        
    def _draw_period_logP(self, q, amin=0.21, amax=412.0, mu=4.8, sigma=2.3):
        """
        Period (P, units of day), sampled in log P 

        Sample directly from the truncated normal distribution for log P
        """

        logP_min = np.log10(OrbitUtil.calculate_period_from_major_axis(amin, self.m1, self.m1 * q))
        logP_max = np.log10(OrbitUtil.calculate_period_from_major_axis(amax, self.m1, self.m1 * q))
        a, b = (logP_min - mu) / sigma, (logP_max - mu) / sigma
        logP = scipy.stats.truncnorm.rvs(a=a, b=b, loc=mu, scale=sigma)
        return np.atleast_1d(logP)
    
    def _draw_eccentricity_e(self, a, logP, a_min=0.21, mu=0.31, sigma=0.17):
        """
        Sample excentiricity from a truncated normal distribution for periods
        shorter than 1000 days and uniform distribution above.
        """

        lb, ub = 0.0, 1.0 - (a_min / a)
        ta, tb = (lb - mu) / sigma, (ub - mu) / sigma
        r1 = scipy.stats.truncnorm.rvs(a=ta, b=tb, loc=mu, scale=sigma, size=a.shape)
        r2 = np.random.uniform(lb, ub, size=a.shape)
        return np.where(logP < 3, r1, r2)
    
    def _draw_initial_mean_anomaly_gamma0(self, size=1):
        """
        Initial mean anomaly (gamma_0, units of radian)
        
        Sample directly from uniform distribution ranging from 0 to 2pi
        """
        
        gamma0 = np.random.uniform(0, (2*np.pi), size=size)
        return np.atleast_1d(gamma0)
    
    def _draw_inclination_i(self, size=1):
        """
        Inclination (i, units of rad)

        Sampled using the inverse CDF method. CDF is given by 1-cos(x) after normalization.
        """

        def inverse_cdf(x):            
            return np.arccos(1 - x)

        i = inverse_cdf(np.random.uniform(size=size))
        return np.atleast_1d(i)
    
    def _draw_arg_of_periastron_omega(self, size=1):
        """
        Argument of periastron (omega, in units of rad)

        Sample from a uniform distribution between 0 and 2pi
        """
        
        omega = np.random.uniform(0, 2 * np.pi, size=size)
        return np.atleast_1d(omega)
            
    def sample(self, size=1):
        """
        Draw a sample of all orbital elements.
        """

        orb = Orbits()
        orb.q = self._draw_mass_ratio_q(size)
        orb.m1 = np.full_like(orb.q, self.m1)
        orb.m2 = orb.q * orb.m1
        orb.logP = self._draw_period_logP(orb.q)
        orb.a = OrbitUtil.calculate_major_axis_from_period(10**orb.logP, orb.m1, orb.m2)
        orb.e = self._draw_eccentricity_e(orb.a, orb.logP)
        orb.gamma0 = self._draw_initial_mean_anomaly_gamma0(size)
        orb.i = self._draw_inclination_i(size)
        orb.omega = self._draw_arg_of_periastron_omega(size)
        orb.theta0 = OrbitUtil.calculate_theta_iterative(orb.gamma0, orb.e)

        # TODO: move these elsewhere
        # orb.v_com = np.random.normal(0, orb.v_spread, size) #line added by Anney, samples center-of-mass velocities
        # orb.lum = self._sample_magnitudes(size) #added by Anney
        
        return orb
