import numpy as np

from .orbits import Orbits
from .observation import Observation

class Binaries():
    def __init__(self):
        self.q = None
        self.m1 = None
        self.m2 = None
        self.logP = None
        self.a = None
        self.e = None
        self.gamma0 = None
        self.i = None
        self.omega = None
        self.theta0 = None

    def observe(self, t, s=np.s_[:]):
        """
        Generate an observation at time t.

        t is the time in units of days
        v_los is in units of km/s
        """

        gamma = Orbits.calculate_gamma(self.gamma0[s], 10**self.logP[s], t)
        theta = Orbits.calculate_theta_iterative(gamma, self.e[s])
        v_los = Orbits.calculate_v_los(self.a[s], 10**self.logP[s] / Orbits.DAYS_PER_YEAR, self.e[s], self.i[s], theta, self.omega[s])
        v_los = Orbits.convert_AU_per_year_to_km_per_sec(v_los)
        v_los_err = np.full_like(v_los, np.nan)

        obs = Observation()
        obs.t = t
        obs.v_los = v_los
        obs.v_los_err = v_los_err

        return obs
    
    # TODO: add method to observe with a mask to simulate 