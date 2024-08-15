import numpy as np

from .orbits import Orbits
from .observation import Observation

class Binaries():
    def __init__(self):
        self.q = None               # mass ratio
        self.m1 = None              # primary mass (m1 > m2)
        self.m2 = None              # secondary mass
        self.logP = None            # log period [days]
        self.a = None               # semi major axis [AU]
        self.e = None               # excentricity
        self.gamma0 = None          # initial mean anomaly [rad]
        self.i = None               # inclination [rad]
        self.omega = None           # argument of periastron [rad]
        self.theta0 = None          # initial true anomaly [rad]
        self.K = None               # Semi-amplitude

    def to_dict(self, s=np.s_[:]):
        return {
            'q': self.q[s],
            'logP': self.logP[s],
            'a': self.a[s],
            'e': self.e[s],
            'gamma0': self.gamma0[s],
            'i': self.i[s],
            'omega': self.omega[s],
            'theta0': self.theta0[s]
        }

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
    
    def simulate(self, gamma, s=np.s_[:]):
        """
        Evaluate the orbits at given delta gamma since gamma0 mean anomaly.

        P is in days
        t is in days
        """

        gamma = np.atleast_1d(gamma)
        gamma = gamma + self.gamma0[s]
        theta = Orbits.calculate_theta_iterative(gamma, self.e[s])
        v_los = Orbits.calculate_v_los(self.a[s], 10**self.logP[s] / Orbits.DAYS_PER_YEAR, self.e[s], self.i[s], theta, self.omega[s])
        v_los = Orbits.convert_AU_per_year_to_km_per_sec(v_los)
        v_los_err = np.full_like(v_los, np.nan)

        obs = Observation()
        obs.t = gamma / 2.0 / np.pi * 10 ** self.logP
        obs.v_los = v_los
        obs.v_los_err = v_los_err

        return obs
    
    # TODO: add method to observe with a mask to simulate 