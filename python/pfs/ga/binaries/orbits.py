import numpy as np

from .orbitutil import OrbitUtil
from .observations import Observations

class Orbits():
    """
    Represents a list of binary star systems with known orbital parameters at t = 0
    and supports calculating observables at any future time.

    Variables:
    ----------
    q : array of float
        mass ratio m2 / m1
    m1 : array of float
        primary mass in solar masses
    m2 : array of float
        companion mass in solar masses
    logP : array of float
        log period in days
    a : array of float
        semi-major axis in AU
    e : array of float
        eccentricity
    gamma0 : array of float
        initial mean anomaly in radians
    i : array of float
        inclination in radians
    omega : array of float
        argument of periastron in radians
    theta0 : array of float
        initial true anomaly in radians
    """

    def __init__(self):
        # Orbital parameters
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

        # TODO: move these elsewhere
        self.v_spread = 5.1 #non-binary velocity sigma #line added by Anney
        self.v_com = None #added by Anney
        self.lum = None #added by Anney, luminosity for each star
        #luminoisity should be sampled only once & fixed per star, just like COM

    def to_dict(self, s=np.s_[:]):
        return dict(
            q = self.q[s],
            logP = self.logP[s],
            a = self.a[s],
            e = self.e[s],
            gamma0 = self.gamma0[s],
            i = self.i[s],
            omega = self.omega[s],
            theta0 = self.theta0[s]
        )
    
    def save(self, filename):
        raise NotImplementedError()
    
    def load(self, filename):
        raise NotImplementedError()

    def calculate_observables(self, t, s=np.s_[:], obs=None):
        """
        Calculate the observable quantities at time t.

        The are calculated by evaluating the orbits at time t
        and calculating the line-of-sight velocity.

        The observation epoch `t` can be either a scalar or a 1d array. In the latter case,
        the leading dimension of the returned observation will index the time.

        The slice `s` can be used to select a subset of the binaries.

        The return value is an `Observation` object with the following attributes:
        t : the time in units of days
        v_los : is in units of km/s

        Parameters:
        -----------
        t : float or array of float
            Time in days
        s : slice
            Slice to select a subset of the binaries
        obs : Observation
            Existing observation object to update. When None, a new object is created.

        Returns:
        --------
        obs : Observation
            simulated observation
        """

        gamma = OrbitUtil.calculate_gamma(self.gamma0[s], 10**self.logP[s], t)
        theta = OrbitUtil.calculate_theta_iterative(gamma, self.e[s])
        v_los = OrbitUtil.calculate_v_los(self.a[s], 10**self.logP[s] / OrbitUtil.DAYS_PER_YEAR, self.e[s], self.i[s], theta, self.omega[s])
        v_los = OrbitUtil.convert_AU_per_year_to_km_per_sec(v_los)

        if obs is None:
            obs = Observations()
        
        obs.t = t
        obs.v_los = v_los

        if np.size(t) > 1:
            obs.dv_los = v_los[1:] - v_los[0] 
        else:
            obs.dv_los = None

        return obs
    
    # TODO: what is this?
    # It appears to evaluate the orbits at equal time intervals for a full period
    # def simulate(self, gamma, s=np.s_[:]):
    #     """
    #     Evaluate the orbits at given delta gamma since gamma0 mean anomaly.

    #     P is in days
    #     t is in days
    #     """

    #     gamma = np.atleast_1d(gamma)
    #     gamma = gamma + self.gamma0[s]
    #     theta = OrbitUtil.calculate_theta_iterative(gamma, self.e[s])
    #     v_los = OrbitUtil.calculate_v_los(self.a[s], 10**self.logP[s] / OrbitUtil.DAYS_PER_YEAR, self.e[s], self.i[s], theta, self.omega[s])
    #     v_los = OrbitUtil.convert_AU_per_year_to_km_per_sec(v_los)
    #     v_los_err = np.full_like(v_los, np.nan)

    #     obs = Observables()
    #     obs.t = gamma / 2.0 / np.pi * 10 ** self.logP
    #     obs.v_los = v_los
    #     obs.v_los_err = v_los_err

    #     return obs
    
        