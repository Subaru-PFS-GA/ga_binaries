import numpy as np
import scipy

class Orbits():
    """
    Implements function to draw samples of binary orbital elements in
    a convenient representation.
    """

    DAYS_PER_YEAR = 365.25
    DEFAULT_TOLERANCE = 1e-7
    MAX_ITERATIONS = 100
    
    @staticmethod
    def calculate_gamma(gamma0, P, t):
        """
        Calculate mean anomaly from initial value, period and time.

        gamma0 is initial mean anomaly in rad
        P is period in any time unit
        t is time in the same unit as P
        """

        gamma0 = np.atleast_1d(gamma0)
        P = np.atleast_1d(P)
        t = np.atleast_1d(t)
        
        # Take the floating point modulo to keep value in the interval of 0 and 2pi.
        # This is important to control numerical errors
        return (gamma0 + 2 * np.pi * t[:, None] / P) % (2 * np.pi)

    @staticmethod
    def calculate_theta_series(gamma, e):
        """
        Calculate the true anomaly from mean anomaly.

        This series expansion is only valid for _small_ values of the eccentricity e.
        Do not use, unless e is known to be small. Function only kept for reference
        Use orbits.true_anomaly_from_mean instead.
        """
        # gamma in rad
        return gamma + (2 * e - 1 / 4 * e**3) * np.sin(gamma) + \
                    5 / 4 * e**2 * np.sin(2 * gamma) + \
                    13 / 12 * e**3 * np.sin(3 * gamma)
    
    @staticmethod
    def calculate_theta_iterative(gamma, e, tolerance=DEFAULT_TOLERANCE):
        """
        Calculate the true anomaly from mean anomaly.

        This uses an iterative algorithm to solve the equation and works even for
        highly eccentric orbits.
        """
        return Orbits.calculate_true_anomaly_from_mean(e, gamma, tolerance=tolerance)

    @staticmethod
    def calculate_t_from_theta(E):
        """
        Calculate the time it takes to go from theta_0 to theta.

        P is period in any time unit
        t is time in the same unit as P
        """
        # E - 2 * np.pi / P * e * np.sin(E)
        raise NotImplementedError()
    
    @staticmethod
    def calculate_v_los(a, P, e, i, theta, omega):
        """
        Calculate the line of sight velocity at a given true anomaly.

        a is semi major axis in AU
        P is period in years
        e is eccentricity
        i is inclination
        theta is true anomaly in rad
        omega is the argument of periastron in rad
        v_los is in units of AU/year
        """
        
        # This is the equivalent of Evan's Eq 1.
        # Originally from Green (and Smart)
        # a in AU, P in years
        # results are in AU / year
        return 2 * np.pi / P * np.sin(i) / np.sqrt(1 - e**2) * (np.cos(theta + omega) + e * np.cos(omega))
    
    @staticmethod
    def calculate_major_axis_from_period(P, m1, m2):
        """
        Calculate the semi major axis from the period and masses
        using Kepler's III

        P is period in days
        m1 is mass in units of Solar masses
        m2 is mass in units of Solar masses
        a is in units of AU
        """

        P_years = P / Orbits.DAYS_PER_YEAR
        a = np.power((m1 + m2) * P_years**2, 1 / 3)
        return a

    @staticmethod
    def calculate_period_from_major_axis(a, m1, m2):
        """
        Calculate the period from the semi major axis and masses
        using Kepler's III

        a is in units of AU
        m1 is mass in units of Solar masses
        m2 is mass in units of Solar masses
        P is period in days
        """
        P_years = np.sqrt(a**3 / (m1 + m2))
        return P_years * Orbits.DAYS_PER_YEAR

    @staticmethod
    def calculate_eccentric_anomaly_from_mean(e, gamma, tolerance=DEFAULT_TOLERANCE):
        """Convert mean anomaly to eccentric anomaly.
        Implemented from [A Practical Method for Solving the Kepler Equation][1]
        by Marc A. Murison from the U.S. Naval Observatory
        [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
        """
        gamma_norm = np.fmod(gamma, 2 * np.pi)
        E0 = gamma_norm + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 / 2 * np.cos(gamma_norm) * e ** 3) * np.cos(gamma_norm)) * np.sin(gamma_norm)        
        dE = tolerance + 1      # Make sure the first iteration runs
        count = 0
        while np.any(dE > tolerance):
            t1 = np.cos(E0)
            t2 = -1 + e * t1
            t3 = np.sin(E0)
            t4 = e * t3
            t5 = -E0 + t4 + gamma_norm
            t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
            E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
            dE = abs(E - E0)
            E0 = E
            count += 1
            if count == Orbits.MAX_ITERATIONS:
                raise Exception('Did not converge after {n} iterations. (e={e!r}, M={M!r})'.format(n=Orbits.MAX_ITERATIONS, e=e, M=gamma))
        return E

    @staticmethod
    def calculate_true_anomaly_from_eccentric(e, E):
        """Convert eccentric anomaly to true anomaly."""
        return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    @staticmethod
    def calculate_true_anomaly_from_mean(e, gamma, tolerance=DEFAULT_TOLERANCE):
        """Convert mean anomaly to true anomaly."""
        E = Orbits.calculate_eccentric_anomaly_from_mean(e, gamma, tolerance)
        return Orbits.calculate_true_anomaly_from_eccentric(e, E)
    
    def convert_AU_per_year_to_km_per_sec(v):
        AU = 1.495978707e8 # km
        yr = 31557600 # s
        return v * AU / yr
    