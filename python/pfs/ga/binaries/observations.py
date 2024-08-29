import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import multivariate_normal

class Observations():
    """
    Represents observations of binary star systems at given epochs. The class
    also supports searching for similar binaries based on velocity differences
    measured at the epochs.
    """

    def __init__(self):
        # Observables
        self.t = None               # epoch at which the observables are calculated
        self.v_los = None           # line-of-sight velocity [km/s]
        self.dv_los = None          # line-of-sight velocity difference [km/s]

        # Indexes
        self.dv_los_kd_index = None

    def to_dict(self, s=np.s_[:]):
        return dict(
            t = self.t,
            v_los = self.v_los[..., s],
            dv_los = self.dv_los[..., s],
        )
    
    def generate_dv_los_index(self):
        """
        Generate a kd-tree index for the velocity differences.

        The velocity differences are calculated by subtracting the line-of-sight velocity
        at the very first epoch (index zero) from the rest of the epochs.
        """

        # TODO: run verifications

        x = np.transpose(self.dv_los)
        self.dv_los_kd_index = KDTree(x, leaf_size=40, metric='euclidean')

    def search_dv_los_index(self, dv, cov, sigma=3):
        """
        Given a list of observed velocity differences `dv`, and the associated covariance matrix `cov`,
        look up the orbits with similar velocity differences.

        Parameters:
        -----------
        dv : array_like
            observed velocity differences, at the same epochs as the observables
        cov : array_like
            covariance matrix of the velocity differences
        sigma : float
            search radius in Maholanobis distance units

        Returns:
        --------
        idx : array
            indexes of the orbits that are within the search radius
        d : array
            Mahalanobis distance of the orbits within the search radius
        p : array
            probability of the orbits within the search radius
        """

        dv = np.atleast_2d(dv)

        # The largest eigenvalue of the covariance matrix determines the search radius
        eigs, eigv = np.linalg.eig(cov)
        rad = np.max(np.sqrt(eigs)) * sigma

        # Search the kd-tree
        idx = self.dv_los_kd_index.query_radius(dv, r=rad)[0]

        # Filter down the points to those that are within the ellipse
        v = self.dv_los[:, idx].T - dv
        d = np.einsum('ik,kl,il->i', v, np.linalg.inv(cov), v)
        mask = d < sigma**2

        # Evaluate the probability for each point within the ellipse
        # p = np.exp(-0.5 * d[mask])
        p = multivariate_normal(mean=dv[0], cov=cov).pdf(self.dv_los[:, idx[mask]].T)

        return idx[mask], d[mask], p
