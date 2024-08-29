import numpy as np

class VelocityError():
    """
    Model the velocity error as a function of the absolute magnitude of the star.
    """

    def __init__(self):
        pass

    # @staticmethod
    # def gen_noise(abs_magnitude, DM=19.2):
    #     apparent_magnitude = abs_magnitude + DM
    #     rv_err = np.exp(0.96 * apparent_magnitude - 18.8)
    #     noises = np.random.normal(scale=rv_err**0.5)
    #     return noises

    
#    def rv_error(abs_magnitude, DM=19.2):
#        #the error (variance) in radial velocity
#        apparent_magnitude = magnitude_conversion(abs_magnitude, DM)
#        rv_err = np.exp(0.96 * apparent_magnitude - 18.8)
#        #If the star is brighter, the magnitude is more negative ->smaller error, makes sense
#        return rv_err

    def eval(self, M, DM=19.2):
        """
        Given an absolute magnitude and a distance modulus, return the typical (mean) velocity error.
        """

        # This is a simple fit of the results from the paper
        v_los_err = np.exp(0.82 * M - 16.55) + 0.6

        return v_los_err
    
    def sample(self, M, DM=19.2, size=None):
        """
        Given an absolute magnitude and a distance modulus, return a sample of velocity errors.
        """

        if size is None:
            size = ()
        elif not isinstance(size, tuple):
            size = tuple((size,))

        a = np.exp(0.82 * np.atleast_1d(M) - 16.55) + 0.6
        s = np.random.normal(loc=0.0, scale=0.2, size=a.shape + size)

        if size == ():
            v_los_err = np.exp(s) * a
        else:
            v_los_err = np.exp(s) * a[..., None]

        if np.size(v_los_err) == 1:
            return v_los_err.item()
        else:
            return v_los_err.squeeze()

