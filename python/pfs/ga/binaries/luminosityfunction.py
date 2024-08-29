#added by Anney (for LF sampling)

import numpy as np
from scipy.interpolate import interp1d


class LuminosityFunction():
    """
    Implements functions to draw samples of luminosity in order to calculate
    velocity errors.

    The current implementation can sample the RGB of an old population only.
    """
    
    def __init__(self):
        # Approximate luminosity function of old binary RGB stars
        self.M = np.array([-3.52093398, -3.45072465, -3.38051532, -3.310306  , -3.24009667,
                            -3.16988734, -3.09967802, -3.02946869, -2.95925937, -2.88905004,
                            -2.81884071, -2.74863139, -2.67842206, -2.60821274, -2.53800341,
                            -2.46779408, -2.39758476, -2.32737543, -2.2571661 , -2.18695678,
                            -2.11674745, -2.04653813, -1.9763288 , -1.90611947, -1.83591015,
                            -1.76570082, -1.6954915 , -1.62528217, -1.55507284, -1.48486352,
                            -1.41465419, -1.34444486, -1.27423554, -1.20402621, -1.13381689,
                            -1.06360756, -0.99339823, -0.92318891, -0.85297958, -0.78277026,
                            -0.71256093, -0.6423516 , -0.57214228, -0.50193295, -0.43172362,
                            -0.3615143 , -0.29130497, -0.22109565, -0.15088632, -0.08067699,
                            -0.01046767,  0.05974166,  0.12995098,  0.20016031,  0.27036964,
                            0.34057896,  0.41078829,  0.48099762,  0.55120694,  0.62141627,
                            0.69162559,  0.76183492,  0.83204425,  0.90225357,  0.9724629 ,
                            1.04267222,  1.11288155,  1.18309088,  1.2533002 ,  1.32350953,
                            1.39371886,  1.46392818,  1.53413751,  1.60434683,  1.67455616,
                            1.74476549,  1.81497481,  1.88518414,  1.95539346,  2.02560279,
                            2.09581212,  2.16602144,  2.23623077,  2.3064401 ,  2.37664942,
                            2.44685875,  2.51706807,  2.5872774 ,  2.65748673,  2.72769605,
                            2.79790538,  2.86811471,  2.93832403,  3.00853336,  3.07874268,
                            3.14895201,  3.21916134,  3.28937066,  3.35957999,  3.42978931,
                            3.49999864])
        self.Phi = np.array([0.00250107, 0.00393025, 0.00226287, 0.00333476, 0.00333476,
                            0.00333476, 0.00393025, 0.00357296, 0.00488304, 0.00404935,
                            0.00428755, 0.00393025, 0.00500214, 0.00464485, 0.00500214,
                            0.00440665, 0.00512124, 0.00678862, 0.00559763, 0.00488304,
                            0.00619313, 0.00619313, 0.0085751 , 0.00726502, 0.00726502,
                            0.00762231, 0.00774141, 0.00928969, 0.00964699, 0.01119527,
                            0.01107617, 0.01167166, 0.01036158, 0.01286265, 0.01584012,
                            0.01953217, 0.02119955, 0.0167929 , 0.01786479, 0.01869848,
                            0.02119955, 0.02965555, 0.04001713, 0.02274783, 0.02250964,
                            0.02524891, 0.02477251, 0.02643989, 0.02858367, 0.03084654,
                            0.03144203, 0.03179933, 0.03084654, 0.03918344, 0.03811156,
                            0.03787336, 0.0427564 , 0.04775854, 0.04573387, 0.04942592,
                            0.05299888, 0.05264159, 0.05800102, 0.06145488, 0.06359866,
                            0.07884328, 0.07217376, 0.08432182, 0.09087224, 0.09813726,
                            0.09777996, 0.10957072, 0.11921771, 0.11802672, 0.13088937,
                            0.14363293, 0.16030673, 0.17007282, 0.19246336, 0.1882949 ,
                            0.20830347, 0.20222944, 0.21652128, 0.22926483, 0.23521976,
                            0.23772083, 0.2576103 , 0.27392681, 0.31191926, 0.3602733 ,
                            0.41839342, 0.49985687, 0.58263041, 0.66457025, 0.74996396,
                            0.83940701, 0.96719982, 1.12417179, 1.29460191, 1.43740114])

        
    # def _sample_magnitudes(self, size=1): #added by Anney
    #     #First, generate CDF from histogram; normalize
    #     RGB_LF = np.load('laszlo_RGB_LF.npy') 
    #     RGB_LF_BINS = np.load('laszlo_RGBLF_bins.npy')
        
    #     bin_midpoints = 0.5 * (RGB_LF_BINS[1:] + RGB_LF_BINS[:-1])
    #     cdf = np.cumsum(RGB_LF)
    #     cdf = cdf / cdf[-1] #normalize the cdf so it adds up to 1
    #     cdf = np.concatenate(([0],cdf), axis=None) #prepend cdf with 0 so the interpolation later doesn't throw "below the range" error
    #     #^Otherwise will get ValueError: A value in x_new is below the interpolation range.
    #     bin_midpoints_plus_leftedge = np.concatenate(([bin_midpoints[0] - (bin_midpoints[1]-bin_midpoints[0])/2],bin_midpoints), axis=None)
    #     inverse_interpolation = scipy.interpolate.interp1d(cdf, bin_midpoints_plus_leftedge) #swap cdf and the bin_midpoints so the interpolation comes out as inverse
    #     uniform_distr = np.random.rand(size)
    #     #print(uniform_distr)
    #     magnitudes = inverse_interpolation(uniform_distr)
    #     return magnitudes

    def sample(self, size=1, M_min=None, M_max=None, smooth=False):
        """
        Draw a sample of absolute magnitudes from the luminosity function using
        the inverse CDF method.

        Parameters:
        -----------
        size : int or tuple
            number of samples to draw
        """

        if smooth:
            # Make the CDF a bit smoother by first interpolating the PDF
            lf = interp1d(0.5 * (self.M[1:] + self.M[:-1]), self.Phi,
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
            M = np.linspace(self.M[0], self.M[-1], 1000)
            Phi = lf(0.5 * (M[1:] + M[:-1]))
        else:
            M = self.M
            Phi = self.Phi

        # First, generate CDF from the histogram and normalize it
        cdf = np.concatenate([[0], np.cumsum(Phi)])
        cdf /= cdf[-1]

        # Interpolate the inverse of the cumulative luminosity function
        inv_lf = interp1d(cdf, M, kind='linear',
                        fill_value=(0.0, 1.0), bounds_error=False)

        # Sample from the uniform distribution and interpolate the magnitudes
        r = np.random.random_sample(size)
        M = inv_lf(r)

        return M
    
    

