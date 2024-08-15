#added by Anney (for LF sampling)

import numpy as np
import scipy.interpolate


class Luminosity():
    """
    Implements function to draw samples of luminosity for error magnitude purposes
    """
    
#     def __init__(self):
#         self.RGB_LF = np.load('laszlo_RGB_LF.npy') 
#         self.RGB_LF_BINS = np.load('laszlo_RGBLF_bins.npy')

    
    @staticmethod
    def gen_noise(abs_magnitude, DM=19.2):
        apparent_magnitude = abs_magnitude + DM
        rv_err = np.exp(0.96 * apparent_magnitude - 18.8)
        noises = np.random.normal(scale=rv_err**0.5)
        return noises

    
#    def rv_error(abs_magnitude, DM=19.2):
#        #the error (variance) in radial velocity
#        apparent_magnitude = magnitude_conversion(abs_magnitude, DM)
#        rv_err = np.exp(0.96 * apparent_magnitude - 18.8)
#        #If the star is brighter, the magnitude is more negative ->smaller error, makes sense
#        return rv_err
    
'''
    def sample_magnitudes(self, size=1):
        #First, generate CDF from histogram; normalize
        RGB_LF = np.load('laszlo_RGB_LF.npy') 
        RGB_LF_BINS = np.load('laszlo_RGBLF_bins.npy')
        
        bin_midpoints = 0.5 * (RGB_LF_BINS[1:] + RGB_LF_BINS[:-1])
        cdf = np.cumsum(RGB_LF)
        cdf = cdf / cdf[-1] #normalize the cdf so it adds up to 1
        cdf = np.concatenate(([0],cdf), axis=None) #prepend cdf with 0 so the interpolation later doesn't throw "below the range" error
        #^Otherwise will get ValueError: A value in x_new is below the interpolation range.
        bin_midpoints_plus_leftedge = np.concatenate(([bin_midpoints[0] - (bin_midpoints[1]-bin_midpoints[0])/2],bin_midpoints), axis=None)
        inverse_interpolation = scipy.interpolate.interp1d(cdf, bin_midpoints_plus_leftedge) #swap cdf and the bin_midpoints so the interpolation comes out as inverse
        uniform_distr = np.random.rand(size)
        #print(uniform_distr)
        magnitudes = inverse_interpolation(uniform_distr)
        return magnitudes
'''
