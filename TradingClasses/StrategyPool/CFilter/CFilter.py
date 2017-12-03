""" A Filter is just a function that assings a value from 0 to 1 to each time instant
This value will be used to determine if a given strategy is suitable for a given time.
For example, we can do a filter based on Variance, if the market is too volatile, we use
this information of the filter to:
    - Maybe ignore Buy-Sell signal from the main system
    - Maybe used to filter the samples the train the systm with.

Of course a Filter can be considered like any other feature, but the way to use it is
constrained and more human like. """
import CFilter_core as Sfc

import utilities_lib as ul
import numpy as np

class CFilter:
    ''' The information given to the filter can be the portfolio itself. 
    It should be able to compute all of the necesary information from scratch.
    Of course each filter can have optional input parameters that would give some
    precomputed data already, so we do not have to compute as much'''
    
    def __init__(self, Portfolio = []):
        if (Portfolio != []):
            self.set_Portfolio(Portfolio)
        
    ### Core functions
    set_Portfolio = Sfc.set_Portfolio
        
        
    def Filter_by_std(self):
        ' Basic filter that will'
        pass
    
    def get_ThresholdMask(self, feature, ths = None, reverse = False):
        # This function computes a mask of the values of a Matrix (Nsamples, Nfeatures)
        # ths is suposed to be a list of thresholds for the feature ths = [0.3, 0.5, 0.9]
        # The selection will have "1" where the selection was.
        # The samples to the left of the first th are "1" and then it conmutes with each new th. 
        # If you want the opposite then you use the "reverse"
        
        # TODO: Apply histeresis to the filters so that they not fluctuate as much ?
        # Feature is an Nx1 matrix
        feature = ul.fnp(feature) 
        # We order the ths in increasing order
        # TODO
        ths = ul.fnp(ths)
#        nan_mask = np.argwhere(feature.isNan())[:,0]
#        feature[nan_mask] = 0
        # We conmutate in the odd number of ths
        mask_sel = np.ones(feature.size) * (reverse^1)
        for i in range(ths.size):
            mask_sel_aux = np.argwhere(feature > ths[i])[:,0]
            mask_sel[mask_sel_aux] = (i % 2 == reverse^1) 
#        print "PENE"

            # TODO: We cannot do it like this if the original has Nans.
#        mask_sel[nan_mask] = 0
        mask_sel = mask_sel.astype(bool)
        return mask_sel
    
    def apply_Mask(self, dataMatrix, mask, replacement = None):
        ' This funciton puts the replacement value in the mask given as parameter'
        if (type(replacement) == type(None)):
            replacement = np.NaN
        X_data_aux = np.ones(dataMatrix.shape) * replacement 
        X_data_aux[mask,:] = dataMatrix[mask,:]
            
        return X_data_aux
        
def filter_by_risk():
    # This funciton will filter the samples used in the analysis 
    # 
    pass
    # We also should analyse abs(ret) pare detectar que cuando hay mucho riesgo
    # despues hay una tendancia clara.

