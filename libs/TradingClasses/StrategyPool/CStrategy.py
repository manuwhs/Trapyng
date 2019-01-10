
import pandas as pd
import numpy as np

import datetime as dt

import CStrategy_core as Stc
import CStrategy_XingAve as SXA
import CStrategy_RobustXingAve as SRXA
import CStrategy_KNNPrice as SKNP
import CStrategy_intraDayTimePatterns as SIDTP

class CStrategy:
    
    # Given a Symbol, this class uses its indicators to
    # formulate the BEST strategy for every market, depending on their parameters. 
    # It goes from detecting crossing averages to creating learning machines

    def __init__(self, Portfolio = []):
        if (Portfolio != []):
            self.set_Portfolio(Portfolio)
        
    ### Core functions
    set_Portfolio = Stc.set_Portfolio
        
    ### Strategies
    XingAverages = SXA.XingAverages
    RobustXingAverages = SRXA.RobustXingAverages
    
    KNNPrice = SKNP.KNNPrice 
    intraDayTimePatterns = SIDTP.intraDayTimePatterns
        
    def Heiken_Aki_view(self):
        print "Heiken- Aki"  
        # The goal of this strategy is to used precalculated rules from the HA plots,
        # detect them and use them. (Maybe like the KNN but computing similarit for...)
        # Detect unique super forms, forms by couples, by trends...
        
    def Gaussian_Process(self):
        # Predict using gaussian process
    
        print "Gaussian"
        
            

