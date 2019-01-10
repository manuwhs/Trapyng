# -*- coding: utf-8 -*-


import pandas as pd
import utilities_lib as ul
#### IMPORT the methods
import TimeData_core as TDc    # Core methods
import TimeData_MAs as TDMAs   # Moving Averages 
import TimeData_oscil as TDos   # Ocillators
import TimeData_volat as TDvo  # Volatility
import TimeData_graph as TDgr   # Graphics
import TimeData_DDBB as TDDB   # Graphics

import MoneyManagement_core as CMyMc

class CMoneyManagement:
    
    def __init__(self, Coliseum = ul.empty_coliseum):
        self.Coliseum = Coliseum    # Symbols we have an open position
        
        self.freeMargin = 0;   # Money we have not invested.
        self.moneyInvested = 0;  # Total of money invested in open positions
        self.marginLevel = 0;   # Nivel de apalanzamiento sobre el margen
        self.Equity = 0;       # How much money we would have if we close all positions
        
        #########################################################################
        ########### For BackTesting we need to fake the real time ###############
        #########################################################################
        self.imaginaryDate = 0;  
    #######################################################################
    ############## DDBB methods ###########################################
    #######################################################################

    #######################################################################
    ############## CORE Methods ###########################################
    #######################################################################
    process_new_actions = CMyMc.process_new_actions
    set_date = CMyMc.set_date
    #######################################################################
    ############## Moving Averages  ###########################################
    #######################################################################
    

    
    
    #######################################################################
    ############## Ocillators  ###########################################
    #######################################################################
    
   
    #######################################################################
    ############## Graphics  ###########################################
    #######################################################################
   

    
#pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
#       keys=None, levels=None, names=None, verify_integrity=False)
       



