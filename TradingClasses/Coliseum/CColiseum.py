# -*- coding: utf-8 -*-

import copy
import pandas as pd
import utilities_lib as ul

import Coliseum_core as Coc
class CColiseum:
    
    def __init__(self, Warriors = ul.empty_coliseum, Portfolio = []):
        self.Warriors = copy.deepcopy(Warriors)    # Symbol of the Security (GLD, AAPL, IDX...)
        self.Portfolio = Portfolio    # It needs the portfolio to get the Symbols type and prices
        
        self.freeMargin = 0;   # Money we have not invested.
        self.moneyInvested = 0;  # Total of money invested in open positions
        self.marginLevel = 0;   # Nivel de apalanzamiento sobre el margen
        self.Equity = 0;       # How much money we would have if we close all positions
        self.Profit = 0;
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
    open_position = Coc.open_position
    get_position_indx = Coc.get_position_indx
    close_position = Coc.close_position
    add_position = Coc.add_position
    get_position_indx = Coc.get_position_indx
    close_position_by_indx = Coc.close_position_by_indx
    get_positions_symbol = Coc.get_positions_symbol
    
    update_prices = Coc.update_prices
    close_positions = Coc.close_positions
    
    set_date = Coc.set_date
    
    load_csv = Coc.load_csv
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
       



