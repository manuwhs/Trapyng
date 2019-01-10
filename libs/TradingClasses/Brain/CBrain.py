
import pandas as pd
import numpy as np

import datetime as dt

import CStrategy_core as Stc
import CStrategy_XingAve as SXA
import CStrategy_RobustXingAve as SRXA
import CStrategy_KNNPrice as SKNP
import CStrategy_intraDayTimePatterns as SIDTP

class CBrain:
    # In charged of receiving the TradingEvents.
    # Decided what to do
    def __init__(self, Portfolio = []):

        self.StrategyPool = None
        self.Colliseum = None
        self.MoneyManagement = None
        
        self.Policy = None # The Brains policy
        
    #### Backtesting Functions #####
    def manage_EntrySignal(self, entrySignal):
        # This function will manage an entry event.
        # The Brain will decide to enter the market or not 
        
        if(Policy = "All in"):
            # In this Policy, the Brain will always accept a trade, it will invest
            # a fixed amount of money to it.
            
            pass
        
        