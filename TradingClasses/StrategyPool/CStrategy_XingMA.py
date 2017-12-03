
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr

import utilities_lib as ul
import datetime as dt
import CEntrySignal as CES

class CStrategy_XingMA:
    # This strategy is given 2 MAs and it will output a 1 if the lines are crossing
    # We initialize it with the signals or with the portfolio and the shit.
    def __init__(self, StrategyID, period, pf = None):
        self.StrategyID = StrategyID    # ID of the strategy so that we can later ID it.
        self.period = period             # It is importan to know the period of the signal
        self.pf = pf 
        self.singalCounter = 0  # It will be the ID for the signals that we will generate
        
    def set_slowMA(self, SymbolName, period = 1440, L = 20, MAtype = "EMA"):
        self.slowMAparam = dict([["SymbolName", SymbolName], 
                                ["period", period], ["L",L], ["MAtype", MAtype]])
        
        timeDataObj = self.pf.get_timeData(SymbolName,period)
        if (MAtype == "EMA"):
            self.MA_fast = timeDataObj.EMA(n = L)
        elif(MAtype == "SMA"):
            self.MA_fast = timeDataObj.SMA(n = L)
            
        self.dates = timeDataObj.get_dates()
            
    def set_fastMA(self, SymbolName, period = 1440, L = 20, MAtype = "EMA"):
        self.slowMAparam = dict([["SymbolName", SymbolName], 
                                ["period", period], ["L",L], ["MAtype", MAtype]])

        timeDataObj = self.pf.get_timeData(SymbolName,period)
        if (MAtype == "EMA"):
            self.MA_slow = timeDataObj.EMA(n = L)
        elif(MAtype == "SMA"):
            self.MA_slow = timeDataObj.SMA(n =L)
            
        self.dates = timeDataObj.get_dates()
        
    def set_outsideMAs(self, MA_slow, MA_fast, dates):
        self.MA_slow = MA_slow
        self.MA_fast = MA_fast
        self.dates = dates
     
    #### BackTesting functions #######
     
    def get_TradeSignals(self):
        # Computes the BULL-SELL triggers of the strategy
        # Mainly for visualization of the triggers
        crosses = ul.check_crossing(self.MA_slow ,self.MA_fast)
        return crosses, self.dates
        
    def get_TradeEvents(self):
        # Creates the EntryTradingSignals for Backtesting
        crosses,dates = self.get_TradeSignals()
        self.singalCounter = 0
        list_events = []
        # Buy signals
        Event_indx = np.where(crosses != 0) # We do not care about the second dimension
        for indx in Event_indx[0]:
            if (crosses[indx] == 1):
                BUYSELL = "BUY"
            else:
                BUYSELL = "SELL"
            # Create the trading sigal !
            entrySignal =  CES.CEntrySignal(StrategyID = self.StrategyID, 
                                            EntrySignalID = str(self.singalCounter), 
                                            datetime = dates[indx], 
                                            symbolID = self.slowMAparam["SymbolName"], 
                                            BUYSELL = BUYSELL)
            entrySignal.comments = "Basic Crossing MA man !"
            
            entrySignal.priority = 0
            entrySignal.recommendedPosition = 1 
            entrySignal.tradingStyle = "dayTrading"
            
            list_events.append(entrySignal)
            self.singalCounter += 1
        return list_events