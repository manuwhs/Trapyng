##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr

import utilities_lib as ul
import datetime as dt
import CEntrySignal as CES
import CExitSignal as CExS

class CTrailingStop:
    # We get out if the price goes a certain amount against us.
    # We set the period as the period of checking and putting 
    # the new stop:
    #   - We will update the real platform stop each end of the period.
    #   - If duting the last period, the stop was in the range of the candle
    #     then we close the deal

    def __init__(self, StrategyID, period, pf = None):
        self.StrategyID = StrategyID    # ID of the strategy so that we can later ID it.
        self.period = period             # It is importan to know the period of the signal
        self.pf = pf 
        self.singalCounter = 0  # It will be the ID for the signals that we will generate
    
        # Parameters regarding the entry.
        # If init_price and date are not provided, we just assume are the 
        # init of t

        self.init_datetime = None
        self.init_price = None
        self.init_index = None   # Index of the sample that contais the date
    # Set the parameters of the trailing stop: Commodity, period and % 
    def set_trailingStop(self, SymbolName, BUYSELL, datetime = None, period = 1440, maxPullback = 3): 
        self.trailingParam = dict([["SymbolName", SymbolName], ["date",datetime],
                                ["period", period], ["maxPullback",maxPullback]])
        self.maxPullback = maxPullback
        self.timeDataObj = self.pf.symbols[SymbolName].get_timeData(period)
        self.dates = self.timeDataObj.get_dates()
        self.BUYSELL = BUYSELL
        # TODO: Preprocess datetome so that it fits in a given interval
        self.init_datetime = datetime
        
    # These are the parameters of the simulated entry.
    # The time and price where we entered the market.
    # It does not have to be same as when the EntrySignal
    # was activated due to slipage
        # TODO: Set interval for the original date ?
        
    def set_outsideMAs(self, MA_slow, MA_fast, dates):
        self.MA_slow = MA_slow
        self.MA_fast = MA_fast
        self.dates = dates
     
    #### BackTesting functions #######
     
    def get_TradeSignals(self):
        pLow = self.timeDataObj.get_timeSeries(["Low"])
        pHigh = self.timeDataObj.get_timeSeries(["High"])

        if (self.BUYSELL == "BUY"):
            pUpdate = pHigh   # Price used to update the line
            pCheckCross = pLow
            self.maxPullback = -np.abs(self.maxPullback)
        else:
            pUpdate = pLow   # Price used to update the line
            pCheckCross = pHigh
            self.maxPullback = np.abs(self.maxPullback)
            
        if (type(self.init_datetime) == type(None)):
            self.init_index = 0
            self.init_date = self.dates[self.init_index]
        
        self.init_index = int(np.where(self.dates == self.init_datetime)[0])
        
        # Now we get the REAL price for what we bought it. It could be within
        # the candle so we do not really know from the historic, what it is exactly
        if (type(self.init_price) == type(None)):
            # If we do not have the exect init_price, se use the midpoint HL
            self.init_price = (pUpdate[self.init_index,0] + pCheckCross[self.init_index,0])/2
        
        init_stop = self.init_price *(1 + self.maxPullback/100.0)                        
        
        # We start the computing. Compute the trailing stop
        Nsamples = self.dates.size
        all_stops = np.zeros(pCheckCross.shape) * np.NaN
        
        all_stops[self.init_index] = init_stop
        
        print (self.init_index +1, Nsamples)
        for i in range (self.init_index +1,Nsamples):
            init_stop_i = pUpdate[i-1] *(1 + self.maxPullback/100.0)
#                print init_stop_i[0], all_stops[i-1]
            if (self.BUYSELL == "BUY"):
                all_stops[i] = np.nanmax([all_stops[i-1],init_stop_i[0]])
            else:
                all_stops[i] = np.nanmin([all_stops[i-1],init_stop_i[0]])
        # Get the crosses
        crosses = ul.check_crossing(pCheckCross , all_stops)
        
        # Remove positive and regative crosses it they do not belong
        if (self.BUYSELL == "BUY"):
            crosses[np.where(crosses == 1)] = 0
        else:
            crosses[np.where(crosses == -1)] = 0
            
        return crosses, all_stops, pCheckCross, self.dates
        
    def get_TradeEvents(self):
        # Creates the EntryTradingSignals for Backtesting
        crosses,dates = self.get_TradeSignals()
        
        list_events = []
        # Buy signals
        Event_indx = np.where(crosses != 0 ) # We do not care about the second dimension
        for indx in Event_indx[0]:
            # Create the Exit signal !
            if crosses[Event_indx] == 1:
                BUYSELL = "BUY"
            else:
                BUYSELL = "SELL"
                
            entrySignal =  CExS.CExitSignal(StrategyID = self.StrategyID, 
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
