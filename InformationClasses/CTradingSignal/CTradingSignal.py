# -*- coding: utf-8 -*-


import pandas as pd
import utilities_lib as ul
#### IMPORT the methods
import TimeData_core as TDc    # Core methods
import TimeData_graph as TDgr   # Graphics
import TimeData_DDBB as TDDB   #
import TimeData_indicators as TDind
import TiemData_pandasIndicators as TDindp

class CTradingSignal_seriesName:
    # This trading signal is meant to have a timeData object and return
    # the timeSeries names for the specific currency

    def __init__(self, tradingSignalID):
        self.tradingSignalID  = tradingSignalID  # Symbol of the Security (GLD, AAPL, IDX...)

    def set_input(self, inputDictionary = None):
        # Function to set all the inputs of the Signal. Usually:
        # Actual_data: Like a timeData or portfolio object
        # Parameters: Like the length of an SMA to apply to it.
        
        # "timeData": timeData object
        # "seriesNames": seriesNames to obtain
        
        self.inputs = inputDictionary
      
    def get_signal(self):
        # Implements logic over the inputs
        # Returns dates, signal.
        inp = self.inputs
        dates = inp["timeData"].get_dates()
        signal = inp["timeData"].get_timeSeries(inp["seriesNames"])
        return dates, signal
    
    def update_signal(self,new_input):
        # We will be given new input, like new samples for the timeData.
        # We have to add this new input (maybe delete some old one as well)
        # We compute the new samples, one per new input
        
        # In this case we expect the input to be new samples in TD format
        inp = self.inputs
        NewSamples, Ncol = new_input.size
        inp["timeData"].add_TD(new_input)
        
        
        pass

