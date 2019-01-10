
import pandas as pd
import numpy as np

import datetime as dt

import CStrategy_core as Stc
import CStrategy_XingAve as SXA
import CStrategy_RobustXingAve as SRXA
import CStrategy_KNNPrice as SKNP
import CStrategy_intraDayTimePatterns as SIDTP


class CEventQueue:
    # This class is a Minimum Priority Queue for the Signals.
    # The key is the datetime. The earliest time is first.

    def __init__(self, Portfolio = []):
        self.Events = []
        
    def add_TradeEvents(self,events):
        # This function can be called to add events to the queue
        for event in events:
            self.Events.append(event)
    
    def send_EventsToBrain(self):
        # This functions will send the events to the Brain so that
        # it can process them.
        for event in self.Events:
            pass
            # Function to send a message to the Brain !!
    
    #### Backtesting ####
    def get_Event(self):
        # Function that will return the most priority event to respond.
        event =  self.Events.pop(0)
        return event 
        
class CStrategyPool:
    # This is a global element that will join all the strategies.

    def __init__(self, Portfolio = []):
        self.Sts = dict([])           # Dictionary of strategies
        
        # List of generated Events from all strategies, the events are ordered
        # in increasing order of time and timeFrame, so that we act first in
        #   - Sooner events in the lower timeframes.
        self.EventQueue = CEventQueue()    

    def add_strategy(self, strategy, strategy_ID):
        # This function adds a strategy to the Pool
        self.Sts[strategy_ID] = strategy
        
    def remove_strategy(self, strategyID):
        # Removes a strategy given by the ID.
        del self.Sts[strategyID]
    
    #### Backtesting Functions #####
    def get_Events(self):
        # This function computes the events for the whole backtesting !
        # It tells the Strategies to compute their Events !
        # and then it stores them in the Queue, which will order them 
        # timewise. From smaller time-frame to bigger time-frame
    
        for sts in self.Sts:
            events = sts.get_TradeEvents()
            self.EventQueue.add_TradeEvents(events)
        
        return self.EventQueue
    
        