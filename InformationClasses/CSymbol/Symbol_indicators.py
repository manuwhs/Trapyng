import numpy  
import pandas as pd  
import math as m
import indicators_pandas as indp

import utilities_lib as ul
#### FOR FUTURE ADDING MAYBE!!!

# The values is already a correct [Nsam, Nsig] numpy matrix
#Moving Average  

# Function to select the period if not specified
def select_period(period, periods):
    if (period == -1): # If no period specified we take all
        if (1440 in periods):
            period = 1440
        else:
            period = max(periods)
        return period
        
def SMA(self, period = -1, *args, **kwargs):
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    SMA = timeData.SMA(*args, **kwargs)
    return SMA
    
#Exponential Moving Average  
def EMA(self, period = -1, *args, **kwargs):
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    EMA = timeData.EMA(*args, **kwargs)
    return EMA

#Pivot Points, Supports and Resistances  
def PPSR(self,  period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    PPSR = timeData.PPSR(*args, **kwargs)
    return PPSR

#Bollinger Bands  
def BBANDS(self, period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    BBANDS = timeData.BBANDS(*args, **kwargs)
    return BBANDS

#Average True Range  
def ATR(self, period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    ATR = timeData.ATR(*args, **kwargs)
    return ATR
    
#Momentum  
def MOM(self, period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    MOM = timeData.MOM(*args, **kwargs)
    return MOM

#Rate of Change  
def ROC(self, period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    ROC = timeData.ROC(*args, **kwargs)
    return ROC

#Stochastic oscillator %D  
def STO(self, period = -1,*args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    STO = timeData.STO(*args, **kwargs)
    return STO

#Relative Strength Index  
def RSI(self, period = -1,*args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    RSI = timeData.RSI(*args, **kwargs)
    return RSI
    
#Average Directional Movement Index  
def ADX(self, period = -1,*args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    RSI = timeData.RSI(*args, **kwargs)
    return RSI
    
#Accumulation/Distribution  
def ACCDIST(self, period = -1,*args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    ACCDIST = timeData.ACCDIST(*args, **kwargs)
    return ACCDIST
    
#MACD, MACD Signal and MACD difference  
def MACD(self, period = -1, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    MACD = timeData.MACD( *args, **kwargs)
    return MACD

#Trix  
## Oscillator similar to MACD
def TRIX(self, period = -1, df = None, *args, **kwargs):  
    period = select_period(period, self.periods)
    timeData = self.TDs[period]
    TRIX = timeData.TRIX(*args, **kwargs)
    return TRIX
    