import numpy  as np
import pandas as pd  
import math as m
import indicators_pandas as indp

import utilities_lib as ul

#### FOR FUTURE ADDING MAYBE!!!
def SMA(self, symbolIDs = [], period = None, *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        SMA = timeData_obj.SMA(*args, **kwargs)
        list_indicators.append(SMA)
        
    return list_indicators

#Exponential Moving Average  
def EMA(self,symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        EMA = timeData_obj.EMA(*args, **kwargs)
        list_indicators.append(EMA)
    return list_indicators

#Pivot Points, Supports and Resistances  
def PPSR(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        PPSR = timeData_obj.PPSR(*args, **kwargs)
        list_indicators.append(PPSR)
    return list_indicators

#Bollinger Bands  
def BBANDS(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        BBANDS = timeData_obj.BBANDS(*args, **kwargs)
        list_indicators.append(BBANDS)
    return list_indicators

#Average True Range  
def ATR(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        ATR = timeData_obj.ATR(*args, **kwargs)
        list_indicators.append(ATR)
    return list_indicators
    
#Momentum  
def MOM(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        MOM = timeData_obj.MOM(*args, **kwargs)
        list_indicators.append(MOM)
    return list_indicators

#Rate of Change  
def ROC(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        ROC = timeData_obj.ROC(*args, **kwargs)
        list_indicators.append(ROC)
    return list_indicators
#Stochastic oscillator %D  
def STO(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        STO = timeData_obj.STO(*args, **kwargs)
        list_indicators.append(STO)
    return list_indicators

#Relative Strength Index  
def RSI(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        RSI = timeData_obj.RSI(*args, **kwargs)
        list_indicators.append(RSI)
    return list_indicators
    
#Average Directional Movement Index  
def ADX(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        ADX = timeData_obj.ADX(*args, **kwargs)
        list_indicators.append(ADX)
    return list_indicators
    
#Accumulation/Distribution  
def ACCDIST(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        ACCDIST = timeData_obj.ACCDIST(*args, **kwargs)
        list_indicators.append(ACCDIST)
    return list_indicators
    
#MACD, MACD Signal and MACD difference  
def MACD(self, symbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        MACD = timeData_obj.MACD(*args, **kwargs)
        list_indicators.append(MACD)
    return list_indicators

#Trix  
## Oscillator similar to MACD
def TRIX(self, ssymbolIDs = [], period = None,  *args, **kwargs):
    symbolIDs, period = self.default_select(symbolIDs, period)
    list_indicators = [] # List of column vectors
    for symbol_n in symbolIDs:
        timeData_obj = self.symbols[symbol_n].timeDatas[period]
        SMA = timeData_obj.SMA(*args, **kwargs)
        list_indicators.append(SMA)
    return list_indicators

    