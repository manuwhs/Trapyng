import numpy  
import pandas as pd  
import math as m
import indicators_pandas as indp

import utilities_lib as ul
#### FOR FUTURE ADDING MAYBE!!!

# The values is already a correct [Nsam, Nsig] numpy matrix
#Moving Average  
def SMA(self, df = None, *args, **kwargs):
    df = self.get_TD()
    SMA = indp.SMA(df,*args, **kwargs)
    return SMA

#Exponential Moving Average  
def EMA(self,  df = None, *args, **kwargs):
    df = self.get_TD()
    EMA = indp.EMA(df,*args, **kwargs)
    return EMA

#Pivot Points, Supports and Resistances  
def PPSR(self,  df = None, *args, **kwargs):  
    df = self.get_TD()
    PPSR = indp.PPSR(df)
    return PPSR
    
def FibboSR(self,  df = None, *args, **kwargs):  
    df = self.get_TD()
    FibboSR = indp.FibboSR(df)
    return FibboSR

def PSAR(self,  df = None, *args, **kwargs):  
    df = self.get_TD()
    PSAR = indp.PSAR(df)
    return PSAR
    
#Bollinger Bands  
def BBANDS(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    BBANDS = indp.BBANDS(df, *args, **kwargs)
    return BBANDS
    
def STD(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    STD = indp.STD(df, *args, **kwargs)
    return STD

#Average high low eange
def AHLR(self,df = None, *args, **kwargs):  
    df = self.get_TD()
    ATR = indp.AHLR(df, *args, **kwargs)
    return ATR
    
#Average True Range  
def ATR(self,df = None, *args, **kwargs):  
    df = self.get_TD()
    ATR = indp.ATR(df, *args, **kwargs)
    return ATR
    
def Chaikin_vol(self,df = None, *args, **kwargs):  
    df = self.get_TD()
    Chaikin_vol = indp.Chaikin_vol(df, *args, **kwargs)
    return Chaikin_vol

#Momentum  
def MOM(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    MOM = indp.MOM(df, *args, **kwargs)
    return MOM

#Rate of Change  
def ROC(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    ROC = indp.ROC(df, *args, **kwargs)
    return ROC

#Stochastic oscillator % 
def STO(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    STO = indp.STO(df, *args, **kwargs)
    return STO
    
#Stochastic oscillator %D  
def STOD(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    STO = indp.STOD(df, *args, **kwargs)
    return STO
#Stochastic oscillator %K
def STOK(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    STOK = indp.STOK(df, *args, **kwargs)
    return STOK
    
#Relative Strength Index  
def RSI(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    RSI = indp.RSI(df, *args, **kwargs)
    return RSI
    
#Average Directional Movement Index  
def ADX(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    ADX = indp.ADX(df, *args, **kwargs)
    return ADX
    
#Accumulation/Distribution  
def ACCDIST(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    ACCDIST = indp.ACCDIST(df, *args, **kwargs)
    return ACCDIST
    
#MACD, MACD Signal and MACD difference  
def MACD(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    MACD = indp.MACD(df, *args, **kwargs)
    return MACD
    
def TRIX(self, df = None, *args, **kwargs):  
    df = self.get_TD()
    TRIX = indp.TRIX(df, *args, **kwargs)
    return TRIX
#Trix  
## Oscillator similar to MACD

    