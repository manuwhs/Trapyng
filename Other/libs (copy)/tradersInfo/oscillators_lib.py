
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import basicMathlib as bMa
import indicators_lib as indl



##############################################################################
################## MOMENTUM INDICATORS #######################################
##############################################################################
""" 
 Generally speaking, momentum measures the rate-of-change of a security's price. 
 As the price of a security rises, price momentum increases. 
 The faster the security rises 
 (the greater the period-over-period price change), 
the larger the increase in momentum. Once this rise begins to slow, 
momentum will also slow.
""" 
def get_momentum (time_series, N = 1):
    """ 
    The Momentum Technical Indicator measures the amount that a 
    security's price has changed over a given time span. 
    It is used with the closed price
     There are several variations of the momentum indicator,
     but whichever version is used, the momentum (M) is a
     comparison between the current closing price (CP) a closing price 
     "n" periods ago (CPn). The "n" is determined by you. 
     In the attached chart, Momentum is set to "10," so the indicator 
     is comparing the current price to the price 10 time instances ago 
     (because it is a 1-minute chart).
    """
#    diff = bMa.diff(time_series, lag = 1, n = 2)
#    diff = bMa.diff([1,2,3,4,5,6,7,8], lag = 1, n = 2)
#    print diff[0:10,:]
#    print time_series[0:10,:]

    momentum = bMa.diff(time_series, lag = N, n = 2)
    
    return momentum


def get_RSI(prices, n=14):
    """
    The relative strength index (RSI) is a momentum indicator that 
    compares the magnitude of recent gains and losses 
    over a specified time period to measure 
    speed and change of price movements of a security. 
    It is primarily used to attempt to identify overbought 
    or oversold conditions in the trading of an asset.
    """
    
    # First we calculate the increases to then separate them into
    # increasing instances and decreasing instances
    deltas = bMa.diff(prices, lag = 1)
    
    # For the first N samples we will not be able to do it properli

    ## What we do is a rolling mean of the positive and the negative

    dUp, dDown = deltas.copy(), deltas.copy()
    ## Vector of positive and negative
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    
    # Calculate the rolling mean, the Window !!
    # Calculates the average dUp and dDown in time
    RolUp = pd.rolling_mean(dUp, n)
    RolDown = np.abs(pd.rolling_mean(dDown, n))

#    print RolUp[0:30]
    # To avoid division by 0
    RS = RolUp / (RolDown +0.0000001)
    RSI = 100. - 100. / (1. + RS)
    return RSI
    
# This uses the smoothing
def get_RSI2(prices, n=14):
#    n = float(n)
    prices = prices[:,0].T.tolist()
    
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros(len(prices))
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
    # It is wrong, it does not remove the other delta
    # Wilder-smoothing = ((previous smoothed avg * (n-1)) + current value to average) / n
    # For the very first "previous smoothed avg" (aka the seed value), we start with a straight average.
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/(down + 0.0000001)
#        print rs
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def get_MACD (time_series, Ls = 12, Ll = 26, Lsmoth = 9, alpha = -1):
    """ 
    Moving Average Convergence/Divergence (MACD) indicates the correlation between 
    two price moving averages.
    
    Usually 26-period and 12-period Exponential Moving Average (EMA).
    In order to clearly show buy/sell opportunities, 
    a so-called signal line (9-period indicators` moving average) is plotted on the MACD chart.
    The MACD proves most effective in wide-swinging trading markets. 
    There are three popular ways to use the Moving Average Convergence/Divergence: 
    crossovers, overbought/oversold conditions, and divergences.

    The MACD is calculated by subtracting the value of a 26-period exponential 
    moving average from a 12-period exponential moving average. 
    A 9-period dotted simple moving average of the MACD (the signal line) 
    is then plotted on top of the MACD.
    """
    
    eMlong = indl.get_EMA(time_series, Ll, alpha)
    eMshort = indl.get_EMA(time_series, Ls, alpha)
    
    MACD = indl.get_SMA(eMshort - eMlong, Lsmoth)
    
    return eMlong, eMshort, MACD
    

def get_StochOsc(timeData, N = 14):
    """
    The stochastic oscillator is a momentum indicator comparing 
     - the closing price of a security 
     - to the range of its prices 
    over a certain period of time. 
    The sensitivity of the oscillator to market movements 
    is reducible by adjusting that time period or by taking 
    a moving average of the result.

    %K = 100(C - L14)/(H14 - L14)
    Where:
    C = the most recent closing price
    L14 = the low of the 14 previous trading sessions
    H14 = the highest price traded during the same 14-day period
    %K= the current market rate for the currency pair
    %D = 3-period moving average of %K
    """
    pass
