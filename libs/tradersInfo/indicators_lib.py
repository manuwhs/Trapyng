
import numpy as np
import copy
import utilities_lib as ul
from graph_lib import gl
import pandas as pd

####

def MDD(timeSeries, window):
    """
    A maximum drawdown (MDD) is the maximum loss from a peak to a trough of a portfolio, 
    before a new peak is attained. 
    Maximum Drawdown (MDD) is an indicator of downside risk over a 
    specified time period. It can be used both as a stand-alone 
    measure or as an input into other metrics such as "Return 
    over Maximum Drawdown" and Calmar Ratio. Maximum Drawdown 
    is expressed in percentage terms and computed as:
    Read more: Maximum Drawdown (MDD) Definition 
    """
    
    ## Rolling_max calculates puts the maximum value seen at time t.
    # We
    
    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = pd.rolling_max(timeSeries, window, min_periods=1)
    
    Roll_Max = ul.fnp(Roll_Max)
    print (Roll_Max.shape)
    # How much we have lost compared to the maximum so dar
    Daily_Drawdown = timeSeries/Roll_Max - 1.0
    print (Daily_Drawdown.shape)
    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)
    
    Max_Daily_Drawdown = ul.fnp(Max_Daily_Drawdown)
    return Daily_Drawdown, Max_Daily_Drawdown

## Another implementation that should give different drawdows in the series

def drawdowns(equity_curve):
    i = np.argmax(np.maximum.accumulate(equity_curve.values) - equity_curve.values) # end of the period
    j = np.argmax(equity_curve.values[:i]) # start of period

    drawdown=abs(100.0*(equity_curve[i]-equity_curve[j]))

    DT=equity_curve.index.values

    start_dt=pd.to_datetime(str(DT[j]))
    MDD_start=start_dt.strftime ("%Y-%m-%d") 

    end_dt=pd.to_datetime(str(DT[i]))
    MDD_end=end_dt.strftime ("%Y-%m-%d") 

    NOW=pd.to_datetime(str(DT[-1]))
    NOW=NOW.strftime ("%Y-%m-%d")

    MDD_duration=np.busday_count(MDD_start, MDD_end)

    try:
        UW_dt=equity_curve[i:].loc[equity_curve[i:].values>=equity_curve[j]].index.values[0]
        UW_dt=pd.to_datetime(str(UW_dt))
        UW_dt=UW_dt.strftime ("%Y-%m-%d")
        UW_duration=np.busday_count(MDD_end, UW_dt)
    except:
        UW_dt="0000-00-00"
        UW_duration=np.busday_count(MDD_end, NOW)

    return MDD_start, MDD_end, MDD_duration, drawdown, UW_dt, UW_duration
    
########################################################
########## Efficient Moving Averages ###################
########################################################

# They are not for predicting, they are calculated using also the measurement
# fot time t, but to get the prediction we just have to shift it once to the right.

def get_convol(signal, window, cval = np.NaN):
        L = window.size
        sM = np.convolve(signal.flatten(),window.flatten(), mode = "full")
        sM[:L] = sM[:L] * cval
        sM = sM[:-L+1]    # Remove the last values since they hare convolved with 0's as well
        sM = sM.reshape ((sM.size,1))
        return sM
        
def get_SMA(timeSeries, L, cval = np.NaN):
    """ Outputs the aritmetic mean of the time series using 
    a rectangular window of size L"""
    timeSeries = ul.fnp(timeSeries)
    Nsam, Nsig = timeSeries.shape
    
    # Create window
    window = np.ones((L,1))
    window = window/np.sum(window) 
    for si in range(Nsig):
        ## Convolution for one of the signals
        signal = timeSeries[:,si]
        sM = get_convol(signal,window,cval)
        if (si == 0):
            total_sM = copy.deepcopy(sM)
        else:
            total_sM = np.concatenate((total_sM,sM), axis = 1)

    return total_sM

def get_WMA(timeSeries, L, cval = np.NaN):
    """ Outputs the aritmetic mean of the time series using 
    a linearly descendent window of size L"""
    Nsam, Nsig = timeSeries.shape
    # Create window
    window = np.cumsum(np.ones((L,1))) / L   
    inverse_range = -np.sort(-np.array(range(0,int(L))))
    inverse_range = inverse_range.tolist()
#        print inverse_range
    window = window[inverse_range]
    window = window/np.sum(window) 
        
    for si in range(Nsig):
        ## Convolution for one of the signals
        signal = timeSeries[:,si]
        sM = get_convol(signal,window,cval)
        if (si == 0):
            total_sM = copy.deepcopy(sM)
        else:
            total_sM = np.concatenate((total_sM,sM), axis = 1)
            
    return total_sM
    
def get_EMA(timeSeries, L, alpha = -1, cval = np.NaN):
    L = int(L)
    if (alpha == -1):
        alpha = 2.0/(L+1)
        
    """ Outputs the exponential mean of the time series using 
    a linearly descendent window of size L"""
    Nsam, Nsig = timeSeries.shape
    window = np.ones((L,1))
    factor = (1 - alpha)
    for i in range(L):
        window[i] *= factor
        factor *= (1 - alpha)
    window = window/np.sum(window) 
    
    for si in range(Nsig):
        ## Convolution for one of the signals
        signal = timeSeries[:,si]
        sM = get_convol(signal,window,cval)
        if (si == 0):
            total_sM = copy.deepcopy(sM)
        else:
            total_sM = np.concatenate((total_sM,sM), axis = 1)
        
    return total_sM

def get_TrCrMr (time_series, alpha = -1):
    """ Triple Cruce de la Muerte. Busca que las exponenciales 4, 18 y 40 se crucen
    para ver una tendencia en el mercado despues de un tiempo lateral """
    L1 = 4
    L2 = 18
    L3 = 40
    
    eM1 = get_EMA(time_series, L1, alpha)
    eM2 = get_EMA(time_series, L2, alpha)
    eM3 = get_EMA(time_series, L3, alpha)
    
    return np.concatenate((eM1,eM2,eM3), axis = 1)

def get_HMA (time_series, L, cval = np.NaN):
    """ Hulls Moving Average !! L = 200 usually"""
    WMA1 = get_WMA(time_series, L/2, cval = cval) * 2
    WMA2 = get_WMA(time_series, L, cval = cval)
    
    HMA = get_WMA(WMA1 - WMA2, np.sqrt(L), cval =cval)
    
    return HMA
    
def get_HMAg (time_series, L, alpha = -1,  cval = np.NaN):
    """ Generalized Moving Average from Hull"""
    ## Moving Average of 2 moving averages.
    ## It uses Exponential Moving averages
    EMA1 = get_EMA(time_series, L/2, alpha, cval = cval) * 2
    EMA2 = get_EMA(time_series, L, alpha, cval = cval)
    
    EMA = get_EMA(EMA1 - EMA2, np.sqrt(L), alpha, cval = cval)
    
    return EMA


def get_TMA(time_series, L):
    """ First it trains the data so that the prediction is maximized"""
    ### Training phase, we obtained the MSQE of the filter for predicting the next value
    time_series = time_series.flatten()
    Ns = time_series.size
    
    Xtrain, Ytrain = ul.windowSample(time_series, L)
    

    window = np.linalg.pinv((Xtrain.T).dot(Xtrain))
    window = window.dot(Xtrain.T).dot(Ytrain)
    window = np.fliplr([window])[0]
    gl.stem([],  window)

    sM = np.convolve(time_series.flatten(),window.flatten(), mode = "full")
    
#    print sM.shape
    sM = sM
    sM = sM/np.sum(window)    # Divide so that it is the actual mean.
    
    sM[:L] = (np.ones((L,1)) * sM[L]).flatten()  # Set the first ones equal to the first fully obtained stimator
    sM = sM[:-L+1]    # Remove the last values since they hare convolved with 0's as well
    return sM
    


 