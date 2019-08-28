# -*- coding: utf-8 -*-


import pandas as pd
import utilities_lib as ul
#### IMPORT the methods
import TimeData_core as TDc    # Core methods
import TimeData_graph as TDgr   # Graphics
import TimeData_DDBB as TDDB   #
import TimeData_indicators as TDind
import TiemData_pandasIndicators as TDindp

""" A container to store prices for a symbol:
This Class will contain for a given symbol:
  - Daily Data.
  - Intraday Data.
  
Functions to:
- Load data into it from any specific source 
- Store itself into Disk to be loaded afterwards
- """

""" Dayly data will be a pandas Dataframe with the structure:

              Open    High     Low   Close    Volume
Date                                               
2015-02-03  121.74  121.76  120.56  121.05   8255863
2015-02-04  121.63  122.22  120.92  121.58   5386747
2015-02-05  120.98  121.83  120.61  121.79   6879945
2015-02-06  119.15  119.52  117.95  118.64  13206906

Where Date is the index and is in dt.datetime.
A set of functions for dealing with it will be specified. 
Set of functions such as add values, delete values will be done.

"""

class CTimeData:
    
    def __init__(self, symbolID: str = None, period: int = None,  TD: pd.DataFrame = ul.empty_df):
        self.symbolID = symbolID    # Symbol of the Security (GLD, AAPL, IDX...)
        self.period = period        # It is the number of minutes of the period: 1 5 15....
        
        ## Time constraining variables
        self.start_time = None  # Start and end of period to operate from the TD
        self.end_time = None
        self.time_mask = None
        
        # There is a set of flags for precomputed shit that does not have to be precomputed again
        self.set_TD(TD)   # To make the interval
        # If timeData is empty initially then no interval will be set
        
        # Primary variables
        self.timeSeries = None;
        self.dates = None
        self.set_seriesNames()   # Names of the signals we are working with ("Open", "Low", "Average", "Volume")
                                 # If indicated with a new function, then they change
        
        self.trimmed = False
        #####################################################
        ##### Variables necesary for interactivity ####
        ##############################################
        
    #######################################################################
    ############## DDBB methods ###########################################
    #######################################################################
    set_csv = TDDB.set_csv    # Set and add timeData from csv's
    add_csv = TDDB.add_csv
    save_to_csv = TDDB.save_to_csv # Save timeData to csv
    update_csv = TDDB.update_csv
    
    set_TD_from_google = TDDB.set_TD_from_google
    
    set_TD_from_yahoo = TDDB.set_TD_from_yahoo
    update_csv_yahoo = TDDB.update_csv_yahoo
    
    
    # Intern functions
    set_TD = TDDB.set_TD
    get_TD = TDDB.get_TD
    add_TD = TDDB.add_TD
    trim_TD = TDDB.trim_TD
    # Other functions
    get_intra_by_days = TDDB.get_intra_by_days
    fill_data = TDDB.fill_data
    #######################################################################
    ############## CORE Methods ###########################################
    #######################################################################
    
    # Basic sets and gets
    set_period = TDc.set_period
    set_interval = TDc.set_interval
    set_seriesNames = TDc.set_seriesNames
    cmp_seriesNames = TDc.cmp_seriesNames
    cmp_indexes = TDc.cmp_indexes
    set_inner_timeSeries = TDc.set_inner_timeSeries
    
    # Get properties of the class
    get_period = TDc.get_period
    get_seriesNames = TDc.get_seriesNames
    get_dates = TDc.get_dates
    get_indexDictByDay =  TDc.get_indexDictByDay
    get_timeSeriesbyName = TDc.get_timeSeriesbyName
    get_final_SymbolID_period = TDc.get_final_SymbolID_period
    # Get data timeSeries
    get_timeSeries = TDc.get_timeSeries
    get_timeSeriesReturn = TDc.get_timeSeriesReturn
    get_timeSeriesCumReturn = TDc.get_timeSeriesCumReturn
    
    get_SortinoR = TDc.get_SortinoR
    get_SharpR = TDc.get_SharpR
    
    ## Other gets
    get_magicDelta = TDc.get_magicDelta
    get_diffPrevCloseCurrMin = TDc.get_diffPrevCloseCurrMin
    get_diffPrevCloseCurrMax = TDc.get_diffPrevCloseCurrMax
    
    ## guessing functions
    guess_period = TDc.guess_period
    guess_openMarketTime = TDc.guess_openMarketTime
    
    #######################################################################
    ############## Indicators  ###########################################
    #######################################################################
   ############## Moving Averages  ###########################################
    get_SMA = TDind.get_SMA
    get_WMA = TDind.get_WMA
    get_EMA = TDind.get_EMA
    get_TrCrMr = TDind.get_TrCrMr
    get_HMA = TDind.get_HMA
    get_HMAg = TDind.get_HMAg
    get_TMA = TDind.get_TMA
    ############## Ocillators  ###########################################
    get_MACD = TDind.get_MACD
    get_momentum = TDind.get_momentum
    get_RSI = TDind.get_RSI
    ############## Volatility  ###########################################
    get_BollingerBand = TDind.get_BollingerBand
    get_ATR = TDind.get_ATR

    #######################################################################
    ############## Indicators from pandas  ###########################################
    #######################################################################

    # Moving Averages
    SMA = TDindp.SMA
    EMA = TDindp.EMA
    # Volatility
    STD = TDindp.STD
    AHLR = TDindp.AHLR
    ATR = TDindp.ATR
    Chaikin_vol = TDindp.Chaikin_vol
    # Price Channels
    PPSR  = TDindp.PPSR
    FibboSR = TDindp.FibboSR
    BBANDS = TDindp.BBANDS
    PSAR = TDindp.PSAR
    # Basic momentums
    MOM  = TDindp.MOM
    ROC = TDindp.ROC
    # Oscillators
    STO = TDindp.STO
    STOK = TDindp.STOK
    STOK = TDindp.STOK
    
    RSI = TDindp.RSI
    MACD = TDindp.MACD
    TRIX = TDindp.TRIX
    
    ADX = TDindp.ADX
    # Volume indicators
    ACCDIST = TDindp.ACCDIST

    #######################################################################
    ############## Graphics  ###########################################
    #######################################################################

    plot_timeSeries = TDgr.plot_timeSeries
    plot_timeSeriesReturn = TDgr.plot_timeSeriesReturn
    plot_timeSeriesCumReturn = TDgr.plot_timeSeriesCumReturn

    scatter_deltaDailyMagic = TDgr.scatter_deltaDailyMagic
    plot_TrCrMr = TDgr.plot_TrCrMr
    
    plot_BollingerBands = TDgr.plot_BollingerBands
#pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
#       keys=None, levels=None, names=None, verify_integrity=False)
       