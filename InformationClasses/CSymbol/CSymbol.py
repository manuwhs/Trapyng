# -*- coding: utf-8 -*-
import pandas as pd

#### IMPORT the methods
import Symbol_core as Syco    # Core methods
import Symbol_DDBB as SyDB   # Graphics
import Symbol_indicators as Syind
class CSymbol:
    
    def __init__(self, symbolID = None, periods = []):
        self.symbolID = symbolID    # Symbol of the Security (GLD, AAPL, IDX...)
        self.timeDatas = dict()    # Dictionary   TimeDatas[period] = CTimeData for that period
        
        # Loop over the periods to get all the TimeDatas
        
        ## TODO define the variables that state the properties of the symbol
        self.open_time = [];   # Time at which it is open
        self.type = "Share"
        self.country = "Spain"
        self.currency = "EUR"
        self.sector = "Energy" 
        self.info = []
        
        # Initialize TD dataframes to 0 and stablish their Symbol and period
        if (len(periods) > 0):
            self.init_timeDatas(symbolID, periods);  


    #######################################################################
    ############## DDBB methods ###########################################
    #######################################################################
    load_csv_timeData_period = SyDB.load_csv_timeData_period
    set_info = SyDB.set_info
    load_info = SyDB.load_info
    
    set_csv = SyDB.set_csv
    add_csv = SyDB.add_csv
    save_to_csv = SyDB.save_to_csv
    update_csv = SyDB.update_csv
   
    set_TDs_from_google = SyDB.set_TDs_from_google
    download_TDs_yahoo = SyDB.download_TDs_yahoo
    update_TDs_yahoo = SyDB.update_TDs_yahoo
    
 
    fill_data = SyDB.fill_data
    
    ################################################################
    ############################ CORE ############################
    ###############################################################
    
    init_timeDatas = Syco.init_timeDatas
    get_periods = Syco.get_periods
    
    get_timeData = Syco.get_timeData
    add_timeData = Syco.add_timeData
    del_timeData = Syco.del_timeData

    get_final_SymbolID_periods = Syco.get_final_SymbolID_periods
    ### Expanding  as well 
    set_interval =  Syco.set_interval
    set_seriesNames = Syco.set_seriesNames
    
    #######################################################################
    ############## Indicators from pandas  ###########################################
    #######################################################################

    SMA = Syind.SMA
    EMA = Syind.EMA
    PPSR  = Syind.PPSR
    BBANDS = Syind.BBANDS
    ATR = Syind.ATR
    MOM  = Syind.MOM
    ROC = Syind.ROC
    STO = Syind.STO
    RSI = Syind.RSI
    
    ADX = Syind.ADX
    ACCDIST = Syind.ACCDIST
    MACD = Syind.MACD
    TRIX = Syind.TRIX
    #########################################
    #########################################
    ### FUNCTIONS RELATED ###################
    
def load_symbols_info(file_dir = "./storage/"):
    # This functions loads the symbol info file, and gets the
    # information about this symbol and puts it into the structure
    whole_path = file_dir + "Symbol_info.csv"
    try:
        infoCSV = pd.read_csv(whole_path,
                              sep = ',')
    except IOError:
        error_msg = "Empty file: " + whole_path 
        print error_msg
        
    return infoCSV