import Portfolio_DDBB as Pdb
import Portfolio_core as CPc
import Portfolio_operations as CPop
import Portfolio_interface as CPin

import Portfolio_indicators as Poind

class Portfolio:
    def __init__(self, portfolioID = None, symbolIDs = [], periods = [], symbols = []):
        portfolioID = portfolioID
        self.symbols = dict()
        # Loop over the symbol_names so loop over all the symbols in the Portfolio
        self.init_symbols(symbolIDs, periods,symbols)  # Create the symbol objects from the periods and names
            
    # secutities will be a dictionary of [symbol]

    init_symbols = CPc.init_symbols
    
    ### Interface for symbol objects
    set_symbols = CPc.set_symbols
    get_symbols = CPc.get_symbols
    add_symbols = CPc.add_symbols
    del_symbols = CPc.del_symbols

    ### Interface for timeDataObjects
    get_timeData = CPc.get_timeData
    get_dates = CPc.get_dates
    get_timeSeries = CPc.get_timeSeries
    get_timeSeriesReturn = CPc.get_timeSeriesReturn
    get_timeSeriesCumReturn = CPc.get_timeSeriesCumReturn
    
    ### Apply functions to all of them
    set_interval = CPc.set_interval
    set_seriesNames = CPc.set_seriesNames
    
    default_select = CPc.default_select

    get_symbolIDs = CPc.get_symbolIDs
    
    #######################################################################
    #### DDBB Operations ##################################################
    #######################################################################
           
    set_csv = Pdb.set_csv
    add_csv = Pdb.add_csv
    save_to_csv = Pdb.save_to_csv
    update_symbols_csv = Pdb.update_symbols_csv
    
    set_symbols_from_google = Pdb.set_symbols_from_google
    
    update_symbols_csv_yahoo = Pdb.update_symbols_csv_yahoo
    download_symbols_csv_yahoo = Pdb.download_symbols_csv_yahoo
    load_symbols_info = Pdb.load_symbols_info
    
    fill_data = Pdb.fill_data
    #######################################################################
    #### Operations over all the prices of the portfolio ##################
    #######################################################################

    get_daily_symbolsPrice = CPop.get_daily_symbolsPrice
    plot_daily_symbolsPrice = CPop.plot_daily_symbolsPrice
    get_daily_symbolsCumReturn = CPop.get_daily_symbolsCumReturn
    
    get_intra_by_days = CPin.get_intra_by_days


    #######################################################################
    ############## Indicators from pandas  ###########################################
    #######################################################################

    SMA = Poind.SMA
    EMA = Poind.EMA
    PPSR  = Poind.PPSR
    BBANDS = Poind.BBANDS
    ATR = Poind.ATR
    MOM  = Poind.MOM
    ROC = Poind.ROC
    STO = Poind.STO
    RSI = Poind.RSI
    
    ADX = Poind.ADX
    ACCDIST = Poind.ACCDIST
    MACD = Poind.MACD
    TRIX = Poind.TRIX
    