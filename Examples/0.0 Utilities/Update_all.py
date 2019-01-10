import os
os.chdir("../../")
import import_folders

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import CSymbol as CSy
import copy as copy
import CPortfolio as CPfl
import gc
import utilities_lib as ul

import DDBB_lib as DBl
### CODE TO EXECUTE FROM TIME TO TIME TO DOWNLOAD A LOT OF DATA FROM MT4 OR
### THE INTERNET

######## SELECT SOURCE ########
dataSource =  "Google"  # Hanseatic  FxPro GCI Yahoo Google

[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource, symbol_info_list = "Current")

## Select the Symbol Names and periods we want to download

periods = [15]
#periods = [60,1440]
#periods = [1,15]

### UPDATE ALL SYMBOLS

if ((dataSource != "Google") and (dataSource != "Yahoo")):
    # Update from the CSVs downloaded already
    Symbol_info = CSy.load_symbols_info(info_folder)
    Symbol_names = Symbol_info["Symbol_name"].tolist()
    for period in periods:
        Cartera = CPfl.Portfolio("BEST_PF", Symbol_names, [period]) 
        Cartera.update_symbols_csv(storage_folder, updates_folder )
        print "Done with period " + str(period)    
        Cartera = 0;
        gc.collect()
else:
    if (dataSource == "Google"):
    # Download the Data from the Web source and merge it with the already stored
        cmplist = DBl.read_NASDAQ_companies(whole_path = "../storage/Google/companylist.csv")
        cmplist.sort_values(by = ['MarketCap'], ascending=[0],inplace = True)
        symbolIDs = cmplist.iloc[0:500]["Symbol"].tolist()
        
        # Load them by the Porfolio.
        # Bad thing is that it does not save the stuff until it loads all symbols
        if(0):
            for period in periods:
                Cartera = CPfl.Portfolio("BEST_PF", symbolIDs, [period]) 
                Cartera.set_symbols_from_google(timeInterval = "10Y" )
                Cartera.add_csv(storage_folder)
                Cartera.save_to_csv(storage_folder)
        #        print "Done with period " + str(period)    
        #        print Cartera.get_symbolIDs()

        # Load them by Symbol !
        for symbolID in symbolIDs:
            mySymbol = CSy.CSymbol(symbolID, periods)
            mySymbol.set_TDs_from_google(timeInterval = "1M" )
            print mySymbol.get_timeData(15).TD.index.size
            print mySymbol.get_timeData(15).TD.index
            mySymbol.add_csv(storage_folder)
            mySymbol.save_to_csv(storage_folder)
            print "Done with symbol " + str(symbolID)    
            break
        
    #        print Cartera.get_symbolIDs()


