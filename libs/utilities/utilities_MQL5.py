#########################################################3
###### General utilities LIBRARY related to MQL5 #########
##########################################################
## Library with function to convert data.
## Initially from .hst to .csv

import pandas as pd

#########################################################
#################### General Data Structures ###############
#########################################################

w = 10  # Width of the images
h = 6   # Height of the images

# Define the empty dataframe structure
keys = ['Open', 'High', 'Low', 'Close', 'Volume']
empty_df= pd.DataFrame(None,columns = keys )

keys_col = ['Symbol','Type','Size','TimeOpen','PriceOpen', 'Comision','CurrentPrice','Profit']
empty_coliseum = pd.DataFrame(None,columns = keys_col )

# Dictionary between period names and value
periods = [1,5,15,30,60,240,1440,10080,43200, 43200*12]
periods_names = ["M1","M5","M15","M30","H1","H4","D1","W1","W4","Y1"]
period_dic = dict(zip(periods,periods_names))
names_dic = dict(zip(periods_names, periods))


def get_foldersData(source = "FxPro", rrf = "../" ):
    # Returns the folders where we can find the previously stored data,
    # new data to download and the info about the symbols we have or 
    # want to download.
#    rrf = "../" # relative_root_folder

    if (source == "Hanseatic"):
        storage_folder = rrf + "./storage/Hanseatic/"
        updates_folder = rrf +"../Hanseatic/MQL4/Files/"
            
    elif (source == "MQL5"):
        storage_folder = rrf + "./storage/MQL5/"
        updates_folder = rrf +"./MT5/MQL5/MQL5/Files/"
            
    elif (source == "Yahoo"):
        storage_folder = rrf +"./storage/Yahoo/"
        updates_folder = rrf +"internet"
        
    elif (source == "Google"):
        storage_folder = rrf +"./storage/Google/"
        updates_folder = rrf +"internet"

    else:
        print ("Not recognized")
    return storage_folder, updates_folder
