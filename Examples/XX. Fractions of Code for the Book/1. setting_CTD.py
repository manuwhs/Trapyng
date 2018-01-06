# Change main directory to the main folder and import folders
import os
os.chdir("../")
import import_folders

import CTimeData as CTD
import utilities_lib as ul

######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo Google
[storage_folder, info_folder, updates_folder] = \
    ul.get_foldersData(source = dataSource)
    
symbol = "Amazon"
period = 1440

######## CREATE THE OBJECT AND LOAD THE DATA ##########
timeData = CTD.CTimeData(symbol,period) # Set main properties
timeData.set_csv(storage_folder)        # Load the data into the model


import datetime as dt

sdate = dt.datetime.strptime("01-01-2016", "%d-%m-%Y")
edate = dt.datetime.strptime("15-05-2016", "%d-%m-%Y")

timeData.set_interval(sdate,edate) # Set the interval period to be analysed
mask = timeData.time_mask
dates = timeData.get_dates()