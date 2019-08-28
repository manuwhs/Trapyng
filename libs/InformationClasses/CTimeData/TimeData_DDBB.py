# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import utilities_lib as ul
import get_data_lib as gdl 
import Intraday_lib as intl
import datetime as dt
import DDBB_lib as DBl  # For getting data

""" LIBRARY FOR OPERATION RELATED TO THE BBDD of the marcket """
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

Where Date is the index and is in numpy.datetime64.
A set of functions for dealing with it will be specified. 
Set of functions such as add values, delete values will be done.

"""


################# TableData FUNCTIONS ##########################

def set_TD (self, TD): # Stablish the initial data
    self.TD = TD;
    self.TD.sort_index(inplace=True)
    self.set_interval(trim = False)  # Set the interval to the maximum possible
    
def get_TD (self, indexes = [], subselect = True):
    if (subselect == False):
        return self.TD
    else:
        if (len(indexes) == 0):
            indexes = self.time_mask
        # The problem is that now the indexes of the new TD will not mach with the previous
        op1 = self.TD.iloc[indexes]
#        op2 = self.TD.ix[indexes]
    return op1

def add_TD(self, new_TD):
    # This function adds new data to the existing Daily data 
    # It places it into the positions refered by the "Index" date.
    # If there are days with the same index, they get overwritten.
    # new_dailyData is expected to have nice format
#        self.dailyData = pd.concat([self.dailyData, new_dailyData], verify_integrity = True)
#        self.dailyData = pd.merge(self.dailyData, new_dailyData)
    # Combine both pandas overwitting the old with the new
    self.TD = new_TD.combine_first(self.TD) 
    self.TD.sort_index(ascending = False)
    self.set_TD(self.TD)   # To make the interval

def trim_TD(self, time_mask):
    # This function receives the list of indexes of the new TD and erases the rest
    self.trimmed = True   # flag that it was trimmed for when we save it
    TD = self.TD.iloc[time_mask]
    self.set_TD(TD)
    
# This function preprocess the RAW table of data into a TD.
# Which is basically processing the "Date" part. It is inline
# TODO: Maybe some processing on the "Date part"
def preprocess_RAW_TD(self, Raw_TD):
    processed_dates = pd.to_datetime(Raw_TD.index)
    Raw_TD.index = processed_dates
    return Raw_TD
    
################# CSV FUNCTIONS ##########################
def save_to_csv(self, file_dir = "./storage/", force = False):
    # This function saves the TD to a csv file
    if (self.trimmed == True and force == False):
        print ("You cannot save the file since you trimmed it, Use force = True")
    else:
        ul.create_folder_if_needed(file_dir)
        whole_path = file_dir + ul.period_dic[self.period] + "/" + \
            self.symbolID + "_" + ul.period_dic[self.period] + ".csv"
        ul.create_folder_if_needed(file_dir + ul.period_dic[self.period] + "/")
        self.TD.to_csv(whole_path, sep=',')

def set_csv(self, file_dir = "./storage/", symbolID = None, period = None, file_name = None):
    

    # This function loads the data from the file  file_dir + file_name if file_name is provided
    # Otherwise it uses the naming convention to find it from the root folder:
    # The file must have the path:  ./TimeScale/symbolName_TimeScale.csv
    
    # If we did not specify the symbolID or period and we set them in the 
    # initialization it will get them from there
    # specific and adds its values to the main structure
    symbolID, period = self.get_final_SymbolID_period(symbolID, period)
#    print("Setting csv: ", file_dir, ",symbol: ", symbolID,", period: ", period, ".File_name:",file_name)
    
    TD = DBl.load_TD_from_csv(file_dir,symbolID,period,file_name)
    self.set_TD(TD)
    
def add_csv(self, file_dir = "./storage/", symbolID = None, period = None):
    print("Setting csv: ", file_dir, ",symbol: ", symbolID,", period: ", period)
    # Loads a CSV and adds its values to the main structure
    symbolID, period = self.get_final_SymbolID_period(symbolID, period)
    newTD = DBl.load_TD_from_csv(file_dir,symbolID,period)
    self.add_TD(newTD)

def update_csv (self,storage_folder, updates_folder,
                 symbolID = None, period = None):
    # Function that loads from 2 different folders, joins the data and saves it back
    self.set_csv(storage_folder,symbolID,period)
    self.add_csv(updates_folder,symbolID,period)
    self.save_to_csv(storage_folder)

    
########### WEBSOURCES ################################
########### Google #######
def set_TD_from_google(self, symbolID = None, period = None, timeInterval = "30d"):
    symbolID, period = self.get_final_SymbolID_period(symbolID, period)
    TD = gdl.download_TD_google(symbolID, period, timeInterval)
    self.set_TD(TD)
    return TD
    
# Update the current data using yahoo
def set_TD_from_yahoo(self,start_date = "01-12-2011", end_date = "01-12-2015", precision = "d" ):
    TD = gdl.download_TD_yahoo(self.symbolID, precision, 
                   start_date , end_date)
    self.set_TD(TD)
    return TD
    
def update_csv_yahoo (self,sdate,edate,file_dir_current = "./storage/"):
    self.download_from_yahoo(sdate,edate)
    self.add_csv(file_dir_current)  ## Add it to the one we already have
    self.save_to_csv(file_dir_current)

######################################################################
############## BBDD data processing ###################################
#######################################################################

def fill_data(self):
    data_TD = self.get_TD()
    ninit = data_TD.shape[0]
    data_TD = intl.fill_by_filling_everything(data_TD, self.start_time, self.end_time)
    nend = data_TD.shape[0]
    self.set_TD(data_TD)
    if (nend > ninit):
        msg = "Missing : %i / %i" % (nend - ninit, nend)
        print (msg)
        
def get_intra_by_days(self):
    result = intl.get_intra_by_days(self.dates, self.get_timeSeries())
    return result
    
def check_data(self):  # TODO
    # Check that there are no blanck data or wrong
    print ("checking")
    
def data_filler(self): # In case we lack some data values
    # We can just fill them with interpolation
    # We do this at csv level 

    self.dailyData 
    start_date = self.dailyData.index[0].strftime("%Y-%m-%d")   #strptime("%Y-%m-%d")
    end_date = dt.datetime(self.dailyData.index[-1].strftime("%Y-%m-%d"))
    
    print (start_date)
    # We have to get all working days between these 2 dates and check if
    # they exist, if they dont, we fill them
    # Create days of bussines
    
    busday_list = []
    
    next_busday = np.busday_offset(start_date, 1, roll='forward')
    busday_list.append(next_busday)
    
    while (next_busday < end_date): # While we havent finished
#        print next_busday, end_date
        next_busday = np.busday_offset(next_busday, 1, roll='forward')
        busday_list.append(next_busday)

    Ndays_list = len(busday_list)   # Number of days that there should be
    Ndays_DDBB = len(self.dailyData.index.tolist())
    
    print (Ndays_list, Ndays_DDBB)  ## TODO
    
    
#    for i in range (Ndays_list):
#        print "e"
        
    print (start_date, end_date)


def fill_in_missing_dates(df, date_col_name = 'date',date_order = 'asc', fill_value = 0, days_back = 30):

    df.set_index(date_col_name,drop=True,inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    d = datetime.now().date()
    d2 = d - timedelta(days = days_back)
    idx = pd.date_range(d2, d, freq = "D")
    df = df.reindex(idx,fill_value=fill_value)
    df[date_col_name] = pd.DatetimeIndex(df.index)

    return df


def data_filler_main_TD():
    time_diff = intl.find_min_timediff(self.get_timeData())
#    print time_diff
#    print type(time_diff)
    
#    idx = intl.get_pdintra_by_days(timeData.TD)
    
    ## Fill the interspaces, create another timeSeries and plot it
    filled_all = intl.fill_everything(self.get_timeData())
    
    timeData2 = copy.deepcopy(timeData)
    timeData2.set_timeData(filled_all)
    timeData2.get_timeSeries(["Close"])
    timeData2.plot_timeSeries()
    print (timeData2.get_timeSeries().shape)
    
    ## Fill missing values by first filling everythin
    filled = intl.fill_by_filling_everything(self.get_timeData())
    timeData2 = copy.deepcopy(timeData)
    timeData2.set_timeData(filled)
    timeData2.get_timeSeries(["Close"])
    timeData2.plot_timeSeries(nf = 0)
    print (timeData2.get_timeSeries().shape)
    
    ### Get the day table
    pd_dayly = intl.get_dayCompleteTable(timeData.get_timeData())
    time_index = intl.find_trade_time_index(timeData.get_timeData())
    index_shit = intl.find_interval_date_index(timeData.get_timeData(), dt.date(2016,3,1),  dt.date(2016,5,1))



## OBSOLETE
#######################################################################
############## Add from the Internet ###################################
#######################################################################
 
def addDaily_from_google (self,start_date, end_date):
    data_daily_google = gdl.get_dayly_google(self.symbol,start_date, end_date )
    
    self.add_DailyData(data_daily_google)

def addDaily_from_yahoo (self,start_date, end_date):
    data_daily_google = gdl.get_dayly_yahoo(self.symbol,start_date, end_date )
    
    self.add_DailyData(data_daily_google)
    
def addIntra_from_google (self,days_back):
    # Days_back is the number of days back we get the data.
    # It cannot exeed 14 for google
    data_intra_google = gdl.get_intra_google(self.symbol, self.period * 60, days_back)
    
    self.add_IntraData(data_intra_google)

def addIntra_from_yahoo (self,days_back):
    # Days_back is the number of days back we get the data.
    # It cannot exeed 14 for google
    data_intra_google = gdl.get_intra_yahoo(self.symbol, self.period * 60, days_back)
    
    self.add_IntraData(data_intra_google)