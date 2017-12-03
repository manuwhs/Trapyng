#########################################################3
############### Intraday LIBRARY  ##############################
##########################################################
## Library to process the intraday day of a timeSeries

import numpy as np
import utilities_lib as ul
import pandas as pd
import datetime as dt
from collections import defaultdict

#### GET the time series divided in days #####
def get_intra_by_days(dates, timeSeries):

    days_list_price = [];
    days_list_dates = [];
    
#    timeSeries = self.get_timeSeries()
#    dates = self.dates

#        print type(dates[0])
    days_dates = ul.get_dates(dates)
    # We first get the different dates that we have
    uniq_days = np.unique(days_dates)
    
    # For each of these days
    for day_i in range (len(uniq_days)):
        # We get the relative samples indexes belonging to that day
        day_intra_indx = np.argwhere(days_dates == uniq_days[day_i]).T[0]
        # We get the samples from that
#        print day_intra_indx.shape
#        print timeSeries.shape
        
        day_intra_price = timeSeries[day_intra_indx,:]
        day_intra_date = dates[day_intra_indx]
        
        days_list_price.append(day_intra_price)
        days_list_dates.append(day_intra_date)
        
    return days_list_price, days_list_dates

def separate_TD_bydays(TD, seriesName = "Close"):
    # Returns a list of pandas dataframes, where every position is
    # the data por one day
    DFList = []
    # TD.groupby([TD.index.year,TD.index.month,TD.index.day]) 
    # TD.groupby(TD.index.day)
    # TD.groupby(TD.index.date)
    
    for group in TD.groupby(TD.index.date):
        # group[0] = TD.index.date
        # group[1] = pf with that index properties
        
        ## Transform its index to time
        TD_day = group[1].set_index(group[1].index.time)
        # Select only the close and put as name the date
        TD_day = TD_day[[seriesName]]
        TD_day = TD_day.rename(columns={seriesName: group[0].isoformat()})
        DFList.append(TD_day)
#        print group
    return DFList
#    DFList = [group[1] for group in df.groupby(df.index.day)]

def find_min_timediff(df):
    # This function finds the minimum time_dist of the index.
    index = df.index
    
    diffs = []
    for i in range(len(index) - 1):
        diffs.append(index[i +1] - index[i])
        
    min_diff = min(diffs)
    return min_diff

def find_trade_time_index(df):
    # This function finds the trading time for the symbol given its TD
    # Of course, in the period terms

    ## First approach !! Find the time hours in the dataset and get the unique ones !
    index = df.index
    data_times = ul.get_times(index)
    time_index = np.unique(data_times)
    ## It returns a numpy array with times, not a DataTimeIndex
    return time_index

def find_trade_days_index(df):
    # This function finds the trading time for the symbol given its TD
    # Of course, in the period terms

    ## First approach !! Find the time hours in the dataset and get the unique ones !
    index = df.index
    data_dates = ul.get_dates(index)
    dates_index = np.unique(data_dates)
    ## It returns a numpy array with times, not a DataTimeIndex
    return dates_index

def find_working_days(df):
    index = df.index
    index = index[np.argwhere(index.dayofweek < 5)]
#    print type(index)
#    print type(index[0])
#    print index.shape
    return index

def find_interval_date_index(TD, dstat, dend):
    ## It tries to find the timevalues that we should have.
    ## First it calculates the trading opening time.
    ## Then we add dates and extend it during the time specified in the parameters
    
    time_index = find_trade_time_index(TD)
    
    # Combine shit with shit
    time_index1 = pd.datetime.combine(dstat, time_index[0])

    ## TODO !! Improve this shit !! 
    
    print time_index1.isoformat()
    return time_index
    
def get_dayCompleteTable(TD, seriesName = "Close"):
    ## It gets the dayly table !! 
#          2016-01-05  2016-01-06  2016-01-07  2016-01-08  2016-01-11  \
#09:00:00       30.54       30.14       29.26       30.40       29.70   
#10:00:00       30.24       30.08       29.37       30.49       29.85   
#11:00:00       30.32       30.02       29.55       30.31       29.51   
#12:00:00       30.24       29.88       29.57       30.10       29.62   
#13:00:00       30.29       29.70       29.61       30.05       29.62   
#14:00:00       30.40       29.84       29.70       30.27       29.67   
#15:00:00       30.41       29.84       30.03       30.02       29.58   
#16:00:00       30.52       30.04       30.17       29.71       29.54   
#17:00:00       30.34       30.19       30.03       29.66       29.54   


    DFList = separate_TD_bydays(TD, seriesName)
    trade_index_time = find_trade_time_index(TD)

    pd_dayly = pd.DataFrame(index = trade_index_time)
    for DFday in DFList:
        pd_dayly = pd.concat([pd_dayly, DFday], axis=1)
    
    # type(pd_dayly.index) = <class 'pandas.core.index.Index'>
    # The type is not DataIndex anymore.
    return pd_dayly
    
def fill_everything(df, sdate = None, edate = None):
    ## So... this function takes the df, calculates the time_diff
    ## and if fucking fills the 24 between them, no matther if the stock
    ## was only fucking available on a given time.
    
    ## If not given
    index = df.index
    if (type(sdate) == type(None)):
        sdate = index[0]
    if (type(edate) == type(None)):
        edate = index[-1]
#    print sdate, edate
    
    min_diff = find_min_timediff(df)
#    print "min_diff: "  + str(min_diff)
    
    # We create index of the dayly timeframes that should be observed.
    freq = str(int(min_diff.total_seconds())) + "S"
    idx = pd.date_range(sdate.date(), pd.datetime.combine(edate.date(),dt.time(23,59,59)), freq = freq)
#    print "Lengths_diff: Before %i, After %i" %(len(index),len(idx))
    pd_dayly = pd.DataFrame(index = idx)  # index = idx
    pd_dayly.set_index(idx)

    # We join with the original
    pd_dayly = pd.concat([pd_dayly, df], axis=1)
    # Fill the NaN
    pd_dayly = pd_dayly.fillna(method = "ffill")
    ## If the first values where Nan, we use backfill as well
    pd_dayly = pd_dayly.fillna(method = "backfill")
 
    return pd_dayly


def fill_by_filling_everything(TD, sdate = None, edate = None):
    ## What we do is like... first fill everything, then obtain only the 
    ## the samples that are in the trade time
#    print sdate, edate
    allTD =  fill_everything(TD, sdate, edate)
#    print "Shape of the fully filled table: ", allTD.shape
#    
    ## Now we eliminate the trading days where there is no trade.
    ## We also eliminate the trading time-frame where no trade,
    # This only applies to the intraday data.
    # It can happen that one of the stocks has a day or several with no 
    # trading but the rest have, so what we do is filling that day.
    # WHAT WE WANT TO DO IS ELIMINATE THE DATES WHERE THESE IS NEVER TRADING !!
    
    # Find the trade timetable
    time_trade = find_trade_time_index(allTD)
    
    # Find the tradings days we have
    date_trade = find_trade_days_index(allTD)
    period = 100
    if (period <= 1440):
        date_trade = find_working_days(allTD)
    
#    print date_trade[0]
#    print time_trade
#    print (len(date_trade))
    indexes = allTD.index.time >= time_trade[0] 
    indexes2 = allTD.index.time <= time_trade[-1]
    
#    print len(date_trade)
#    print allTD.index[0].date()
#    print allTD.index[-1].date()
    ## Now we check if we had trading data for that day
    ## Maybe check that the price of the last samples is not the same
    ## Although that is not perfect.
    ## Or create a dictionary with the different dates
    
    ## TODO, learn really the different timeData types 
    dictionary_days = dict()
    for date in date_trade:
#        print date
#        print date.dtype
        # dtype('<M8[s]')
        if (period <= 1440):
            date = dt.datetime.utcfromtimestamp(date.astype(int)/1e9).date()
#        print date
        dictionary_days[date.isoformat()] = True
    dictionary_days = defaultdict(lambda: False, dictionary_days)
    
    indexes3 = [dictionary_days[x.isoformat()] for x in allTD.index.date]
#    print np.sum(indexes3)
#    indexes3 = indexes3[0]
#    print indexes3
#    print indexes
#    print indexes2
    indexes = indexes & indexes2 & indexes3
#    print "Final concatenation filling all days %i" % len(allTD)
 
    allTD = allTD.loc[indexes]
    
#    print "Length of the valid days %i" % len(allTD)
 
    return allTD
    
def get_pdintra_by_days(df):
    # This accepts a fucking pd dataframe !!
    # With its date index well put there.
    
    index = df.index
    sdate = index[0]  #.date
    print sdate 
    
    ##################################################
    ############### Obtain the dayly range of times 
    ######## that should exist #######################
    ################################################
    
    method = 2
    
    if (method == 1):
        ### TODO
        ### Basics of datetime !!
        date_only = sdate.date()       # Get only the date
        time_only = sdate.time()  # Get only the time
        str_date = sdate.isoformat()  # '2015-10-22T08:00:00'
        diff_dates = index[1] - index[0]  # TimeDelta structure
        
        ### We can also apply a lot of shit to the index  !!
        # Basically the same as to the individual but we do not use ()
        index.time
        index.date
        # Obtain it from the minimum time diff of the sequence
        ## Time difference
        min_diff = find_min_timediff(df)
        ## The time of a day
        time_day = dt.timedelta(days = 1)
        nperiods = int(time_day/min_diff)
        
        # We create index of the dayly timeframes that should be observed.
        freq = str(int(min_diff.total_seconds())) + "S"
        
        print nperiods 
        idx = pd.date_range(sdate, periods = nperiods,  freq = freq)

    elif(method == 2):
        # Obtaining it from the data itself, get them first for a few of them
        # and then obtain them
        
        data_times = ul.get_times(index)
        time_index = np.unique(data_times)
        
        pd_dayly = pd.DataFrame(index = time_index)

        ## Now we fucking have it nice, we just concatenate to this pandas
        ## the rest of the days as columns 

        pd_dayly = pd.concat([pd_dayly, BoW], axis=1)
        
        # Fill the NaN
        pd_dayly = pd_dayly.fillna(0)

        
    return idx

    
def separate_days (price, date):
    # This function separates the unidimensional price data into a list of days.
    # date has the day.minutes
    
    date = np.array(date,dtype = int)  # We only keep the indicator of the day
    date_days = date/1000000;  # Vector of days 
    hour_days = date%1000000;  # Vector of time of the days
    
    days = np.unique(date_days)   # Different days
    Ndays = days.size
    
    Price_list_days = []
    Hours_list_days = []
    
    for i in range (Ndays):  # Get the price y hora of every day
        index_day = np.where(date_days == [days[i]])
        
        prices_day = price[index_day]
        hour_day = hour_days[index_day]
        
        Price_list_days.append(prices_day)
        Hours_list_days.append(hour_day)
        
    return Price_list_days, Hours_list_days, days

def time_normalizer (hour_data, prices, time_span_sec):
    # This function normalizes the sample vector given and creates a list
    # where every element contains the mean price of the timespan. 
    # If there is no time for a Span, we give it the price of the previous

    # Hours = HHMMSS 
    """ We can use the histogram function"""
    # Transform the hours into seconds
    seconds_data = (hour_data/10000)*3600 + ((hour_data/100)%100)*60 + hour_data%100

    price_day = []
    
    total_s = 24*60*60
    N_bins = total_s  /time_span_sec
#    print seconds_data[:100]
    for i in range (N_bins):
        bin_prices_indx = np.where(seconds_data < (i+1)*time_span_sec)
        bin_prices = prices[bin_prices_indx]
        
        Npr = bin_prices_indx[0].size
        seconds_data = seconds_data[Npr:]  # Remove the previous 
        prices = prices[Npr:]  # Remove the previous 
                                                    # assuming prices are ordered.
#        print bin_prices
        
        if (bin_prices.size == 0): # If we have no data for the slice
            if (i == 0):  # If it is the first slice
                price_day.append(-1)
            else:
                price_day.append(price_day[-1])
        else:
            price_day.append(np.mean(bin_prices))
    
    return price_day

def get_close_open_diff (prices_days):
    # Compute the difference between openning and closing prices
    # prices_days[days][prices]
    shape = prices_days.shape
    
    diff = []
    
    for i in range (shape[0] - 1):
        diff.append(prices_days[i+1][0] - prices_days[i][-1])
    
    diff = np.array(diff)
    return diff
    
def get_open_close_diff (prices_days):
    # Compute the difference between openning and closing prices of the same day
    # prices_days[days][prices]
    shape = prices_days.shape
    
    diff = []
    
    for i in range (shape[0]):
        diff.append(prices_days[i][-1] - prices_days[i][0])
    
    diff = np.array(diff)
    return diff
    
#==============================================================================
#     hours = hour_data/10000
#     mins = (hour_data/100)%100
#     
#     Allhours = np.unique(hours)
#     Nhours = Allhours.size
#     
#     N_samples_by_hour = 60/time_span_min
#     
#     price_day = []
#     for h in range (Nhours):
#         prices_indx_hour = np.where(hours == Allhours[h])  # Subselect the hour index
#         princes_hour = prices[prices_indx]
#         
#         for m in range (N_samples_by_hour):  # Subselect the spans (histogram)
#             prices_indx =  
#             price_span = np.where(prices[prices_indx] < time_span_min*
#             price_day.append()
#==============================================================================
        
