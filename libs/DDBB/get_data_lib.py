# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:04:26 2016

@author: montoya
"""

import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt
import copy as copy
import time as time
import utilities_lib as ul

from pandas_datareader import wb
import pandas_datareader as pdr



############################################################################
########################### YAHOO Finance ##################################
############################################################################

def download_TD_yahoo(symbol = "AAPL", precision = "m", 
                   start_date = dt.datetime(2016,1,1), end_date = dt.datetime(2017,1,1)):

    # data1 = dt.datetime.fromtimestamp(1284101485)
    sdate = start_date
    edate = end_date
    
#    sdate_ts = int(get_timeStamp(sdate))
#    edae_ts = int(get_timeStamp(edate))

#    url_root = "https://finance.yahoo.com/quote/"
#    url_root += symbol
#    url_root += "/history?"
#    url_root += "period1=" + str(sdate_ts)
#    url_root += "&period2=" + str(edate_ts)
#    url_root += "&interval=" + precision
#    url_root += "&filter=history&frequency=" + precision
 
    url_root = "http://chart.finance.yahoo.com/table.csv?"
    url_root += "s=" + symbol
    url_root += "&a=" +str(sdate.day)+ "&b=" +str(sdate.month)+ "&c=" +str(sdate.year)
    url_root += "&d=" +str(edate.day)+ "&e=" +str(edate.month)+"&f="+str(edate.year)
    url_root += "&g=" + precision
    url_root += "&ignore=.csv"
    
#    print url_root
    response = urllib2.urlopen(url_root)
    data = response.read().split('\n')
    nlines = len(data)
    for i in range(nlines):
        data[i] = data[i].split(",")
        
#    print data[0:4]
    df = pd.DataFrame(data)
    df.columns = df.ix[0]  #['Date','Open', 'High', 'Low', 'Close', 'Volume', "Adj Close"]
#    print df.columns
 
    
    ### REMOVE FIRST ROW (Headers) 
    df.drop(0, inplace = True)
    ### REMOVE LAST ROW (Nones)
#    print len(df) - 1
#    print df.ix[len(df) - 1]
    df.drop(df.index.values[len(df) - 1], inplace = True)
    ### CONEVERT DATES TO TIMESTAMPS (Nones)
#    print df.Date
    df.index = ul.str_to_datetime(df.Date)
    
    del df['Date']
    ## 
    # We have to
    return df
    
## There used to be 2 of them
def download_TD_yahoo2(symbol = "AAPL", precision = "1mo", 
                   start_date = "01-12-2011", end_date = "01-12-2015"):

    # data1 = dt.datetime.fromtimestamp(1284101485)
    sdate = dt.datetime.strptime(start_date, "%d-%m-%Y")
    edate = dt.datetime.strptime(end_date, "%d-%m-%Y")
    
#    sdate_ts = int(get_timeStamp(sdate))
#    edae_ts = int(get_timeStamp(edate))

#    url_root = "https://finance.yahoo.com/quote/"
#    url_root += symbol
#    url_root += "/history?"
#    url_root += "period1=" + str(sdate_ts)
#    url_root += "&period2=" + str(edate_ts)
#    url_root += "&interval=" + precision
#    url_root += "&filter=history&frequency=" + precision
 
    url_root = "http://chart.finance.yahoo.com/table.csv?"
    url_root += "s=" + symbol
    url_root += "&a=" +str(sdate.day)+ "&b=" +str(sdate.month)+ "&c=" +str(sdate.year)
    url_root += "&d=" +str(edate.day)+ "&e=" +str(edate.month)+"&f="+str(edate.year)
    url_root += "&g=" + "m"
    url_root += "&ignore=.csv"
    
#    print url_root
    response = urllib2.urlopen(url_root)
    data = response.read().split('\n')
    nlines = len(data)
    for i in range(nlines):
        data[i] = data[i].split(",")
        
    print (data[0:4])
    df = pd.DataFrame(data)
    df.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume', "Adj Close"]
    
    print (df.columns)
    df.index = df.Date
#    del df['Date']
    return df
    
def download_D1_TD_yahoo_prev(symbol,start_date, end_date ):
    df = web.DataReader(symbol, 'yahoo', start_date, end_date)
#    print df.shape
    return df

def download_D1_TD_yahoo(symbol,start_date, end_date ):
    df = pdr.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)
#    print df.shape
    return df

############################################################################
########################### GOOGLE Finance ##################################
############################################################################
    
def download_D1_TD_google(symbol,start_date, end_date ):
    df = web.DataReader(symbol, 'google', start_date, end_date)
#    print df.shape
    return df
    
def download_TD_google(symbolID, period, timeInterval):
    """This function downloads data from Google and converts it into TD
     Google only stores the intraday data of the last few days or something
     The parameters to indicate are:
         - symbolID: The name of the symbol in the google part
         - period: The period in minutes of the timeframe we want.
           1, 5, 15, 30, 1440... 
         - timeInterval: Period back that we want to download. 
           We cannot download all we want since Google will not give it to us.
           Examples: Last 12 days: 12d. Last 3 years: 3Y
    """
    
#    print symbolID, period, timeInterval
    period_seconds = period*60
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period_seconds) + '&p=' + timeInterval
    url_root += '&f=d,o,h,l,c,v&df=cpct&q=' + symbolID
    
#    print url_root
    try:
        response = urllib2.urlopen(url_root)

    except urllib2.HTTPError as err:
        print ("Error downloading from google ",  symbolID, "Period: " + str(period))
        print ("Error code ", err.code)
        print ("Meaning of 503: HTTP Error 503: Service Unavailable")
        return ul.empty_df
    data = response.read().split('\n')
    #actual data starts at index = 7
    #first line contains full timestamp,
    #every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
            #first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period_seconds)
                parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
            except:

                pass # for time zone offsets thrown into data
    df = pd.DataFrame(parsed_data)
    if (df.shape[1] == 0):
        print ("We could not download data for ", symbolID, "Period: " + str(period))
        return ul.empty_df
#    print df
    df.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']
    df.index = df.Date
    del df['Date']
    TD = df
    print ("Downloaded: " + str(TD.shape[0]) + " samples", symbolID, period)
    return TD



def get_spread(base, hedge, ratio, period, window):
    b = get_google_data(base, period, window)
    h = get_google_data(hedge, period, window)
    combo = pd.merge(pd.DataFrame(b.c), pd.DataFrame(h.c), left_index = True, right_index = True, how = 'outer')
    combo = combo.fillna(method = 'ffill')
    combo['spread'] = combo.ix[:,0] + ratio * combo.ix[:,1]
    return(combo)
