import numpy as np
import pandas as pd
import copy
import datetime as dt
import CTimeData as CTD
import basicMathlib as bMl

def preprocess_data(timeData, sdate = None, edate = None, remove_nontrading_days = True, 
                    adjust_session_start = True, remove_small_trading_days = True):
    # This function preprocesses the Raw data given in the Toptal screening project
    if (remove_nontrading_days):
        ################# Remove all non-trading samples #######################
        timeData.TD = timeData.TD[timeData.TD["Volume"] > 0]
        timeData.TD.reset_index()
        # Reset the index values of the table and call set_interval to reset the internal mask
        timeData.set_interval(sdate,edate) 
    
    ## Readjust time so that the session starts at 00:00 and not at 22:00 of the
    # previous natural day 
    if (adjust_session_start):
        ## Get the begginings of the starting sessions !!
        week_start_times = np.where(timeData.TD.index[1:]-timeData.TD.index[:-1] > \
                                    dt.timedelta(minutes = 15)) [0] 
        week_start_times += 1  # We shift all the postiions one to get the start of the week
        
        ################################################
        ######  Readjust time by aligning each init of trading session to 00:00:00
        ######################################################
        index_copy = copy.deepcopy(timeData.TD.index)
        timeData.TD['Original Time'] = pd.Series(index_copy, index=timeData.TD.index)
        
        ## The first week we assume the starting session is at 22:00
        start_hour = 22
        timeData.TD.index.values[:week_start_times[0]] =  timeData.TD.index[:week_start_times[0]] + \
        dt.timedelta(hours =   24 - start_hour)
        
        for i in range(week_start_times.size -1):
            start_hour = timeData.TD.index[week_start_times[i]].hour
            if (start_hour > 0):
                # We cannot change the index, but we can change the values
                timeData.TD.index.values[week_start_times[i]:week_start_times[i+1]] =  \
                timeData.TD.index[week_start_times[i]:week_start_times[i+1]] +  \
                dt.timedelta(hours =   24 - start_hour)
        ## The last week:
        start_hour = timeData.TD.index[week_start_times[-1]].hour
        if (start_hour > 0):
            # We cannot change the index, but we can change the values
            timeData.TD.index.values[week_start_times[-1]:] =  timeData.TD.index[week_start_times[-1]:] +  \
            dt.timedelta(hours =   24 - start_hour)
        
        ## Rest interval after we changed the index
        timeData.set_interval(sdate,edate) 
        
    ############ Delete Irregular days #############
    if (remove_small_trading_days):
        days_keys, day_dict = timeData.get_indexDictByDay()
        print ("Number of initial days: %i"%(len(days_keys)))
        for day_i in range(len(days_keys)):  
            # We go from end to begining
            day_index = len(days_keys) -1 - day_i
            day = days_keys[day_index]
            if (len(day_dict[day]) !=96):
                timeData.TD.drop(timeData.TD.index[day_dict[days_keys[day_index]]], inplace=True)
    #            print ("timeData shape: ", timeData.TD.shape)
                
        # Reset the index of the pd Datagrame and the timeData structure
        timeData.TD.reset_index()
        timeData.set_interval(sdate,edate) 
    
    return timeData


def get_daily_timedata(timeData, symbolID):
    
    # Transform the data from intraday to daily
    days_keys, day_dict = timeData.get_indexDictByDay()
    ## Get the daily HLOC
    H = np.array([np.max(timeData.TD["High"][day_dict[day_i]])  for day_i in days_keys])
    L = np.array([np.min(timeData.TD["Low"][day_dict[day_i]])  for day_i in days_keys])
    O = np.array([timeData.TD["Open"][day_dict[day_i][0]]  for day_i in days_keys])
    C = np.array([timeData.TD["Close"][day_dict[day_i][-1]]  for day_i in days_keys])
    V = np.array([np.sum(timeData.TD["Volume"][day_dict[day_i]])  for day_i in days_keys])
    Original_time = np.array([timeData.TD["Original Time"][day_dict[day_i][-1]]  for day_i in days_keys])
        
    daily_TD = pd.DataFrame(
    {'Time': days_keys,'Original Time': Original_time,'High': H,'Low': L,"Close":C,"Open":O,"Volume":V})
    daily_TD.set_index('Time',inplace = True)
    timeData_daily = CTD.CTimeData(symbolID,1440)
    timeData_daily.set_TD(daily_TD)

    return timeData_daily

def get_output_df(dates_predictions,Ypred):
    
    hard_decision = np.zeros(Ypred.shape)
    hard_decision[np.where(Ypred > 0.5)] = 1
    output_df = pd.DataFrame(
    {'Time':  dates_predictions, "Probability" : Ypred, "Class": hard_decision})
    output_df.set_index('Time',inplace = True)

    return output_df


def add_lagged_values(df, feature_vector, name_feature, Nlags = 1):
    # Function that adds lagged values to the dataframe
    for i in range(Nlags):
        signal = bMl.shift(feature_vector,lag = i+1).flatten()
        # print (type(signal))
        df[name_feature + "_%i"%(i+1)] =signal
