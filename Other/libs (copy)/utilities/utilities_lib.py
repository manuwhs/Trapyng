#########################################################3
############### General utilities LIBRARY  ##############################
##########################################################
## Library with function to convert data.
## Initially from .hst to .csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import matplotlib.colors as ColCon
from scipy import spatial
import datetime as dt
import time
import magic
import shutil


#from graph_lib import gl

#########################################################
#################### General Data Structures ##########################
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

#########################################################
#################### Matrix Format Func ##########################
#########################################################

def fnp(ds):
    # This function takes some numpy element or list and transforms it
    # into a valid numpy array for us.
    # It works for lists arrays [1,2,3,5], lists matrix [[1,3],[2,5]]
    # Vectors will be column vectors
    # Working with lists
    
    # Convert tuple into list
    if (type(ds).__name__ == "tuple"):
        ds2 = []
        for i in range(len(ds)):
            ds2.append(ds[i])
        ds = ds2
        
    if (type(ds).__name__ == "list"):
        # If the type is a list 
        # If we are given an empty list 
        N_elements = len(ds)
        if (N_elements == 0):  # 
            ds = np.array(ds).reshape(1,0)
            return ds
            
        # We expect all the  elements to be vectors of some kind
        # and of the same length

        Size_element = np.array(ds[0]).size
        
            # If we have a number or a column vector or a row vector
        if ((Size_element == 1) or (Size_element == N_elements)):
            ds = np.array(ds)
    #            print ds.shape
            ds = ds.reshape(ds.size,1) # Return column vector
    
        # If we have an array of vectors
        elif(Size_element > 1):
            total_vector = []
    #            if (Size_element > N_elements):
                # We were given things in the from [vec1, vec2,...]
            for i in range(N_elements):
                vec = fnp(ds[i])
                total_vector.append(vec)
                
            axis = 1
            if (vec.shape[1] > 1):
                ds = np.array(ds)
                # If the vectors are matrixes 
                # We join them beautifully
            else:
                ds = np.concatenate(total_vector, axis = 1)
#                print "GETBE"
#                print total_vector[0].shape
#                if (Size_element > N_elements):
#                    ds = np.concatenate(total_vector, axis = 1)
#                else:
#                    ds = np.concatenate(total_vector, axis = 1).T
    # Working with nparrays
    elif (type(ds).__name__ == 'numpy.ndarray' or type(ds).__name__ == "ndarray"):

        if (len(ds.shape) == 1): # Not in matrix but in vector form 
            ds = ds.reshape(ds.size,1)
            
        elif(ds.shape[0] == 1):
            # If it is a row vector instead of a column vector.
            # We transforme it to a column vector
            ds = ds.reshape(ds.size,1)
            
    elif (type(ds).__name__ == 'DatetimeIndex'):
        ds = pd.to_datetime(ds)
        ds = np.array(ds).reshape(len(ds),1) 
    
    elif(type(ds).__name__ == 'Series'):
        ds = fnp(np.array(ds))
    
    elif (np.array(ds).size == 1):
        # If  we just receive a number
        ds = np.array(ds).reshape(1,1)
        
    return ds
    
def convert_to_matrix (lista, max_size = -1):
    # Converts a list of lists with different lengths into a matrix 
    # filling with -1s the empty spaces 

    Nlist = len(lista)
    
    listas_lengths = []
    
    if (max_size == -1):
        for i in range (Nlist):
            listas_lengths.append(lista[i].size)
        
        lmax = np.max(listas_lengths)
    else:
        lmax = max_size 
        
    matrix = -1 * np.ones((Nlist,lmax))
    
    for i in range (Nlist):
        if (lista[i].size > lmax):
            matrix[i,:lista[i].size] = lista[i][:lmax].flatten()
        else:
            matrix[i,:lista[i].size] = lista[i].flatten()
    
    return matrix

#########################################################
#################### General Data Structure ##########################
#########################################################

def windowSample (sequence, L):
    """ Transform a sequence of data into a Machine Learning algorithm,
    it transforms the sequence into X and Y being """
    
    sequence = np.array(sequence).flatten()
    Ns = sequence.size
    
    X = np.zeros((Ns - (L +1), L ))
    Y = np.zeros((Ns - (L +1),1) )
    for i in range (Ns - (L +1)):
        X[i,:] = sequence[i:i+L]
        Y[i] = sequence[i+L]
    # We cannot give the output of the first L - 1 sequences (incomplete input)
    return X, Y

def sort_and_get_order (x, reverse = True ):
    # Sorts x in increasing order and also returns the ordered index
    x = x.flatten()  # Just in case we are given a matrix vector.
    order = range(len(x))
    
    if (reverse == True):
        x = -x
        
    x_ordered, order = zip(*sorted(zip(x, order)))
    
    if (reverse == True):
        x_ordered = -np.array(x_ordered)
        
    return np.array(x_ordered), np.array(order)

def remove_list_indxs(lista, indx_list):
    # Removes the set of indexes from a list
    removeset = set(indx_list)
    newlist = [v for i, v in enumerate(lista) if i not in removeset]
    
    return newlist

#########################################################
#################### TIME FUNC ##########################
#########################################################

def get_dates(dates_list):
    # Gets only the date from a timestapm. For a list
    only_day = []
    for date in dates_list:
        only_day.append(date.date())
    return np.array(only_day)

def get_times(dates_list):
    # Gets only the time from a timestapm. For a list
    only_time = []
    for date in dates_list:
        only_time.append(date.time())
    return np.array(only_time)
    
def str_to_datetime(dateStr):
    # This function converts a str with format YYYY-MM-DD HH:MM:SS to datetime
    dates_datetime = []
    for ds in dateStr:
        dsplited = ds.split(" ")
        date_s = dsplited[0].split("-") # Date
        
        if (len(dsplited) > 1):  # Somo files have hours, others does not
            hour_s = dsplited[1].split(":")  # Hour 
            datetim = dt.datetime(int(date_s[0]), int(date_s[1]), int(date_s[2]),int(hour_s[0]), int(hour_s[1]))
        else:
            datetim = dt.datetime(int(date_s[0]), int(date_s[1]), int(date_s[2]))
            
        dates_datetime.append(datetim)
    return dates_datetime

def get_timeStamp(date):
    return time.mktime(date.timetuple())

def transform_time(time_formated):
    # This function accepts time in the format 2016-01-12 09:03:00
    # And converts it into the format [days] [HHMMSS]
    # Remove 
    
    data_normalized = []
    for time_i in time_formated:
        time_i = str(time_i)
#        print time_i
        time_i = time_i[0:19]
        time_i = time_i.replace("-", "")
        time_i = time_i.replace(" ", "")
        time_i = time_i.replace(":", "")
        time_i = time_i.replace("T", "")
#        print time_i
        data_normalized.append(int(time_i))
        
    return data_normalized 
    
import matplotlib.dates as mdates
def preprocess_dates(X):
    # Dealing with dates !
    ## Format of time in plot [736203.87313988095, 736204.3325892858]
    if (type(X).__name__ != "list"):
        if (type(X[0,0]).__name__ == "datetime64"):
            X = pd.to_datetime(X).T.tolist()  #  DatetimeIndex
            X = mdates.date2num(X)
#        else:  #  DatetimeIndex
#            X = X.T.tolist()[0]  
        
        
    return X
    
#########################################################
#################### File Management ##########################
#########################################################

def create_folder_if_needed (folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_allPaths(rootFolder, fullpath = "yes"):
    ## This function finds all the files in a folder
    ## and its subfolders

    allPaths = []

    for dirName, subdirList, fileList in os.walk(rootFolder):  # FOR EVERY DOCUMENT
#       print "dirName"
       for fname in fileList:
            # Read the file
            path = dirName + '/' + fname;
            if (fullpath == "yes"):
                allPaths.append(os.path.abspath(path))
            else:
                allPaths.append(path)
    
    return allPaths

def type_file(filedir):
    mime = magic.Magic()
    filetype = mime.id_filename(filedir)
#    filetype = mime.id_filename(filedir, mime=True)
    
    # This will be of the kind "image/jpeg" so "type/format"
    filetype = filetype.split(",")[0]
    return filetype

def copy_file(file_source, file_destination, new_name = ""):
    # Copies a file into a new destination.
    # If a name is given, it changes its name

    file_name = "" 
    file_path = ""
    
    file_name = file_source.split("/")[-1]
    file_path = file_source.split("/")[0]
    
    if (len(new_name) == 0): # No new name specified
        file_name = file_source.split("/")[-1]
    else:
        file_name = new_name
    
    create_folder_if_needed(file_destination)
    
    shutil.copy2(file_source, file_destination + "/" + file_name)


def simmilarity(patterns,query,algo):
    # This funciton computes the similarity measure of every pattern (time series)
    # with the given query signal and outputs a list of with the most similar and their measure.

    Npa,Ndim = patterns.shape
    sims = []
    if (algo == "Correlation"):
        for i in range(Npa):
            sim =  np.corrcoef(patterns[i],query)[1,0]
            sims.append(sim)
        sims = np.array(sims)
        sims_ored, sims_or = sort_and_get_order (sims, reverse = True )
        
    if (algo == "Distance"):
        sims = spatial.distance.cdist(patterns,np.matrix(query),'euclidean')
        sims = np.array(sims)
        sims_ored, sims_or = sort_and_get_order (sims, reverse = False )
    return sims_ored, sims_or

def convert_dates_str(X):
    # We want to convert the dates into an array of char so that we can plot 
    # this shit better, and continuous

    Xdates_str = []
    for date_i in X:
        name = date_i.strftime("%Y/%m/%d:%H:%M")
        Xdates_str.append(name)
    return Xdates_str
    
def get_Elliot_Trends (yt, Nmin = 4, Noise = -1):
    
    Nsamples, Nsec = yt.shape
    if (Nsec != 1):
        print "Deberia haber solo una senal temporal"
        return -1;
        
#    yt = yt.ravel()
    
#    yt = np.array(yt.tolist()[0])
    
    print yt.shape
    trends_list = []   # List of the trends
    
    support_t = 0   # Support index
    trend_ini = 0   # Trend start index

    support = yt[support_t]  # If support is broken then we dont have trend
    

    """ UPPING TRENDS """    
    for i in range (1,Nsamples-1):
        if (Noise == -1):
            tol = support/200
            
        #### Upper trends
        if (yt[i] > support- tol): # If if is not lower that the last min
            if (yt[i +1 ] < yt[i] - tol):  # If it went down, we have a new support
                support_t = i
                support = yt[support_t]
            
        else:   # Trend broken
            
            if ((i -1 - trend_ini) > Nmin): # Minimum number of samples of the trend
                trends_list.append([trend_ini, i -1])  # Store the trend
            
            # Start over
            trend_ini = i
            support_t = i
            support = yt[support_t]
    
    """ Lowing TRENDS """  
    
    for i in range (1,Nsamples-1):
        if (Noise == -1):
            tol = support/200
            
        #### Upper trends
        if (yt[i] < support + tol): # If if is not lower that the last min
            if (yt[i + 1] > yt[i] + tol):  # If it went up, we have a new support
                support_t = i
                support = yt[support_t]
            
        else:   # Trend broken
            
            if ((i - trend_ini) > Nmin): # Minimum number of samples of the trend
                trends_list.append([trend_ini, i -1])  # Store the trend
            
            # Start over
            trend_ini = i
            support_t = i
            support = yt[support_t]
    return trends_list
        

def support_detection(sequence, L):
    # This fuction get the support of the last L signals
    Nsamples, Nsec = sequence.shape
    
    sequence_view = sequence[-L:]
    index_min = np.argmin(sequence_view)
    
    return index_min + Nsamples - L 

def get_grids(X_data, N = [10]):
    # This funciton outputs the grids  of the given variables.
    # N is the number of points, if only one dim given, it is used to all dims
    # X_data = [Nsam][Nsig]
    Nsa, Nsig = X_data.shape
    
    ranges = []
    for i in range(Nsig):
        # We use nanmin to avoid nans
        ranges.append([np.nanmin(X_data[:,i]),np.nanmax(X_data[:,i])])
    
    grids = []
    for range_i in ranges:
        grid_i = np.linspace(range_i[0], range_i[1], N[0])
        grids.append(grid_i)
    
    return grids
    
def get_stepValues(x, y1, y2=0, step_where='pre'):
    # This function gets the appropiate x and ys for making a step plot
    # using the plot function and the fill func
    ''' fill between a step plot and 

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
#    if np.isscalar(y1):
#        y1 = np.ones_like(x) * y1
#
#    print y2
    if np.isscalar(y2):
        y2 = np.ones(x.shape) * y2
    # .astype('m8[s]').astype(np.int32)
    # .astype('m8[m]').astype(np.int32)
    y1 = fnp(y1)
#    print x.shape, y1.shape, y2.shape
#    print type(x[0,0])
#    print x[0,0]
    
    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.concatenate((y1, y2),axis = 1).T

#    print vertices.shape
    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    
    ## What we will do is just create a plot, where the next point is just
    ## the following in the same position 
    
    X =  preprocess_dates(x)
    X_new = []
    if step_where == 'pre':
        for xi in X:
            X_new.append(xi)
            X_new.append(xi)
        X_new = X_new[:-1]
#        x_steps = np.zeros(2 *x.shape[0] - 1)
#        x_steps[0::2], x_steps[1::2] = x[:,0], x[:-1,0]
        
        y_steps = np.zeros((2, 2 * x.shape[0] - 1), np.float)
        y_steps[:, 0::2], y_steps[:, 1:-1:2] = vertices[:, :], vertices[:, 1:]

    elif step_where == 'post':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    yy1, yy2= y_steps
#    print xx
#    print x_steps.shape
#    xx = preprocess_dates(ul.fnp(x_steps))
    xx = X_new
#    print yy1
#    print len(yy1)
#    print len(xx)
    # now to the plotting part:
    return xx, yy1, yy2
    
def get_foldersData(source = "FxPro"):
    rrf = "../" # relative_root_folder
    if (source == "Hanseatic"):
        storage_folder = rrf + "./storage/Hanseatic/"
        updates_folder = rrf +"../Hanseatic/MQL4/Files/"
        info_folder = storage_folder # updates_folder
        
    elif (source == "FxPro" ):
        storage_folder = rrf +"./storage/FxPro/"
        updates_folder = rrf +"../FxPro/MQL4/Files/"
        info_folder = storage_folder # updates_folder
    #    updates_folder = "../FxPro/history/CSVS/"

    elif (source == "GCI" ):
        storage_folder = rrf +"./storage/GCI/"
        updates_folder = rrf +"../GCI/MQL4/Files/"
        info_folder = storage_folder # updates_folder
    #    updates_folder = "../GCI/history/CSVS/"
    
    elif (source == "Yahoo"):
        storage_folder = rrf +"./storage/Yahoo/"
        updates_folder = rrf +"internet"
        info_folder = storage_folder # updates_folder
    return storage_folder, info_folder, updates_folder

def datesToNumbers(dates):
    # This funciton is suposed to transform the datetimes to numbers
    # that a ML learning algorithm would use.
    # For dayly data we have to take into account weekeds.
    # For intraday data we have to take into accoun other stuff.
    
    new_dates = (dates - dt.datetime(1970,1,1)).total_seconds()
    new_dates = new_dates/(60*60*24)
    return new_dates
