import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import utilities_lib as ul
import matplotlib.gridspec as gridspec
import copy
import scipy.stats as stats
from matplotlib.finance import candlestick_ohlc

def candlestick(self, data, ax = None, *args, **kwargs):

    if (type(ax) == type(None)):
        ax = self.axes
        if (ax is None):
            if (self.fig is None):
                self.init_figure()
            ax = self.create_axes()
    plotting = candlestick_ohlc(ax,data, *args, **kwargs)
    
    self.plots_type.append(["candlestick"])
    self.plots_list.append([plotting]) # We store the pointers to the plots
    
    data_i = [data, ax]
    self.Data_list.append(data_i)
    return plotting
    
def histogram(self, residual, 
              labels = ["return"], 
              nf = 1,
              alpha = 0.9,
              bins = 20,
              fake_dates = 1
              ):
    ## Fit a gaussian and plot it
    mean = np.mean(residual)
    std = np.std(residual)
    
    hist, bin_edges = np.histogram(residual, density=True, bins = bins)
    self.bar(bin_edges[:-1], hist, 
           labels = ["Distribution",labels[0], "Probability"],
           legend = ["%s. m: %0.3f, s: %0.3f"%(labels[0],mean,std)],
           alpha = alpha,
           nf = nf)
    ## Fit a gaussian and plot it
    mean = np.mean(residual)
    std = np.std(residual)
    
    x_grid = np.linspace(min(bin_edges), max(bin_edges), num = 100)
    Z = (x_grid - mean)/std

    
    y_values = stats.norm.pdf(Z) / std# * stats.norm.pdf(-mean/std)
    self.plot(x_grid, y_values, nf = 0)


def Velero_graph(self, data, 
                 labels = [],nf = 1,
                fake_dates = 1,
                 ws = -1):
    """ This function plots the Heiken Ashi of the data 
    data[4][Ns] """
    
    colorFill = "green"  # Gold
    colorBg = "#7fffd4" # Aquamarine
    colorInc = '#FFD700'
    colorDec = "red" 
    

    Close = data["Close"]
    Open = data["Open"]
    High = data["High"]
    Low = data["Low"]
    Volume = data["Volume"]
    dates = data.index

    if (fake_dates == 1):
        dates = np.array(range(len(dates)))
        
    Nsam = len(dates)  # Number of samples
    incBox = []
    decBox = []
    allBox = []
    # Calculate upper and lowe value of the box and the sign
    for i in range(Nsam):
        diff = Close[i] - Open[i]
    #        print diff
        if (diff >= 0):
            incBox.append(i)
        else:
            decBox.append(i)
        allBox.append(i)
        
    ## All_bars !!
    self.bar(dates[allBox], High[allBox] - Low[allBox], bottom = Low[allBox],
             width = 0.1, color = colorFill, ws = ws, nf = 1)  
             
    self.bar(dates[allBox], abs(Open[allBox] - Close[allBox]), 
             bottom = np.min((Close[allBox],Close[allBox]),axis = 0),
             width = 0.9, color = colorDec, ws = ws, nf = 0)


#    ## Increasing bars !!
#    self.bar(dates[incBox], Close[incBox] - Open[incBox], bottom = Open[incBox],
#             width = 0.9, color = colorInc, ws = ws, nf = nf)
#    self.bar(dates[incBox], High[incBox] - Close[incBox], bottom = Close[incBox],
#             width = 0.1, color = colorFill, ws = ws, nf = 0)     
#    self.bar(dates[incBox], Open[incBox] - Low[incBox], bottom = Low[incBox],
#             width = 0.1, color = colorFill, ws = ws, nf = 0)  
#             
#    ## Decreasing bars !!
#    self.bar(dates[decBox], Open[decBox] - Close[decBox], bottom = Close[decBox],
#             width = 0.9, color = colorDec, ws = ws, nf = 0)
#    self.bar(dates[decBox], High[decBox] - Open[decBox], bottom = Open[decBox],
#             width = 0.1, color = colorFill, ws = ws, nf = 0)     
#    self.bar(dates[decBox], Close[decBox] - Low[decBox], bottom = Low[decBox],
#             width = 0.1, color = colorFill, ws = ws, nf = 0)  


#    
    plt.ylim(min(Low)* (0.95))
   
#     Plot the volume
    self.bar(dates, Volume, alpha = 0.5,
             width = 0.9, color = colorBg, ws = ws, nf = 0, na = 1) 
    
    plt.ylim(plt.ylim()[0], max(Volume)* 4)


def Heiken_Ashi_graph(self, data, labels = [], nf = 1):
    r_close = data["Close"].values
    r_open = data["Open"].values
    r_max = data["High"].values
    r_min = data["Low"].values
    x_close  = (r_close + r_open + r_max + r_min)/4
    x_open = (r_close[1:] + x_close[:-1])/2  # Mean of the last 
    
    # Insert the first open sin we cannot calcualte the prevoius
    # The other opion is to eliminate the first of everyone
    x_open = np.insert(x_open, 0, r_open[0], axis = 0)
    
#    print x_close.shape, x_open.shape
    x_max = np.max(np.array([r_max,x_close,x_open]), axis = 0)
    x_min = np.min(np.array([r_min,x_close,x_open]), axis = 0)
    
    
    ### Lets create another pandas dataframe with this data
    new_data = copy.deepcopy(data)
    new_data["Close"] = x_close
    new_data["Open"] = x_open
    new_data["High"] = x_max
    new_data["Low"] = x_min
    
    self.Velero_graph(new_data, labels = [], nf = 1)
    
#    x_close  = np.mean(data,0)
#    x_open = data[0][1:] + upper_box[:-1]  # Mean of the last 

def plot_timeSeriesRange(self, X, Y, sigma, k = 1.96, nf = 0, legend = ["95% CI f(x)"]):
    # Plots the time series with its 1.96 percent interval
    gl = self
    gl.plot(X,Y, nf = 0,legend = legend)
    gl.plot_filled(X, np.concatenate([Y - 1.9600 * sigma,
           Y + 1.9600 * sigma],axis = 1), alpha = 0.5, nf = 0)
    
def plot_timeRegression(self,Xval, Yval, sigma,
                        Xtr, Ytr,sigma_eps = None, 
                        labels = ["Gaussian Process Estimation","Time","Price"]):
    # This is just a general plotting funcion for a timeSeries regression task:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    # eps is the estimated noise of the training samples.
    """
    sigma is the std of each of the validation samples 
    sigma_eps is the considered observation noise
    """
    gl = self
    gl.plot(Xval,Yval, labels = labels, legend = ["Estimated Mean"])
    
    gl.plot_filled(Xval, np.concatenate([Yval - 1.9600 * sigma,
           Yval + 1.9600 * sigma],axis = 1), alpha = 0.5, legend = ["95% CI f(x)"], nf = 0)
    
    # If we are given some observaiton noise we also plot the estimation of it
    if type(sigma_eps) != type(None):
        # Ideally we have a func that tells us for each point, what is the observation noise.
        # We are suposed to know what is those values for training, 
        # TODO: And for testing ? Would that help ? I think we are already assuming so in the computation of K.
        # The inner gaussian process can also be specified "alpha", we will research more about that later.
        # I guess it is for hesterodasticity.
        # We are gonna consider that the observation noise of the predicted samples is homocedastic.
        sigma = np.sqrt(sigma**2 + ul.fnp(sigma_eps)[0]**2)
        # TODO: maybe in the future differentiate between sigma_eps and dy
        gl.plot_filled(Xval, np.concatenate([Yval - 1.9600 * sigma,
               Yval + 1.9600 * sigma],axis = 1), alpha = 0.2, legend = ["95% CI y(x)"], nf = 0)
                       
    # TODO: what is this     ec='None'

    if type(sigma_eps) != type(None):
        plt.errorbar(Xtr.ravel(), Ytr, sigma_eps, fmt='k.', markersize=10, label=u'Observations')
    else:
        gl.scatter(Xtr, Ytr, legend = ["Training Points"], nf = 0, color = "k")
        
def putamadere(wd):
    ## Previous velero plotting
    # The width of the bowes is the same for all and the x position is given by
    # the position 

    ## Obtain the 4 parameters of the square
    ## Box parameters

    """ WE are gonna plot the Velas in one axis and the volume in others """

    ####### Plot all the cubes !!!!
    fig, ax = plt.subplots()
    
    fig.facecolor = colorBg
    
    for i in range(Ns):
        # Calculate upper and lowe value of the box and the sign
        diff = r_close[i] - r_open[i]
#        print diff
        if (diff >= 0):
            low_box = r_open[i]
            sign = colorInc
        else:
            low_box = r_close[i]
            sign = colorDec
        
        # Create the box
        ax.broken_barh([(date_indx[i] + 0.05, 1 - 0.1)],
                        (low_box, abs(diff)),
                         facecolors=sign)
                         
        # Create the box upper line                
        ax.broken_barh([(date_indx[i] + 0.45, 0.1)],
                        (low_box + abs(diff), r_max[i] - low_box + abs(diff)),
                         facecolors= "red")
                         
        # Create the box lower line                
        ax.broken_barh([(date_indx[i] + 0.45, 0.1)],
                        (r_min[i] , low_box - r_min[i]),
                         facecolors= "red")
    
    """ PLOT VOLUME """ 
    ax1_2 = ax.twinx()
    
    #ax1_2.plot(date, (ask-bid))
#    print volume.shape
#    print len(date)
    ax1_2.bar(date, volume, facecolor= colorFill,alpha=.5)
#    ax1_2.fill_between(date, 0, volume, facecolor= colorFill,alpha=.5)
    
#    broken_barh ( xranges, yrange, **kwargs)
#        xranges	sequence of (xmin, xwidth)
#        yrange	 sequence of (ymin, ywidth)
        
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    
    if (len(labels) >3 ):
        plt.legend(labels[3])
    
#    plt.grid()
    plt.show()
    