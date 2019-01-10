import numpy as np
import matplotlib.pyplot as plt
import utilities_lib as ul
import copy

from matplotlib.patches import Rectangle
import datetime as dt
from matplotlib import collections  as mc
from matplotlib.lines import Line2D

def barchart(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [], legend = [],       # Basic Labelling
        color = None,  lw = 2, lw2 = 2,alpha = 1.0,  # Basic line properties
        nf = 0, na = 0,          # New axis. To plot in a new axis         # TODO: shareX option
        ax = None, position = [], projection = "2d", # Type of plot
        sharex = None, sharey = None,
        fontsize = 20,fontsizeL = 10, fontsizeA = 15,  # The font for the labels in the axis
        xlim = None, ylim = None, xlimPad = None, ylimPad = None, # Limits of vision
        ws = None, Ninit = 0,     
        loc = "best",    
        dataTransform = None,
        xaxis_mode = None,yaxis_mode = None,AxesStyle = None   # Automatically do some formatting :)
       ):         
           
    
    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    NpX, NcX = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)

    ############### CALL PLOTTING FUNCTION ###########################

#    X = X.astype(dt.datetime)
#    self.X = self.X.astype(dt.datetime)
#    print X.shape
    High,Low, Open,Close = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    X = ul.preprocess_dates(X)
    self.X = X
    
#    print X
    width_unit = self.get_barwidth(X)
    dist = width_unit /2.2
    # High-Low
    linesHL = [[(X[i].astype(dt.datetime),Low[i]),(X[i].astype(dt.datetime), High[i])] for i in range(NpX)]
    linesO = [[(X[i].astype(dt.datetime) - dist,Open[i]),(X[i].astype(dt.datetime), Open[i])] for i in range(NpX)]
    linesC = [[(X[i].astype(dt.datetime),Close[i]),(X[i].astype(dt.datetime) + dist, Close[i])] for i in range(NpX)]

#    lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
#    print mdates.date2num(X[i,0].astype(dt.datetime)), type(mdates.date2num(X[i,0].astype(dt.datetime))) 
    self.zorder = self.zorder + 1  # Setting the properties
    colorFinal = self.get_color(color)
    
    # TODO: Handle the legend better
    lcHL = mc.LineCollection(linesHL, colors= colorFinal, linewidths=lw, antialiased=True, label = legend[0])
    lcO = mc.LineCollection(linesO, colors= colorFinal, linewidths=lw2, antialiased=True)
    lcC = mc.LineCollection(linesC, colors= colorFinal, linewidths=lw2, antialiased=True)
    ax.add_collection(lcHL)
    ax.add_collection(lcO)
    ax.add_collection(lcC)
    
    
    ax.autoscale()  # TODO: The zoom is not changed if we do not say it !
#    ax.margins(0.1)
    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(ax = ax, xlim = xlim,ylim = ylim, xlimPad = xlimPad,ylimPad = ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)

#    xfmt = mdates.DateFormatter('%b %d')
#    ax.xaxis.set_major_formatter(xfmt)
    return ax


def candlestick(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [], legend = [],       # Basic Labelling
        color = None,  lw = 2, alpha = 1.0,  # Basic line properties
        nf = 0, na = 0,          # New axis. To plot in a new axis         # TODO: shareX option
        ax = None, position = [], projection = "2d", # Type of plot
        sharex = None, sharey = None,
        fontsize = 20,fontsizeL = 10, fontsizeA = 15,  # The font for the labels in the axis
        xlim = None, ylim = None, xlimPad = None, ylimPad = None, # Limits of vision
        ws = None, Ninit = 0,     
        loc = "best",    
        dataTransform = None,
        xaxis_mode = None,yaxis_mode = None,AxesStyle = None,   # Automatically do some formatting :)
        barwidth = None, colorup = "g", colordown = "r"
       ):
       
    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)
    
    ##################### PREPROCESSING AND PLOTTING #######################
    
    # Prepare the data
    openp = Y[self.start_indx:self.end_indx,0]
    closep = Y[self.start_indx:self.end_indx,1]
    highp = Y[self.start_indx:self.end_indx,2]
    lowp = Y[self.start_indx:self.end_indx,3]

    dates = X[self.start_indx:self.end_indx]
    dates = ul.preprocess_dates(dates)
    if (type(barwidth) == type(None)):
        barwidth = self.get_barwidth(dates, barwidth) * 0.8
        
    # PLOTTING !!
    Npoints = dates.size
    
    OFFSET = barwidth / 2.0
    
    line_factor = 0.15
    barwidth_HL = barwidth * line_factor 
    OFFSET_HL = barwidth_HL / 2.0
    
    lines = []
    patches = []
    for i in range(Npoints):
        if closep[i] >= openp[i] :
            color = colorup
            baseRectable = openp[i]
        else:
            color = colordown
            baseRectable = closep[i]
            
        height = np.abs(openp[i]  - closep[i])
        
        ## High-low line

#        line_HL = Line2D(
#            xdata=(dates[i],dates[i]), ydata=(lowp[i], highp[i]),
#            color=color,
#            linewidth=lw,
#            antialiased=True,
#        )

        rect_HL = Rectangle(
            xy=(dates[i] - OFFSET_HL, lowp[i]),
            width=barwidth_HL,
            height=highp[i] - lowp[i],
            facecolor=color,
            edgecolor=color,
        )
        
#        print type(dates[i]), type(OFFSET)
        ## Open Close rectangle
        rect_OP = Rectangle(
            xy=(dates[i] - OFFSET, baseRectable),
            width=barwidth,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect_OP.set_alpha(alpha)
#
#        lines.append(line_HL)
#        patches.append(rect_OP)
        
#        ax.add_line(line_HL)
        ax.add_patch(rect_OP)
        ax.add_patch(rect_HL)
        
#    lines = mc.LineCollection(lines)
#    ax.add_collection(lines)
#    ax.add_collection(patches)
    
    ax.autoscale()  # TODO: The zoom is not changed if we do not say it !
    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(ax = ax, xlim = xlim,ylim = ylim, xlimPad = xlimPad,ylimPad = ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    self.plots_type.append(["candlestick"])
#    self.plots_list.append([plotting]) # We store the pointers to the plots
    
    data_i = [Y, ax]
    self.Data_list.append(data_i)
    
    return ax

def histogram(self, X,  bins = 20, orientation = "vertical",
        *args, **kwargs):   
             
    hist, bin_edges = np.histogram(X, density=True, bins = bins)
    self.bar(bin_edges[:-1], hist, orientation = orientation,*args, **kwargs)


#    if (orientation == "vertical"):
#        self.plot(x_grid, y_values, nf = 0, *args, **kwargs)
#    else:
#        self.plot(y_values, x_grid, nf = 0, *args, **kwargs)

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
             barwidth = 0.1, color = colorFill, ws = ws, nf = 1)  
             
    self.bar(dates[allBox], abs(Open[allBox] - Close[allBox]), 
             bottom = np.min((Close[allBox],Close[allBox]),axis = 0),
             barwidth = 0.9, color = colorDec, ws = ws, nf = 0)


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
             barwidth = 0.9, color = colorBg, ws = ws, nf = 0, na = 1) 
    
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
                        labels = ["Gaussian Process Estimation","Time","Price"], nf = 0):
    # This is just a general plotting funcion for a timeSeries regression task:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    # eps is the estimated noise of the training samples.
    """
    sigma is the std of each of the validation samples 
    sigma_eps is the considered observation noise
    """
    
    Xval = ul.fnp(Xval)
    Yval = ul.fnp(Yval)
    Xtr = ul.fnp(Xtr)
    Ytr = ul.fnp(Ytr)

    sigma = ul.fnp(sigma)
    sigma_eps = ul.fnp(sigma_eps)
    
    gl = self
    gl.plot(Xval,Yval, labels = labels, legend = ["Estimated Mean"], nf = nf)
    
    gl.plot_filled(Xval, np.concatenate([Yval - 1.9600 * sigma,
           Yval + 1.9600 * sigma],axis = 1), alpha = 0.5, legend = ["95% CI f(x)"])
    
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
               Yval + 1.9600 * sigma],axis = 1), alpha = 0.2, legend = ["95% CI y(x)"])
                       
    # TODO: what is this     ec='None'

    if type(sigma_eps) != type(None):
        if (ul.fnp(sigma_eps).size == 1):
            sigma_eps = np.ones((Xtr.size,1)) * sigma_eps
        plt.errorbar(Xtr.ravel(), Ytr, sigma_eps, fmt='k.', markersize=10, label=u'Observations')
    else:
        gl.scatter(Xtr, Ytr, legend = ["Training Points"], color = "k")
        
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
    