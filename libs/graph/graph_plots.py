import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities_lib as ul
import matplotlib.gridspec as gridspec
import copy
import graph_basic as grba
import datetime as dt
import pylab as pl
from matplotlib import collections  as mc
# The common properties will be explained here once and shortened in the rest
def plot(self, X = [],Y = [],           # X-Y points in the graph.
        labels = [], legend = [],       # Basic Labelling
        color = None,  lw = 2, alpha = 1.0,  # Basic line properties
        # Figure and axes management 
        nf = 0,          # New figure. If 1 it will either create new fig or go to next subplot
        na = 0,          # New axis. To plot in a new axis         # TODO: shareX option
        
        ax = None,      # Axes where this will be plotted. If none, it will be the last one.
        position = [],   # If given it will create a new axes [x,y,w,h]
        sharex = None, sharey = None, # When nf = 1, we are creating a new figure and we can choose
                                     # that new axes share the same x axis or yaxis than another one.
        projection = "2d", # Type of plot
        
        # Advanced fonts
        fontsizes = [None, None, None],   # This is the fontsizes of [tittle, xlabel and ylabel, xticks and yticks]
        
        # Layout options
        xlimPad = None, ylimPad = None, # Padding in percentage of the plotting, it has preference
        xlim = None, ylim = None, # Limits of vision
        
        # Widgets options
        ws = None,      # Only plotting the last window of the data.
        initX = None,   # Initial point to plot
        # Basic parameters that we can usually find in a plot
        loc = "best",    # Position of the legend
        
        ### Special options 
        fill = 0,  #  0 = No fill, 1 = Fill and line, 2 = Only fill
        fill_offset = 0,  # The 0 of the fill
        ls = "-",
        marker = [None, None, None],
        
        # Formatting options
        xaxis_mode = None,# Perfect for a few good ones :)
        yaxis_mode = None, # Perfect for a few good ones :)
        AxesStyle = None,   # Automatically do some formatting :)
        dataTransform = None   # Specify if we are gonna format somehow the data. 
                            # for intraday for example.
        ):         
    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y, dataTransform = dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)
    
    if (Y.size != 0):  # This would be just to create the axes
    ############### CALL PLOTTING FUNCTION ###########################
        for i in range(NcY):  # We plot once for every line to plot
            self.zorder = self.zorder + 1  # Setting the properties
            colorFinal = self.get_color(color)
            legend_i = None if i >= len(legend) else legend[i]
            alpha_line = alpha if fill == 0 else 1
            plot_i, = ax.plot(X[self.start_indx:self.end_indx],Y[self.start_indx:self.end_indx:,i], 
                     lw = lw, alpha = alpha_line, color = colorFinal,
                     label = legend_i, zorder = self.zorder,
                     ls = ls, marker = marker[0], markersize = marker[1], markerfacecolor = marker[2])
            plots.append(plot_i)
            plots_typ.append("plot")
            # Filling if needed
            if (fill == 1):  
                self.fill_between(x = X[self.start_indx:self.end_indx],
                                  y1 = Y[self.start_indx:self.end_indx,i],
                                    y2 = fill_offset, color = colorFinal,alpha = alpha)

    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    return ax

def stem(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None]
       ):         

    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)

    ############### CALL PLOTTING FUNCTION ###########################
    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1  # Setting the properties
        colorFinal = self.get_color(color)
        legend_i = None if i >= len(legend) else legend[i]
        markerline, stemlines, baseline = ax.stem(X,Y[:,i], lw = lw, alpha = alpha, 
                 color = colorFinal, label = legend_i, zorder = self.zorder, 
                 markerfmt = marker[0], markersize = marker[1], markerfacecolor = marker[2],
                 antialiased = True)
        plt.setp(markerline, 'markerfacecolor', marker[2])
        plt.setp(baseline, 'color', 'r', 'linewidth', 2)
        plt.setp(stemlines, 'linewidth', lw)
    
    ############### Last setting functions ###########################
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    return ax
    

#a=np.datetime64('2002-06-28').astype(datetime)
#plot_date(a,2)
import matplotlib.dates as mdates

def add_vlines(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None]
       ):         

    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)

    ############### CALL PLOTTING FUNCTION ###########################
#    X = X.astype(dt.datetime)
#    self.X = self.X.astype(dt.datetime)
    X = ul.preprocess_dates(X)
    self.X = X
    
    lines = [[(X[i].astype(dt.datetime), Y[i,0]),(X[i].astype(dt.datetime), Y[i,1])] for i in range(NpX)]
#    lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
#    print mdates.date2num(X[i,0].astype(dt.datetime)), type(mdates.date2num(X[i,0].astype(dt.datetime))) 
    
    lc = mc.LineCollection(lines, colors= "k", linewidths=lw)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    
    ############### Last setting functions ###########################
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    return ax
    
def add_hlines(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None]
       ):         

    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)

    ############### CALL PLOTTING FUNCTION ###########################
#    X = X.astype(dt.datetime)
#    self.X = self.X.astype(dt.datetime)
    X = ul.preprocess_dates(X)
    self.X = X
    
    width_unit = self.get_barwidth(X)
    width = width_unit * (1 - 0.8)/2
    # TODO: might be wrong to use Npx due to subselection
    lines = [[(X[i].astype(dt.datetime)- width, Y[i,0]),(X[i].astype(dt.datetime) +width, Y[i,0])] for i in range(NpY)]
#    lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
#    print mdates.date2num(X[i,0].astype(dt.datetime)), type(mdates.date2num(X[i,0].astype(dt.datetime))) 
    
    lc = mc.LineCollection(lines, colors= "k", linewidths=lw)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    
    ############### Last setting functions ###########################
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    return ax

def scatter(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None]
       ):         

    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    print X.shape
    D = []
    for i in range(NpX):
        D.append(X[i,0])
    self.X = D
    X = D
#    print X
    
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)


    ############### CALL SCATTERING FUNCTION ###########################
    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1  # Setting the properties
        colorFinal = self.get_color(color)
        legend_i = None if i >= len(legend) else legend[i]
        
        scatter_i =  ax.scatter(X,Y, lw = lw, alpha = alpha, color = colorFinal,
                    label = legend_i, zorder = self.zorder)
        plots.append(scatter_i)
        plots_typ.append("scatter")

    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    
    return ax


def step(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [],     # Labels for tittle, axis and so on.
        legend = [],     # Legend for the curves plotted
        nf = 1,          # New figure
        na = 0,          # New axis. To plot in a new axis
        # Basic parameters that we can usually find in a plot
        color = None,    # Color
        lw = 2,          # Line width
        alpha = 1.0,      # Alpha
        
        fontsize = 20,   # The font for the labels in the title
        fontsizeL = 10,  # The font for the labels in the legeng
        fontsizeA = 15,  # The font for the labels in the axis
        loc = "best",
        
        where = "pre", # pre post mid ## TODO, part of the step. How thw shit is done
        ### Super Special Shit !!
        fill = 0,  #  0 = No fill, 1 = Fill and line, 2 = Only fill
        
        # Widgets shit !!
        ws = -1      # Only plotting the last window of the data.
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y,dataTransform)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure


    
    ##################################################################
    ############### CALL PLOTTING FUNCTION ###########################
    ##################################################################
    ## TODO. Second case where NcY = NcX !!

    if (ws == -1):  # We only show the last ws samples
        ws = NpX
        
    plots = []
    plots_typ = []
    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1
        colorFinal = self.get_color(color)
        if (i >= len(legend)):
            plot_i, = plt.step(X[(NpX-ws):],Y[(NpX-ws):,i], lw = lw, alpha = alpha, 
                     color = colorFinal, zorder = self.zorder,  where = where)
        else:
#            print X.shape
#            print Y[:,i].shape
            plot_i, = plt.step(X[(NpX-ws):],Y[(NpX-ws):,i], lw = lw, alpha = alpha, color = colorFinal,
                     label = legend[i], zorder = self.zorder, where = where)
        
        if (fill == 1):  ## Fill this shit !!
            XX,YY1, YY2 = ul.get_stepValues(X[(NpX-ws):],Y[(NpX-ws):,i], y2 = 0, step_where = where)
            fill_i = self.fill_between(XX,YY1,y2 = 0, color = colorFinal,alpha = alpha)
        plots.append(plot_i)
        plots_typ.append("plot")
        

    ## Store pointers to variables for interaction
    self.plots_type.append(plots_typ)
    self.plots_list.append(plots) # We store the pointers to the plots
    
    data_i = [X,Y]
    self.Data_list.append(data_i)
    
    self.update_legend(legend,NcY,loc = loc)
    self.format_axis( nf, fontsize = fontsizeA, wsize = ws, val = NpX-ws)
    

    if (na == 1 or nf == 1):
        self.format_plot()
    
    ax = self.axes
    return ax

def plot_filled(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None],
        fill_mode =  "independent" # "between", "stacked","independent"
       ):         

    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)

    x = X[self.start_indx:self.end_indx]
    ############### CALL PLOTTING FUNCTION ###########################
    for i in range(0,NcY -1):  # We plot once for every line to plot

        if (fill_mode ==  "stacked"):
            if (i == 0):   # i  for i in range(NcY)
                y1 = Y[self.start_indx:self.end_indx,i]
                y2 = 0 + y1
            else:
                y2 += Y[self.start_indx:self.end_indx,i-1]
                y1 = y2 + Y[self.start_indx:self.end_indx,i]
                
        elif(fill_mode ==  "between"):
                y2 = Y[self.start_indx:self.end_indx,i-1]
                y1 = Y[self.start_indx:self.end_indx,i]
                
        y2 = Y[self.start_indx:self.end_indx,i]
        y1 = Y[self.start_indx:self.end_indx,i+1]
        
        self.zorder = self.zorder + 1  # Setting the properties
        colorFinal = self.get_color(color)
        legend_i = None if i >= len(legend) else legend[i]
        # With this we add the legend ?
#        plot_i, = ax.plot([X[0],X[0]],[y1[0],y1[0]], lw = lw, alpha = alpha, 
#                 color = colorFinal, zorder = self.zorder)
        fill_i = self.fill_between(x = x,y1 = y1 ,y2 = y2, color = colorFinal,alpha = alpha)
        
        plots.append(fill_i)
        plots_typ.append("plot")
    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)

    return ax


def fill_between(self, x, y1,  y2 = 0, 
                 ax = None, where = None, alpha = 1.0 , color = "#888888", 
                 legend = [],
                 *args, **kwargs):
    # This function fills a unique plot.
    ## We have to fucking deal with dates !!
    # The fill function does not work properly with datetime64
                 
    x = ul.fnp(x)
    y1 = ul.fnp(y1)
    if (type(ax) == type(None)):
        ax = self.axes
    x =  ul.preprocess_dates(x)
    x = ul.fnp(x)
#    print len(X), len(ul.fnp(Yi).T.tolist()[0])
#    print type(X), type(ul.fnp(Yi).T.tolist()[0])
#    print X.shape
#    print len(X.T.tolist()), len(ul.fnp(Yi).T.tolist()[0])
    x = x.T.tolist()[0]
    
#    print x
    y1 = ul.fnp(y1).T.tolist()[0]

    
    if (where is not None):
#        print len(x), len(y1), len(where)
        
        where = ul.fnp(where)
#        where = np.nan_to_num(where)
        where = where.T.tolist()[0]
        
    y2 = ul.fnp(y2)
    if (y2.size == 1):
        y2 = y2[0,0]
    else:
        y2 = y2.T.tolist()[0]
#        print where[0:20]
#        print y2
#    print len(where)
#    print x[0:5], y1[0:5]
    
    ln = ax.fill_between(x = x, y1 = y1, y2 = y2, where = where,
                     color = color, alpha = alpha, zorder = self.zorder) #  *args, **kwargs) 

    self.plots_type.append(["fill"])
    self.plots_list.append([ln]) # We store the pointers to the plots
    
    data_i = [x,y1,y2, where, ax, alpha,color, args, kwargs]
    self.Data_list.append(data_i)
            
    return ln

def bar(self, X = [],Y = [],  # X-Y points in the graph.
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
        marker = [" ", None, None],
        fill_mode =  "independent", # "between", "stacked","independent"
        # Particular pararm
        orientation = "vertical",
        barwidth = None,      # Rectangle width
        bottom = None,    ## If the y-axis start somewhere else
        despx = 0      # Displacement in the x axis, it is done for the dates
                        # so that we can move some other things (Velero graph)
       ):         

    # Management of the figure and properties
    ax = self.figure_management(nf, na, ax = ax, sharex = sharex, sharey = sharey,
                      projection = projection, position = position)
    ## Preprocess the data given so that it meets the right format
    X, Y = self.preprocess_data(X,Y,dataTransform)
    NpY, NcY = Y.shape
    plots,plots_typ =  self.init_WidgetData(ws)
    
    ## We asume that X and Y have the same dimensions
    print self.formatXaxis
    if (self.formatXaxis == "dates" or self.formatXaxis == "intraday"):
        X = ul.preprocess_dates(X)
        print "Formating bar X to dates"
    if (type(barwidth) == type(None)):
        barwidth = self.get_barwidth(X, barwidth) * 0.8
        
    if (Y.size != 0):  # This would be just to create the axes
    ############### CALL PLOTTING FUNCTION ###########################
        for i in range(NcY):  # We plot once for every line to plot
            self.zorder = self.zorder + 1  # Setting the properties
            colorFinal = self.get_color(color)
            legend_i = None if i >= len(legend) else legend[i]
            if(type(bottom) != type(None)):
                bottom = bottom[self.start_indx:self.end_indx]
            if (orientation == "vertical"):
                plot_i  = self.axes.bar(X[self.start_indx:self.end_indx], Y[self.start_indx:self.end_indx:,i], 
                            width = barwidth, align='center',
                              facecolor= colorFinal,alpha=alpha,
                              label = legend_i, zorder = self.zorder,
                              bottom = bottom)
            else:  # horixontal
                plot_i  = self.axes.bar(width = Y[self.start_indx:self.end_indx:,i], 
                              height = barwidth, align='center',
                              facecolor= colorFinal,alpha=alpha,
                              label = legend_i, zorder = self.zorder,
                              left = bottom,
                              bottom = X[self.start_indx:self.end_indx],
                             orientation = "horizontal")
            plots.append(plot_i)
            plots_typ.append("plot")

    ############### Last setting functions ###########################
    self.store_WidgetData(plots_typ, plots)     # Store pointers to variables for interaction
    
    self.update_legend(legend,NcY,ax = ax, loc = loc)    # Update the legend 
    self.set_labels(labels)
    self.set_zoom(xlim,ylim, xlimPad,ylimPad)
    self.format_xaxis(ax = ax, xaxis_mode = xaxis_mode)
    self.format_yaxis(ax = ax, yaxis_mode = yaxis_mode)
    self.apply_style(nf,na,AxesStyle)
    return ax
    



#fig, ax_list = plt. subplots(3, 1)
#x = y = np.arange(5)
#
#for ax, where in zip(ax_list, ['pre', 'post', 'mid']):
#    ax.step(x, y, where=where, color='r', zorder=5, lw=5)
#    fill_between_steps(ax, x, y, 0, step_where=where)