import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import utilities_lib as ul
import matplotlib.gridspec as gridspec
import copy
import graph_basic as grba

def plot(self, X = [],Y = [],  # X-Y points in the graph.
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
        loc = "best",    # Position of the legend
        projection = "2d", # Type of plot
        
        position = [],  # The position properties of the axes [x,y,w,h]
        ### Super Special Shit !!
        fill = 0,  #  0 = No fill, 1 = Fill and line, 2 = Only fill
        fill_offset = 0,  # The 0 of the fill
        ax = None,  # Axes where this will be plotted. If none, it will be the last one.
        # Widgets shit !!
        ws = -1,      # Only plotting the last window of the data.
        Ninit = 0,     # Initial point to plot
        
        # Layout options
        maxY = 1.0  # Relative maximum height of the plot. Lo limit it
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    ax = self.figure_management(nf, na, labels, fontsize, ax = ax, position = position, projection = projection)

#    return .1
    ##################################################################
    ############### CALL PLOTTING FUNCTION ###########################
    ##################################################################
    ## TODO. Second case where NcY = NcX !!
#    X =  grba.preprocess_dates(X)
    if (ws == -1):  # We only show the last ws samples
        ws = NpX
        
    plots = []
    plots_typ = []
    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1
        colorFinal = self.get_color(color)
        if (i >= len(legend)):
            plot_i, = ax.plot(X[(NpX-ws):],Y[(NpX-ws):,i], lw = lw, alpha = alpha, 
                     color = colorFinal, zorder = self.zorder)
        else:
#            print X.shape
#            print Y[:,i].shape
            plot_i, = ax.plot(X[(NpX-ws):],Y[(NpX-ws):,i], lw = lw, alpha = alpha, color = colorFinal,
                     label = legend[i], zorder = self.zorder)
        
        if (fill == 1):  ## Fill this shit !!
            fill_i = self.fill_between(x = X[(NpX-ws):],y1 = Y[(NpX-ws):,i],y2 = fill_offset, color = colorFinal,alpha = alpha)
        
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
        pass
    
    if (maxY < 1.0):
#        print type(Y), Y.shape
#        print Y
        max_signal = np.max(Y[~np.isnan(Y)])
        min_signal = np.min(Y[~np.isnan(Y)])
#        print min_signal, max_signal
        self.set_ylim(ymin = min_signal, ymax = (max_signal - min_signal)*(1.0/maxY))
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
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    self.figure_management(nf, na, labels, fontsize)

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
        labels = [],     # Labels for tittle, axis and so on.
        legend = [],     # Legend for the curves plotted
        nf = 1,          # New figure
        na = 0,          # New axis. To plot in a new axis
        # Basic parameters that we can usually find in a plot
        color = None,    # Color
        lw = 4,          # Line width
        alpha = 1.0,      # Alpha
        
        fontsize = 20,   # The font for the labels in the title
        fontsizeL = 10,  # The font for the labels in the legeng
        fontsizeA = 15,  # The font for the labels in the axis
        loc = "best",
        projection = "2d",
        
        position = [],  # The position properties of the axes [x,y,w,h]
        ### Super Special Shit !!
        fill = 0,  #  0 = No fill, 1 = Fill and line, 2 = Only fill
        
        ax = None,  # Axes where this will be plotted. If none, it will be the last one.
        # Widgets shit !!
        ws = -1,      # Only plotting the last window of the data.
        Ninit = 0     # Initial point to plot
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    ax = self.figure_management(nf, na, labels, fontsize, ax = ax, position = position, projection = projection)

#    return .1
    ##################################################################
    ############### CALL PLOTTING FUNCTION ###########################
    ##################################################################
    ## TODO. Second case where NcY = NcX !!
#    X =  grba.preprocess_dates(X)
    if (ws == -1):  # We only show the last ws samples
        ws = NpX
        
    plots = []
    plots_typ = []
    for i in range(0,NcY -1):  # We plot once for every line to plot
        self.zorder = self.zorder + 1
        colorFinal = self.get_color(color)
        
#        if (i == 0):   # i  for i in range(NcY)
#            y1 = Y[(NpX-ws):,i]
#            y2 = 0 + y1
#        else:
#            y2 += Y[(NpX-ws):,i-1]
#            y1 = y2 + Y[(NpX-ws):,i]
        
        y2 = Y[(NpX-ws):,i]
        y1 = Y[(NpX-ws):,i+1]
        
        if (i >= len(legend)):
            plot_i, = ax.plot([X[0],X[0]],[y1[0],y1[0]], lw = lw, alpha = alpha, 
                     color = colorFinal, zorder = self.zorder)
        else:
#            print X.shape
#            print Y[:,i].shape
            plot_i, = ax.plot([X[0],X[0]],[y1[0],y1[0]], lw = lw, alpha = alpha, color = colorFinal,
                     label = legend[i], zorder = self.zorder)
                     
        fill_i = self.fill_between(x = X[(NpX-ws):],y1 = y1 ,y2 = y2, color = colorFinal,alpha = alpha)
        
#        plots.append(fill_i)
#        plots_typ.append("plot")
        
    ## Store pointers to variables for interaction
    self.plots_type.append(plots_typ)
    self.plots_list.append(plots) # We store the pointers to the plots
    
    data_i = [X,Y]
    self.Data_list.append(data_i)
    
#    print legend, NcY
    self.update_legend(legend,NcY,loc = loc)
    self.format_axis( nf, fontsize = fontsizeA, wsize = ws, val = NpX-ws)

    if (na == 1 or nf == 1):
        self.format_plot()
        pass
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
                     color = color, alpha = alpha) #  *args, **kwargs) 

    self.plots_type.append(["fill"])
    self.plots_list.append([ln]) # We store the pointers to the plots
    
    data_i = [x,y1,y2, where, ax, alpha,color, args, kwargs]
    self.Data_list.append(data_i)
            
    return ln
        
        
def stem(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [],     # Labels for tittle, axis and so on.
        legend = [],     # Legend for the curves plotted
        nf = 1,          # New figure
        na = 0,          # New axis. To plot in a new axis
        # Basic parameters that we can usually find in a plot
        color = None,    # Color
        lw = 2,          # Line width
        alpha = 1.0,      # Alpha
        ws = -1,
        fontsize = 20,   # The font for the labels in the title
        fontsizeL = 10,  # The font for the labels in the legeng
        fontsizeA = 15,  # The font for the labels in the axis
        markerfmt=" ",    # The marker format   
        loc = 1
        
        ### Super Special Shit !!
       ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    self.figure_management(nf, na, labels, fontsize)

    ##################################################################
    ############### CALL PLOTTING FUNCTION ###########################
    ##################################################################
    ## TODO. Second case where NcY = NcX !!

    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1
        colorFinal = self.get_color(color)
        if (i >= len(legend)):
            markerline, stemlines, baseline = plt.stem(X,Y[:,i], lw = lw, alpha = alpha, 
                     color = colorFinal, zorder = self.zorder, markerfmt = markerfmt)

#            plt.setp(markerline, 'markerfacecolor', 'b')
#            plt.setp(baseline, 'color', 'r', 'linewidth', 2)


        else:
#            print X.shape
#            print Y[:,i].shape
            markerline, stemlines, baseline = plt.stem(X,Y[:,i], lw = lw, alpha = alpha, color = colorFinal,
                     label = legend[i], zorder = self.zorder,  markerfmt = markerfmt)
        
        plt.setp(stemlines, 'linewidth', lw)
    self.update_legend(legend,NcY,loc = loc)
    self.format_axis( nf, fontsize = fontsizeA, wsize = ws, val = NpX-ws)
    

    if (na == 1 or nf == 1):
        self.format_plot()
    
    return 0

def scatter(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [],     # Labels for tittle, axis and so on.
        legend = [],     # Legend for the curves plotted
        nf = 1,          # New figure
        na = 0,          # New axis. To plot in a new axis
        # Basic parameters that we can usually find in a plot
        color = None,    # Color
        lw = 2,          # Line width
        alpha = 1.0,     # Alpha
        ws = -1,
        fontsize = 20,   # The font for the labels in the title
        fontsizeL = 10,  # The font for the labels in the legeng
        fontsizeA = 15,  # The font for the labels in the axis
        loc = 1
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    self.figure_management(nf, na, labels, fontsize)

    ##################################################################
    ############### CALL SCATTERING FUNCTION ###########################
    ##################################################################
    
    ## We asume that X and Y have the same dimensions
    colorFinal = self.get_color(color)
    X =  ul.preprocess_dates(X)
    
    self.zorder = self.zorder + 1
    if (len(legend) == 0):
        plt.scatter(X,Y, lw = lw, alpha = alpha, 
                    color = colorFinal, zorder = self.zorder)
    else:
#        print X.shape, Y.shape
        plt.scatter(X,Y, lw = lw, alpha = alpha, color = colorFinal,
                    label = legend[0], zorder = self.zorder)
    
    self.format_axis( nf, fontsize = fontsizeA, wsize = ws, val = NpX-ws)
    self.update_legend(legend,NcY,loc = loc)

    if (na == 1 or nf == 1):
        self.format_plot()
    
    return 0


def bar(self, X = [],Y = [],  # X-Y points in the graph.
        labels = [],     # Labels for tittle, axis and so on.
        legend = [],     # Legend for the curves plotted
        nf = 1,          # New figure
        na = 0,          # New axis. To plot in a new axis
        # Basic parameters that we can usually find in a plot
        color = None,    # Color
        width = -1.0,       # Rectangle width
        bottom = None,    ## If the y-axis start somewhere else
        alpha = 1.0,     # Alpha
        despx = 0,      # Displacement in the x axis, it is done for the dates
                        # so that we can move some other things (Velero graph)
        fontsize = 20,   # The font for the labels in the title
        fontsizeL = 10,  # The font for the labels in the legeng
        fontsizeA = 15,  # The font for the labels in the axis
        ws = -1,
        loc = 1
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X_copy = copy.deepcopy(X)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
#    print X.shape, Y.shape
    # Management of the figure
    self.figure_management(nf, na, labels, fontsize)

    ##################################################################
    ############### CALL SCATTERING FUNCTION ###########################
    ##################################################################

    if (ws == -1):  # We only show the last ws samples
        ws = NpX
    if (type(bottom) == type(None)):
        bottom = np.zeros((NpX,1))
        
    plots = []
    plots_typ = []

    ## We asume that X and Y have the same dimensions
    X =  ul.preprocess_dates(X)
#    print X
    width = self.get_barwidth(X, width)
    self.zorder = self.zorder + 1
#    print self.zorder 
#    X = ul.fnp(X)  # It kills stuff
    
    for i in range(NcY):
        colorFinal = self.get_color(color)
        if (len(legend) == 0):
            plot_i= self.axes.bar(X[(NpX-ws):], Y[(NpX-ws):,[i]], width = width, align='center',
                          facecolor= colorFinal,alpha=alpha,
                          bottom = bottom[(NpX-ws):], zorder = self.zorder)
        else:
            plot_i  = self.axes.bar(X[(NpX-ws):], Y[(NpX-ws):,[i]], width = width, align='center',
                          facecolor= colorFinal,alpha=alpha,
                          label = legend[0], zorder = self.zorder,
                          bottom = bottom[(NpX-ws):])

        plots.append(plot_i)
        plots_typ.append("bar")
        
    #    print plots_typ
        ## Store pointers to variables for interaction
        self.plots_type.append(plots_typ)
        self.plots_list.append(plots) # We store the pointers to the plots
        
        data_i = [X,Y,bottom]
        self.Data_list.append(data_i)
        
    if (type(X_copy[0]) == type("D")):
        plt.xticks(X[(NpX-ws):], X_copy)
        
#    print len(X)
#    print Y.shape
#    print bottom.shape
    
    self.format_axis( nf, fontsize = fontsizeA, wsize = ws, val = NpX-ws)
    self.update_legend(legend,NcY,loc = loc)
    
    # When nf = 0 and na = 0, we lose the grid for some reason.
    if (na == 1 or nf == 1):
        self.format_plot()
    
#    print self.plots_type
    ax = self.axes
    return ax
    


def plot2(self):
    fig, ax1 = plt.subplots(figsize = [self.w,self.h])   # Get the axis and we duplicate it as desired
    self.figure = fig
    axis = [ax1]
    print self.labels
    for i in range(self.nplots):
        if (i > 0):
            axis.append(ax1.twinx())  # Create a duplicate of the axises (like new superpeusto plot)
    
        axis[i].plot(self.plot_x[i],self.plot_y[i],lw = self.lw)
        axis[i].set_xlabel('time (s)')
        axis[i].set_ylabel('exp')
        axis[i].legend (self.labels[i])
#            for tl in axis[i].get_yticklabels():
#                tl.set_color('b')
    plt.grid()
    plt.show()


#fig, ax_list = plt. subplots(3, 1)
#x = y = np.arange(5)
#
#for ax, where in zip(ax_list, ['pre', 'post', 'mid']):
#    ax.step(x, y, where=where, color='r', zorder=5, lw=5)
#    fill_between_steps(ax, x, y, 0, step_where=where)