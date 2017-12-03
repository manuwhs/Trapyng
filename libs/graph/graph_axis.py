import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import utilities_lib as ul
import matplotlib.gridspec as gridspec
import copy
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import basicMathlib as bMa
import time
from matplotlib.ticker import FuncFormatter
#####  BUILDING FUNCTIONS #####

#from matplotlib.ticker import FuncFormatter

def format_xaxis (self, ax = None, 
                  Nticks = 10,    # Number of ticks we would like
                  formatting = None,  # Specified formatting 
                  xaxis_mode = None): # Several automatic modes 

    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes

    ### Already configurated modes #####
    if (type(xaxis_mode) != type(None)):
        # If we have some profile of axis that we want
        if (xaxis_mode == "hidden"):
            return self.hide_xaxis(ax)
                 # TODO: also maybe delete the last ytick so that they do not overlap
        if (xaxis_mode == "dayly"):
            self.set_textRotations(xticks = 45)
            return self.format_xaxis(formatting = '%Y-%m-%d:%h')
        if (xaxis_mode == "intraday"):
            self.set_textRotations(xticks = 45)
            return self.format_xaxis(formatting = '%Y-%m-%d:%h')
    else:
        # If we had set ticklabels already when preprocessing the X values
        if (self.formatXaxis == "categorical"):
    #        ax.set_xticklabels(self.ticklabels[val:val + wsize])  # [1::period]
            ax.set_xticks(self.X[self.start_indx:self.end_indx], minor=False)
            ax.set_xticklabels( self.Xcategories[self.start_indx:self.end_indx][:,0], minor=False)
            # plt.xticks(self.X[val:val + wsize], self.ticklabels[val:val + wsize])
        
        elif(self.formatXaxis == "numerical"):
            # Set the number of levels in X 
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = Nticks,  prune='upper'))
    #        ax.get_xaxis().get_major_formatter().set_useOffset(False)
            
        elif(self.formatXaxis == "dates"):
            # Set the formatting of the numbers, dates or strings.
            if type(formatting) == type(None):
                formatting = '%Y-%m-%d'
            ax.xaxis.set_major_formatter(mdates.DateFormatter(formatting))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = Nticks,  prune='upper'))
            ax.xaxis_date()
          #  ax.xaxis.set_major_formatter(FuncFormatter(self.ticklabels[val:val + wsize]))
        elif(self.formatXaxis == "intraday"):
            # set the ticks of the x axis only when starting a new day
#            ndays = np.unique(np.trunc(data[:,0]), return_index=True)
#            xdays =  []
            formatter = FuncFormatter(ul.detransformer_Formatter)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = Nticks,  prune='upper'))
def format_yaxis (self, ax = None, 
                  Nticks = 10,    # Number of ticks we would like
                  formatting = None,  # Specified formatting 
                  yaxis_mode = None): # Several automatic modes 

    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes
    # If we had set ticklabels already when preprocessing the X values
    if (self.formatYaxis == "categorical"):
#        ax.set_xticklabels(self.ticklabels[val:val + wsize])  # [1::period]
        ax.set_yticks(self.Y[self.start_indx:self.end_indx], minor=False)
        ax.set_yticklabels( self.Ycategories[self.start_indx:self.end_indx][:,0], minor=False)
        # plt.xticks(self.X[val:val + wsize], self.ticklabels[val:val + wsize])
    
    elif(self.formatYaxis == "numerical"):
        # Set the number of levels in X 
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins = Nticks,  prune='upper'))
#        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        
    elif(self.formatYaxis == "dates"):
        # Set the formatting of the numbers, dates or strings.
        if type(formatting) == type(None):
            formatting = '%Y-%m-%d'
        ax.yaxis.set_major_formatter(mdates.DateFormatter(formatting))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins = Nticks,  prune='upper'))
      #  ax.xaxis.set_major_formatter(FuncFormatter(self.ticklabels[val:val + wsize]))

    ### Already configurated modes #####
    if (type(yaxis_mode) != type(None)):
        # If we have some profile of axis that we want
        if (yaxis_mode == 0):
            return self.hide_yaxis(ax)
                 # TODO: also maybe delete the last ytick so that they do not overlap
        if (yaxis_mode == 1):
            return self.format_yaxis(rotation = 45, yaxis_mode = None)
                 
def color_axis(self, ax = None, color_spines = "w", color_axis = "w"):
    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes
    ax.spines['bottom'].set_color(color_spines)
    ax.spines['top'].set_color(color_spines)
    ax.spines['left'].set_color(color_spines)
    ax.spines['right'].set_color(color_spines)
    ax.yaxis.label.set_color(color_axis)
    ax.tick_params(axis='y', colors=color_axis)
    ax.tick_params(axis='x', colors=color_axis)
    pass

def format_axis2(self,ax, Nx = 10, Ny = 5, 
                 fontsize = None, rotation = None, hideXaxis = False, 
                 X = None, val = 0, wsize = 20):
    
    # Set the ticks
    if (type(X) != type(None)):
        ax.set_xticks(self.X[val:val + wsize][0::period], minor=False)
        ax.set_xticklabels( self.ticklabels[val:val + wsize][0::period], minor=False)
                   
#    ax.get_yaxis().get_major_formatter().set_useOffset(False). It does not allow for offset in the scale, like 3000 + 0.2, 0.4

def subplots_adjust(self, hide_xaxis = True, left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0):
    # Adjusting the properties of the subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if (hide_xaxis):
        all_axes = self.get_axes()
        #Hides all of the Xaxis exept the last one.
        #TODO: the -2 is only if the last element has shared axes, we should detect that
        for i in range(len(all_axes)-2):
            ax = all_axes[i]
            self.hide_xaxis(ax)
            
def hide_xaxis(self, ax = None):
    # This function hides the axes of the funct
    if (type(ax) == type(None)):
        ax = self.get_axes()[-1]
    plt.setp(ax.get_xticklabels(), visible=False)

def hide_yaxis(self, ax = None):
    # This function hides the axes of the funct
    if (type(ax) == type(None)):
        ax = self.get_axes()[-1]
    plt.setp(ax.get_yticklabels(), visible=False)
    