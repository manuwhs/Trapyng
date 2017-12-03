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
#####  BUILDING FUNCTIONS #####

def subplot2grid(self, *args, **kwargs): #divisions, selection):
    # Usar **kwargs to get the dictionary ? 
    # Same as the original :) ! But we put the new axis into the array
    # It creates an axes with the desired dimensions by dividing 
    # the axes in "divisions" and then subselecting the desired ones.
    if (self.fig == None):       # If there is no figure
        self.init_figure()
    ax = plt.subplot2grid(*args,**kwargs)
    self.axes = ax
    self.axes_list.append(ax)
    return ax
    
def set_subplots(self, nr, nc, projection = "2d", sharex=False):
    self.init_figure()
    # State a subplot partitition of a new figure.
    # nr is the number of rows of the partition
    # nc is the numbel of columns
    
    self.sharex_aux = sharex
    self.subplotting = 1  
    # Variable that indicates that we are subplotting
    # So when nf = 1, if this variable is 0, then 
    # we do not create a new figure but we paint in a
    # different window 
    
    self.nc = nc
    self.nr = nr
    
    
    if (self.subplotting_mode == 1):
        # Using gridspec library
        self.G = gridspec.GridSpec(nr, nc)
        
    elif(self.subplotting_mode == 0):
        ax = plt.subplot2grid((nr,nc),(0, 0))
        
    self.nci = 0
    self.nri = 0
    
    # We can select the subplot with  plt.subplot(G[r, c])

    self.first_subplot = 1

    ## Set the space of the things
#    self.G.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

#    print self.subplotting

def next_subplot(self, projection = "2d", sharex = None, sharey = None):
    # Moves to the next subpot to plot.
    # We move from left to right and up down.

    if (self.first_subplot == 1): # If it is the first plot
        # then we do not set the new subplot due to nf = 1
        # This way all the subplottings are the same
        # Select first plot.
        self.first_subplot = 0
    else:
        self.nci = (self.nci + 1) % self.nc
        if (self.nci == 0): # If we started new row
            self.nri = (self.nri + 1) % self.nr
            
        if (self.nri == (self.nr-1) and self.nci == (self.nc-1)): # If we are in the last graph 
            self.subplotting = 0

    if (self.subplotting_mode == 1):
        if (projection == "2d"):
            ax = plt.subplot(self.G[self.nri, self.nci], sharex = sharex, sharey = sharey )
        elif(projection == "3d"):
            ax = plt.subplot(self.G[self.nri, self.nci], projection='3d',  sharex = sharex, sharey = sharey  )
        elif (projection == "polar"):
            ax = plt.subplot(self.G[self.nri, self.nci],projection='polar',  sharex = sharex, sharey = sharey )
        
#        ax = plt.axes(position = position )
        # YEAH !!! Fucking axes does not work !!
#            position = ax.get_position()
#            ax.axis('off')
##            print position
#            ax = self.fig.add_axes(position, projection = projection)
            print "subplot"
            
    elif(self.subplotting_mode == 0):
        if (projection == "2d"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), sharex = sharex , sharey = sharey )
        elif(projection == "3d"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), projection='3d', sharex = sharex , sharey = sharey )
        elif(projection == "polar"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), projection= 'polar', sharex = sharex , sharey = sharey )
            print "subplot2grid"
    if (self.nci + self.nri == 0):
        plt.tight_layout()  # So that the layout is tight

    self.axes = ax
    self.axes_list.append(ax)
    
def subplots_adjust(self, hide_xaxis = False, left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0):
    # Adjusting the properties of the subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if (hide_xaxis):
        all_axes = self.get_axes()
        #Hides all of the Xaxis exept the last one.
        #TODO: the -2 is only if the last element has shared axes, we should detect that
        for i in range(len(all_axes)-2):
            ax = all_axes[i]
            self.hide_xaxis(ax)
    
def apply_style(self, nf,na, AxesStyle = None):
    # This function applies standard specfied formattings :)

    self.axes.grid()
    
    ax = self.axes
    # TODO, should it be if nf == 1
    if (type(AxesStyle) != type(None)):
        options = AxesStyle.split(" - ")
        style = options[0]
        if (style == "Normal"):
            self.set_fontSizes(title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 11, yticks = 13)
            self.set_textRotations(xticks = 60)
            self.color_axis(color_spines = "k", color_axis = "k")
            self.format_xaxis (Nticks = 20,formatting = None)
            self.format_legend(handlelength=1.5, borderpad=0.5,labelspacing=0.3, ncol = 2)
            
            if (type( self.axes.get_legend()) != type(None)):     
                self.axes.get_legend().get_title().set_fontsize(25)
#            self.axes.legend(fontsize=25)    
        elif (style == "Normal2"):
            self.set_fontSizes(title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 15, yticks = 15)
            self.format_xaxis (Nticks = 10,formatting = None)
            self.format_legend(handlelength=1.5, borderpad=0.5,labelspacing=0.3, ncol = 2)
            if (type( self.axes.get_legend()) != type(None)):     
                self.axes.get_legend().get_title().set_fontsize(25)
            
            ax.axhline(linewidth=1.7, color="black",marker = ">",ms = 6)
            ax.axhline(linewidth=1.7, color="black",marker = "<")
            ax.axvline(linewidth=1.7, color="black",marker = "^")
            ax.axvline(linewidth=1.7, color="black",marker = "v")
            if (0):
                # A way to draw the axes
                ax.spines['left'].set_position('zero')
                ax.spines['right'].set_color('none')
                ax.spines['bottom'].set_position('zero')
                ax.spines['top'].set_color('none')
                # remove the ticks from the top and right edges
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

        if (len(options) > 1):
            for i in range (1,len(options)):
                otherOption = options[i]
                if (otherOption == "No xaxis"):
                    self.hide_xaxis()
                elif (otherOption == "No yaxis"):
                    self.hide_yaxis()
                
                # See if it tries to set any other param
                suboptions = options[i].split(":")
                if (suboptions[0] == "Ny"):
                    self.format_yaxis (Nticks = int(suboptions[1]),formatting = None)
    plt.show()
