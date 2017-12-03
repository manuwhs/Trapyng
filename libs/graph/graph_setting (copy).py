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

def init_variables(self,w = 20, h = 12, lw = 2):
        self.w = w;    # X-width
        self.h = h;    # Y-height
        self.lw = lw   # line width
        
        self.prev_fig = []  # List that contains the previous plot.
        # When a new figure is done, we put the current one here and
        # create a new variable. Silly way to do it though
        
        self.fig = None
        self.axes = None
        
        self.nplots = 0;  # Number of plots
        self.labels = []  # Labels for the plots
        self.plot_y = []  # Vectors of values to plot x-axis
        self.plot_x = []  # Vectors of values to plot y-axis
        
        # Set of nice colors to iterate over when we do not specify color
        self.colors = ["b", "g","k", "r", "c", "m","y"] 
        self.colorIndex = 0;  # Index of the color we are using.
        self.X = ul.fnp([])
        self.legend = []
        
        self.subplotting_mode = 1 # Which functions to use for subplotting
        self.subplotting = 0;  # In the beggining we are not subplotting
        self.ticklabels = []   # To save the ticklabels in case we have str in X
        
        self.Xticklabels = []  # For 3D
        self.Yticklabels = []
        self.zorder = 1   # Zorder for plotting
        
        ## Store enough data to redraw and reference the plots for the interactivity
        self.plots_list = []; # List of the plots elements, 
        self.plots_type = []; # Type of plot of every subplot
        
        self.axes_list = []   # We store the list of indexes
        self.Data_list = []  # We need to store a pointer of the data in the graph
        self.widget_list = []  # We need to store reference to widget so that
        self.num_hidders = 0
        # TODO Data_list is different than plots_list, data list containes only one pos with all the Y, the other does it separately
        # it does not dissapear.
        # A plot in plot_list can be composed by different plots if when the functions where called
        # more signals where given.)

def init_figure(self, projection = "2d", position = [], subplotting = 0):
    # This function initializes the data structures of the new figure
#    print "FR"
    # Copy the previous self if any and reinit variables

#    the_class = self.__class__
#    a2 = k2()

    # If we are really creating a new figure
    self.prev_fig.append(copy.copy(self))
    ## Reinit everything
    self.init_variables(self.w, self.h, self.lw)

    fig = plt.figure()  
    self.fig = fig
    
#    fig.set_facecolor("w")

def figure_management(self, nf, na, labels, fontsize, ax = None, projection = "2d", position = []):
    # This function si suposed to deal with everything that
    # has to do with initializating figure, axis, subplots...

    if (nf == 1):   
        self.colorIndex = 0 # Restart colors again
        # If we want to create a new figure
        if (self.subplotting == 1):
            # If we are subploting so it will be plotted into another axes
            self.next_subplot(projection = projection) # We plot in a new subplot !
        else:
            self.init_figure(projection = projection)  # We create a new figure !!
        
    # If we want to create a new axis
    ax = self.manage_axes(na = na, position = position, ax = ax, projection = projection)
    self.set_labels(labels, fontsize)
    return ax

""" $$$$$$$$$$$$$$$ AXES FUNCTIONS $$$$$$$$$$ """
def twin_axes(self, ax = None):
    # Creates a twin axes and puts it in the end
    if (ax == None):
        ax = self.axes
        
    ax = ax.twinx()  # Create a twin axis
    self.axes = ax
    self.axes_list.append(ax)
    
    return ax
    
def create_axes(self, position = [], projection = "2d"):
    # Get the axes of the plot. 
    # We might want to plot different axes sometimes.

    if (self.fig == None):       # If there is no figure
        self.init_figure()
        
    if (projection == "2d"):
        if (len(position) == 0):
            position = [0.1, 0.1, 0.8, 0.8]
        
#        ax = plt.axes(position = position )
        # YEAH !!! Fucking axes does not work !!
        ax = self.fig.add_axes(position)
        # TODO If the position is the same, then,
        # the x-axis is merged for some reason
#        print "fR"
        # We are defining 40x40 subdivisions and filling them all
#        self.axes = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40, projection = projection)  
    elif (projection == "3d"): # No really need, since the 3D func create axis anyway
        ax = plt.axes(projection='3d')  # Or plt.plot
#        print "grgr"
    elif ( projection == "polar"):
        position = [0.1, 0.1, 0.8, 0.8]
        
#        ax = plt.axes(position = position )
        # YEAH !!! Fucking axes does not work !!
        ax = self.fig.add_axes(position, projection = projection)

    else:
        print "No valid projection"
        
    self.axes = ax
    self.axes_list.append(ax)
    return ax

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
    
def get_axes(self):
    return self.axes_list
    
def manage_axes(self, na = 0, ax = None, position = [], projection = "2d"):
    # This function manages the creation of the new axes
    # Or the reusing of the previous one.
   
   # If we indicated an axes, we just plot on it
    if (type(ax) != type(None)):    
        return ax
    ## If we do not have an axes yet
    if (type(self.axes) == type(None)):
        ax  = self.create_axes(position = position, projection = projection)# Self.axes is the last axes we do stuff in

    else:
        # If we already have the axes
        if (na == 0):
            # If we do not want a new axes we do nothing
            ax = self.axes
#            print "PEEFEFE"
        else:
            # If we want a new axes
            if (len(position) == 0):
                # If we do not indicate the position, we use the same
                ax = self.twin_axes()
            else:
                ax  = self.create_axes(position = position, projection = projection)
    return ax
#            ax = plt.axes(position = position )
#            ax.set_visible(True)
#            print ax  
#            ax.set_position(position)
#            ax.set_axis_bgcolor("w")
#            ax.patch.set_alpha(0.0)
#            # This works because patch is just the figure (the Artist in matplotlib parlance) 
#            #  that represents the background as I understand from the documentation:
#            ax.patch.set_facecolor('w')
#            print position
#        
#            print position
#            print position
#            plt.show()
#            time.sleep(5)
#            if (len(position) != 0): 
#                ax.set_position(position, which='both')
                
#                print ax.get_position()
#            print ax
                
#    print len(self.axes_list)
#    print self.axes_list
#    # Function to change position of axes !! TODO
#    pos_prev =  ax.get_position()
#    print pos_prev
#    if (len(position) != 0):
#        ax.set_position(position, which='both')
#    # pos = [left, bottom, width, height]

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

def next_subplot(self, projection = "2d"):
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

    
    # Select next plot.
    axis_sharing = None
    if (self.sharex_aux == True):
        if (len(self.get_axes()) > 0):
            axis_sharing = self.get_axes()[-1]
    if (self.subplotting_mode == 1):

        if (projection == "2d"):
            ax = plt.subplot(self.G[self.nri, self.nci], sharex = axis_sharing )
        elif(projection == "3d"):
            ax = plt.subplot(self.G[self.nri, self.nci], projection='3d', sharex = axis_sharing )
        elif (projection == "polar"):
            ax = plt.subplot(self.G[self.nri, self.nci],projection='polar', sharex = axis_sharing )
        
#        ax = plt.axes(position = position )
        # YEAH !!! Fucking axes does not work !!
#            position = ax.get_position()
#            ax.axis('off')
##            print position
#            ax = self.fig.add_axes(position, projection = projection)
            print "subplot"
            
    elif(self.subplotting_mode == 0):
        if (projection == "2d"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), sharex =axis_sharing )
        elif(projection == "3d"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), projection='3d', sharex = axis_sharing )
        elif(projection == "polar"):
            ax = plt.subplot2grid((self.nr,self.nc),(self.nri, self.nci), projection= 'polar', sharex = axis_sharing )
            print "subplot2grid"
    if (self.nci + self.nri == 0):
        plt.tight_layout()  # So that the layout is tight

    self.axes = ax
    self.axes_list.append(ax)
    
def set_labels(self, labels, fontsize, loc = 1):
    # This function sets the labels of the graph when created
    # labels: If new figure, we expect 3 strings.
    # Set the main labels !!
    ax = self.axes
    if (len(labels) > 0):
        title = labels[0]
        ax.title.set_text(title)
        
    if (len(labels) > 1):
        xlabel = labels[1]
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if (len(labels) > 2):
        ylabel = labels[2]
        ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.title.set_fontsize(fontsize=fontsize)

def update_legend(self, legend, NcY, loc = "best"):
    # TODO: make something so that the legends do not overlap when we have shared axes.
       # If labels specified
    ax = self.axes
    if(len(legend) > 0):
        self.legend.extend(legend)
    else:
        self.legend.extend(["Line"]*NcY)
    # Plot the legend
#    self.axes.legend(self.legend, loc=loc)
#    l = plt.legend()
        

    if (ax.legend()):
        ax.legend().set_zorder(100000) # Set legend on top
        ax.legend(loc=loc)

def set_xlim(self, xmin = -1, xmax = -1):
    # This function sets the limits for viewing the x coordinate
    ax = self.axes
    
    if (xmin == -1):
        xmin = self.X[0,0]
    if (xmax == -1):
        xmin = self.X[-1,0]
        
    ax.set_xlim([xmin,xmax])

def set_ylim(self, ymin = -1, ymax = -1, ax = -1):
    # This function sets the limits for viewing the x coordinate
    
    if (ax == -1):
        # Select the axes to limit
        ax = self.axes
    
    if (ymin == -1):
        ymin = min(self.Y)
    if (ymax == -1):
        ymin = max(self.Y)
        
    ax.set_ylim([ymin,ymax])


#from matplotlib.ticker import FuncFormatter
def format_axis (self, nf = 0, fontsize = -1 , period = 10,  val = 0, wsize = -1):
    ## Labels the X axis bithch !! 
    ## val and wsize represent the proportion we are painting
    # period is how many label we put period
    
    ax = self.axes
    if (wsize == -1):
        wsize = self.X.shape[0]

    if (len(self.ticklabels) > 0):
#        print "FE"
#        ax.set_xticklabels(self.ticklabels[val:val + wsize])  # [1::period]
#        max_yticks = 20
#        yloc = plt.MaxNLocator(max_yticks)
#        ax.xaxis.set_major_locator(yloc)
        
#        ax.xaxis.set_major_formatter(FuncFormatter(self.ticklabels[val:val + wsize]))

#        plt.xticks(self.X[val:val + wsize], self.ticklabels[val:val + wsize])
        ax.set_xticks(self.X[val:val + wsize][0::period], minor=False)
        ax.set_xticklabels( self.ticklabels[val:val + wsize][0::period], minor=False)

#        self.ticklabels = []  # Delete them. We only keep the last ones ?

    ## Also, rotate the labels
    if (nf == 1):
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
#    plt.subplots_adjust(bottom=.15)  # Done to view the dates nicely

    if (fontsize != -1):
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)  
            
def color_axis(self, ax, color_spines = "w", color_axis = "w"):
    ax.spines['bottom'].set_color(color_spines)
    ax.spines['top'].set_color(color_spines)
    ax.spines['left'].set_color(color_spines)
    ax.spines['right'].set_color(color_spines)
    ax.yaxis.label.set_color(color_axis)
    ax.tick_params(axis='y', colors=color_axis)
    ax.tick_params(axis='x', colors=color_axis)
    pass
def format_axis2(self,ax, Nx = 10, Ny = 5, fontsize = -1, rotation = 45):
    ax.xaxis.set_major_locator(mticker.MaxNLocator(Nx))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins = Ny, prune='upper'))  # prune='upper'   Ny
    
    if (fontsize != -1):
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)   
            
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(rotation)
#    ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
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
        ax = self.get_axes[-1]

    plt.setp(ax.get_xticklabels(), visible=False)
    
def format_plot(self):
    plt.grid()
    plt.show()
    pass

def convert_dates_str(X):
    # We want to convert the dates into an array of char so that we can plot 
    # this shit better, and continuous

    Xdates_str = []
    for date_i in X:
        name = date_i.strftime("%Y %M %D")
        Xdates_str.append(name)
    return Xdates_str
    
def preprocess_data(self,X,Y):
   # Preprocess the variables X and Y
   ### X axis date properties, just to be seen clearly
   ### First we transform everything to python format
    X = ul.fnp(X)
    Y = ul.fnp(Y)
#    print Y.shape
    # Each plot can perform several plotting on the same X axis
    # So there could be more than one NcX and NcY
    # NpX and NpY must be the same.
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # If the type are labels 
    
    if (X.size > 0):
        if (type(X[0,0]).__name__ == "str" or type(X[0,0]).__name__ == "string_"):
            self.ticklabels = X.T.tolist()[0]
            X = ul.fnp([])

    # If we have been given an empty X
    if (X.size == 0):
        # If we are also given an empty Y
        if (Y.size == 0): 
            return -1 # We get out.
            # This could be used to create an empty figure if
            # fig = 1
        # If we wanted to plot something but we did not specify X.
        # We use the previous X, if it does not exist, we use range()
        else:  
            X = ul.fnp(range(NpY)) # Create X axis
                
    # The given X becomes the new axis
    self.X = X    
    self.Y = Y
    return X,Y
    
def get_color(self, color):
    # This function outputs the final color to print for a given
    # plotting

    if (type(color).__name__ == "NoneType"):
        # If no color specified. We use one of the list
        colorFinal = self.colors[self.colorIndex]
        self.colorIndex = (self.colorIndex + 1) %len(self.colors)
    else:
        colorFinal = color
    return colorFinal
    
def add_text(self, positionXY = [], text = r'an equation: $E=mc^2$',fontsize = 15):
    ax = self.axes
    ## PositonXY should be given in termns of the X and Y axis variables
    if (len(positionXY) == 0):
        positionXY = [0,0]
        
    self.axes.text(positionXY[0], positionXY[1], text, fontsize=fontsize)


def get_barwidth(self,X, width):
    # The Xaxis could be dates and so on, so we want to calculate
    # the with of this bastard independently of that

    if (width < 0):
        width = 1
#        print width
    if (type(X[0]).__name__ == "Timestamp"):
        width_size = min(ul.diff(X))
        
        width_size = (width_size.total_seconds())/ (24.0*60*60) 
    else:
        width_size = (X[1] - X[0]) 
        width_size = min(bMa.diff(X, cval = 10000))
        
    width = width_size * width
#    print type(X[0])
    width = float(width)
    
    return width
    
def savefig(self,file_dir = "./image.png", 
            bbox_inches = 'tight',
            sizeInches = [],  # The size in inches as a list
            close = False,   # If we close the figure once saved
            dpi = 100):      # Density of pixels !! Same image but more cuality ! Pixels
    ## Function to save the current figure in the desired format
    ## Important !! Both dpi and sizeInches affect the number of pixels of the image
    # dpi: It is just for quality, same graph but more quality.
    # sizeInches: You change the proportions of the window. The fontsize and 
    # thickness of lines is the same. So as the graph grows, they look smaller.
    F = self.fig  # ?? NO work ? 
#    F = pylab.gcf()

    Winches, Hinches = F.get_size_inches()  # 8,6
    
#    print Winches, Hinches 
    
    if (len(sizeInches) > 0):
        F.set_size_inches( (sizeInches[0], sizeInches[1]) )
    
    self.fig.savefig(file_dir,
            bbox_inches='tight',
            dpi = dpi
            )
    # Transform back
    F.set_size_inches((Winches, Hinches) )  

    if (close == True):
        plt.close()
        
#    gl.savefig('foodpi50.png', bbox_inches='tight',  dpi = 50)
#    gl.savefig('foodpi100.png', bbox_inches='tight',  dpi = 100)
#    gl.savefig('foodpi150.png', bbox_inches='tight',  dpi = 150)
#
#    gl.savefig('foosize1.png', bbox_inches='tight',  sizeInches = [3,4])
#    gl.savefig('foosize2.png', bbox_inches='tight',  sizeInches = [6,8])
#    gl.savefig('foosize3.png', bbox_inches='tight',  sizeInches = [8,11])
#    gl.savefig('foosize4.png', bbox_inches='tight',  sizeInches = [10,14])

#
#import matplotlib.pyplot as plt
#import numpy as np
#
## create some data to use for the plot
#dta = 0.001
#t = np.arange(0.0, 10.0, dta)
#r = np.exp(-t[:1000]/0.05)               # impulse response
#x = np.random.randn(len(t))
#s = np.convolve(x, r)[:len(x)]*dta  # colored noise
#
## the main axes is subplot(111) by default
#plt.plot(t, s)
#plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
#plt.xlabel('time (s)')
#plt.ylabel('current (nA)')
#plt.title('Gaussian colored noise')
#
## this is an inset axes over the main axes
#a = plt.axes([.65, .6, .2, .2], axisbg='y')
#n, bins, patches = plt.hist(s, 400, normed=1)
#plt.title('Probability')
#plt.xticks([])
#plt.yticks([])
#
## this is another inset axes over the main axes
#a = plt.axes([0.2, 0.6, .2, .2], axisbg='y')
#plt.plot(t[:len(r)], r)
#plt.title('Impulse response')
#plt.xlim(0, 0.2)
#plt.xticks([])
#plt.yticks([])
#
#plt.show()