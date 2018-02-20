import numpy as np
import matplotlib.pyplot as plt

#####  BUILDING FUNCTIONS #####

def twin_axes(self, ax = None):
    # Creates a twin axes and puts it in the end
    if (type(ax) == type(None)):
        ax = self.axes
        
    ax = ax.twinx()  # Create a twin axis
    self.axes = ax
    self.axes_list.append(ax)
    
    return ax
    
def create_axes(self, position = [], projection = "2d",
                sharex = None, sharey = None):
    # Get the axes of the plot. 
    # We might want to plot different axes sometimes.
#    print sharex
    self.colorIndex
    
    if (type(sharex) == type(True)):
        sharex = self.axes
    if (type(sharey) == type(True)):
        sharey = self.axes
        
    if (self.fig == None):       # If there is no figure
        self.init_figure()
        
    if (projection == "2d"):
        if (len(position) == 0):
            position = [0.1, 0.1, 0.8, 0.8]
        
#        ax = plt.axes(position = position )
        # YEAH !!! Fucking axes does not work !!
        ax = self.fig.add_axes(position, sharex, sharey)
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
        print ("No valid projection")
        
    self.axes = ax
    self.axes_list.append(ax)
    return ax

def get_axes(self):
    return self.axes_list
    
def manage_axes(self, na = 0, 
                ax = None, position = [], 
                sharex = None, sharey = None, projection = "2d"):
                    
    # This function manages the creation of the new axes
    # Or the reusing of the previous one.
   
   # If we indicated an axes, we just plot on it
    if (type(ax) != type(None)):    
        self.axes = ax
        self.axes_list.append(ax)
#        print "F"
        return ax
    ## If we do not have an axes yet
    if (type(self.axes) == type(None)):
        ax  = self.create_axes(position = position, projection = projection,
                               sharex = sharex, sharey = sharey)# Self.axes is the last axes we do stuff in

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
                ax  = self.create_axes(position = position, projection = projection,
                                       sharex = sharex, sharey = sharey)
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
                


# Advanced settings for the zoom
def set_zoom(self, ax = None, xlim = None ,X =None, Y = None, ylim = None, xlimPad = None ,ylimPad = None):
    if (type(ax) == type(None)):
        ax = self.axes
    if (type(Y) == type(None)):
        Y = self.Y[self.start_indx:self.end_indx]
    if (type(X) == type(None)):
        X = self.X[self.start_indx:self.end_indx]
        
    # Set the padding relative to the maximum
    if (type(ylimPad) != type(None)):

        max_signal = np.max(Y[~np.isnan(Y)])
        min_signal = np.min(Y[~np.isnan(Y)])
#        min_signal = np.min([min_signal,0])
        signal_range = max_signal - min_signal
        if (signal_range == 0):
            signal_range = max_signal
            if ( signal_range > 0):
                min_signal = 0
            else:
                max_signal = 0
        self.set_ylim(ax = ax, ymin = min_signal - signal_range* ylimPad[0] , ymax = max_signal + signal_range*ylimPad[1])
    elif (type(ylim) != type(None)):
        self.set_ylim(ax = ax, ymin = ylim[0], ymax = ylim[1])
        
    if (type(xlimPad) != type(None)):
#        print type(self.X.flatten()[0])
#        max_signal = np.max(self.X[~np.isnan(self.X.flatten())])
#        min_signal = np.min(self.X[~np.isnan(self.X)])
        max_signal = np.max(X[~np.isnan(X)])
        min_signal = np.min(X[~np.isnan(X)])
#        min_signal = np.min([min_signal,0])
        signal_range = max_signal - min_signal
        if (signal_range == 0):
            signal_range = max_signal
            if ( signal_range > 0):
                min_signal = 0
            else:
                max_signal = 0
        self.set_xlim(ax = ax, xmin = min_signal - signal_range* xlimPad[0] ,xmax = max_signal + signal_range*xlimPad[1])

    elif (type(xlim) != type(None)):
        self.set_xlim(ax = ax, xmin = xlim[0], xmax =xlim[1])
 
def set_xlim(self, ax = None, X = None, xmin = None, xmax = None):
    # This function sets the limits for viewing the x coordinate
    if (type(ax) == type(None)):
        ax = self.axes
    if (type(X) == type(None)):
        X = self.X[self.start_indx:self.end_indx]
        
    if (type(xmin) == type(None)):
        xmin = np.min(X[~np.isnan(X)])
    if (type(xmax) == type(None)):
        xmax = np.max(X[~np.isnan(X)])
        
        
    ax.set_xlim([xmin,xmax])

def set_ylim(self, ax = None, Y = None, ymin = None, ymax = None):
    # This function sets the limits for viewing the x coordinate
    
    if (type(Y) == type(None)):
        Y = self.Y[self.start_indx:self.end_indx]
        
    if (type(ax) == type(None)):
        ax = self.axes
    
    if (type(ymin) == type(None)):
        ymin = np.min(Y[~np.isnan(Y)])
    if (type(ymax) == type(None)):
        ymax = np.max(Y[~np.isnan(Y)])

        
    ax.set_ylim([ymin,ymax])



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