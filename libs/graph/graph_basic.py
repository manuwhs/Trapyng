import matplotlib.pyplot as plt
import utilities_lib as ul
import copy
from trapyngColors import cd
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

#        self.colors = ["k","b", "g", "r", "c", "m","y"] 
        self.colors = [cd["dark navy blue"],cd["golden rod"],
                       cd["blood"], cd["chocolate"],  cd["cobalt blue"],
                 cd["cement"], cd["amber"], cd["dark olive green"]]
        
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
    
    return fig
#    fig.set_facecolor("w")

def figure_management(self, nf, na, ax = None, sharex = None, sharey = None, 
                      projection = "2d", position = []):
    # This function si suposed to deal with everything that
    # has to do with initializating figure, axis, subplots...
#    if (self.fig == None):
#        self.init_figure()
    
    # If we want a new figure or we advance to the next subplotting.
    if (nf == 1):   
        self.colorIndex = 0 # Restart colors again
        # If we want to create a new figure
        if (self.subplotting == 1):
           # print ("XXXXXXXXXXXXXXXXX")
            # If we are subploting so it will be plotted into another axes
            ax = self.next_subplot(projection = projection, sharex = sharex, sharey = sharey) # We plot in a new subplot !
        else:
            self.init_figure(projection = projection)  # We create a new figure !!
        
    # If we want to create a new axis and how
    ax = self.manage_axes(na = na, position = position,
                          ax = ax, projection = projection)
    return ax

def close(self,  *args, **kwargs):
    return  plt.close( *args, **kwargs)
    
def savefig(self,file_dir = "./image.png", 
            bbox_inches = 'tight',
            sizeInches = [],  # The size in inches as a list
            close = False,   # If we close the figure once saved
            dpi = 100):      # Density of pixels !! Same image but more cuality ! Pixels
    
    try:
        folders = file_dir.split("/")
        folders.pop(-1)
        path = "/".join(folders)
        ul.create_folder_if_needed(path);
    except:
        pass
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
            bbox_inches = bbox_inches,
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



