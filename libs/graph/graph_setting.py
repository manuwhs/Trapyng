import matplotlib.pyplot as plt
import utilities_lib as ul
import basicMathlib as bMl
import copy
from collections import OrderedDict

from trapyngColors import cd
#####  BUILDING FUNCTIONS #####

def format_legend(self, ax = None, handlelength=None, # Length of handle
                  handletextpad= None,   # Distance between handle and text
                  borderpad=None,        # Distance between border legend and content
                  labelspacing=None,     # Horizontal spacing between labels
                  borderaxespad= 0.1,
                  columnspacing = 0.1,
                  ncol=None              # Number of columns of the legend
                  ):
                      
    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes
    ax.legend(loc='best', handlelength=handlelength, borderpad=borderpad, 
              labelspacing=labelspacing, ncol=ncol, borderaxespad = borderaxespad, columnspacing = columnspacing)

def set_fontSizes(self, ax = None, title = None, xlabel = None, ylabel = None, 
                  legend = None, xticks = None, yticks = None):
                      
    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes
    
    elif (type(ax) == type(list())):
        for ax_i in ax:
            self.set_fontSizes(ax = ax_i, title = title, xlabel = xlabel, ylabel = ylabel, 
                  legend = legend, xticks = xticks, yticks = yticks)
    else:
        # Set fontsize of the tittle
        if (type(title) != type(None)):
            ax.title.set_fontsize(fontsize=title)
            
        # Set fontsize of the axis labels
        if (type(xlabel) != type(None)):
            ax.xaxis.label.set_size( fontsize = xlabel)
        if (type(ylabel) != type(None)):
            ax.yaxis.label.set_size( fontsize = ylabel)
            
        # Set the fontsize of the ticks
        if (type(xticks) != type(None)):
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(xticks) 
        if (type(yticks) != type(None)):
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(yticks) 
        
        # Set the fontsize of the legend
        if (type(legend) != type(None)):
            ax.legend(fontsize=legend)    

def set_textRotations(self, ax = None, title = None, xlabel = None, ylabel = None, 
                  legend = None, xticks = None, yticks = None):
    
    if (type(ax) == type(None)):    # Select the axes to plot on
        ax = self.axes
    # Set the rotation
    if (type(xticks) != type(None)):
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(xticks)
            
    if (type(yticks) != type(None)):
        for label in ax.yaxis.get_ticklabels():
            label.set_rotation(yticks)
            
def set_labels(self, labels):
    # This function sets the labels of the graph when created
    # labels: If new figure, we expect 3 strings.
    # Set the main labels !!
    ax = self.axes
#    print ax
    if (len(labels) > 0):
        title = labels[0]
#        ax.title.set_text(title)
#        ax.title(title, y=1.01)
        ax.set_title(title, pad = 20)
#        plt.suptitle("frgrrg")
    if (len(labels) > 1):
        xlabel = labels[1]
        ax.set_xlabel(xlabel)
    if (len(labels) > 2):
        ylabel = labels[2]
        ax.set_ylabel(ylabel)

def update_legend(self, legend, NcY, ax = None, loc = "best"):
    # TODO: make something so that the legends do not overlap when we have shared axes.
       # If labels specified
    if (type(ax) == type(None)):
        ax = self.axes
        
    if(len(legend) > 0):
        self.legend.extend(legend)
    else:
        self.legend.extend(["Line"]*NcY)
        
    # Plot the legend
#    self.axes.legend(self.legend, loc=loc)
#    l = plt.legend()
    if(len(legend) > 0):   
        if (ax.legend()):
#            ax.legend().set_zorder(0) # Set legend on top
            ax.legend(loc=loc)
    else:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
#from matplotlib.ticke

def convert_dates_str(X):
    # We want to convert the dates into an array of char so that we can plot 
    # this shit better, and continuous

    Xdates_str = []
    for date_i in X:
        name = date_i.strftime("%Y %M %D")
        Xdates_str.append(name)
    return Xdates_str

def detect_AxisFormat(values):
    # This function automatically detects the formating that should be given
    # to the information when plotting.
#    print (type(values))
#    print (values.shape)
    V_type = type(values[0,0]).__name__ 
    
#    print (V_type)
    if ( V_type == "str" or V_type == "string_" or  V_type == 'numpy.string_' or  V_type =="str_"):
        V_format = "categorical"
        
    elif(V_type == "datetime64" or V_type == "Timestamp" or  V_type == "datetime"):
        V_format = "dates"
        
    else:
        V_format = "numerical"

    return V_format

    
def preprocess_data(self,X,Y, dataTransform = None ):
   # Preprocess the variables X and Y
   ### First we transform everything to python format
    X = ul.fnp(X)
    Y = ul.fnp(Y)
    # Whe can plot several Y over same X. So NcY >= 1, NcX = 0,1
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    if (X.size == 0):  # If the X given is empty
        if (Y.size == 0):          # If we are also given an empty Y
            Y = ul.fnp([])
        else:  
            X = ul.fnp(range(NpY)) # Create X axis with sample values
            
    self.X = X; self.Y = Y
    
# Get the automatic formatting of the X and Y axes !! 

    # If we are given values to X, these could be of 3 types:
    # - Numerical: Then we do nothing 
    # - String: Then it is categorical and we set the ticklabels to it
    # - Datetimes: We set the formatter ?
    self.formatXaxis = detect_AxisFormat(X)
    self.formatYaxis = detect_AxisFormat(Y)
#        print X,Y, self.formatXaxis, self.formatYaxis
    if (type(dataTransform) != type(None)):
        if (dataTransform[0] == "intraday"):
            # In this case we are going to need to transform the dates.
            openhour = dataTransform[1] 
            closehour = dataTransform[2]
            self.formatXaxis = "intraday"
            # Virtual transformation of the dates !
            self.Xcategories = self.X
            
            transfomedTimes = ul.transformDatesOpenHours(X,openhour, closehour )
            Mydetransfromdata = ul.deformatter_data(openhour, closehour, None)
            # Setting the static object of the function
            ul.detransformer_Formatter.format_data = Mydetransfromdata
            self.X = ul.fnp(transfomedTimes) 
            
    if (self.formatXaxis == "categorical"): # Transform to numbers and later we retransform
        self.Xcategories = self.X
        self.X = ul.fnp(range(NpX)) 
    
    if (self.formatYaxis == "categorical"): # Transform to numbers and later we retransform
        self.Ycategories = self.Y
        self.Y = ul.fnp(range(NpY)) 
    
#    if (self.formatXaxis == "dates"): # Transform to numbers and later we retransform
#        self.Xcategories = self.X
#        self.X = ul.preprocess_dates(range(NpX)) 
        
    return self.X,self.Y


    
def get_color(self, color = None):
    # This function outputs the final color to print for a given
    # plotting

    if (type(color) == type(None)):
        # If no color specified. We use one of the list
        colorFinal = self.colors[self.colorIndex]
        self.colorIndex = (self.colorIndex + 1) %len(self.colors)
        
    elif(type(color) == type([])):
        colorFinal = color
    else:
        if(color in cd.keys()):
            colorFinal = cd[color]
        else:
            colorFinal = color
    return colorFinal
    
def add_text(self, positionXY = [], text = r'an equation: $E=mc^2$',fontsize = 15, ax = None):
    if (type(ax) == type(None)):
        ax = self.axes
    ## PositonXY should be given in termns of the X and Y axis variables
    if (len(positionXY) == 0):
        positionXY = [0,0]
        
    return self.axes.text(positionXY[0], positionXY[1], text, fontsize=fontsize)


def get_barwidth(self,X, width = None):
    # The Xaxis could be dates and so on, so we want to calculate
    # the with of this bastard independently of that

    if (len(X.shape)):
        X = X.flatten()
    print (X.shape)
    print("X axis type: ", type(X[0]).__name__ )
    if (type(width) == type(None)):
        width = 1
#        print width
    if (type(X[0]).__name__ == "Timestamp" ): #  or (type(X[0]).__name__ == "datetime64" )
        width_size = min(bMl.diff(X)[1:])
        width_size = (width_size.total_seconds())/ (24.0*60*60) 
    else:
        
        width_size = min(bMl.diff(X)[1:]) # (X[1] - X[0]) 
#        print type(X[0,0])
#        print X.shape
#        width_size = min(bMa.diff(X, cval = 10000))
#        width_size = (width_size.total_seconds())/ (24.0*60*60) 
    width = width_size * width
#    print type(X[0])
    width = float(width)
    print("width is: ", width)
    return width
    

def store_WidgetData(self, plots_typ, plots):
    # This funciton will store the data needed to later use the widgets
    self.plots_type.append(plots_typ)
    self.plots_list.append(plots) # We store the pointers to the plots
    data_i = [copy.deepcopy(self.X),copy.deepcopy(self.Y)]
    self.Data_list.append(data_i)
    
def init_WidgetData(self, initX = None, ws =  None):
    ## TODO. Second case where NcY = NcX !!
    if (type(initX) == type(None)):
        initX = 0
    if (type(ws) == type(None)):  # We only show the last ws samples
        NpX = self.Y.shape[0]
        ws = NpX - initX
    self.ws = ws
    self.start_indx = initX
    self.end_indx = initX + ws
    
    plots = []
    plots_typ = []
    
    return plots,plots_typ