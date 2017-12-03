import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import utilities_lib as ul
import matplotlib.gridspec as gridspec
import graph_basic as grba
#import tasks_lib as tal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons,Slider, Button, SpanSelector
from matplotlib.widgets import AxesWidget
import sys
from matplotlib.finance import candlestick_ohlc

def add_hidebox(self, plots_affected = [], 
               names = [],
               activation = [],   # Initial activation
               func = "timeSlide",
               args = {}):
                   
    ## Number of hidders to put.
    if (len(plots_affected) == 0):
        plots_affected = range(len(self.plots_list))
    
    nh = len(plots_affected)
    ## Inner variables to dynamically manage the adding of this
    self.num_hidders += nh
    
    rax = plt.axes([0.05, 0.4, 0.1, 0.05 * nh])
    
    if (len(names) == 0):
        for i in range(nh):
            names.append("Signal %i"% (i+1))
    if (len(activation) == 0):
        for i in range(nh):
            activation.append(True)
    
    print (names)
    print (activation)
    
    check = CheckButtons(rax, names, activation)

    def hide(label):
        plots_affected = hide.plots_affected
        signal_names = hide.signal_names
        for plot_i in plots_affected:
            lns = self.plots_list[plot_i]  # The reference to the plot
            # We might have more than one signal plotter though
            # We change the Y data shown
#            print "In bitch 2"
#            print self.plots_type[plot_i]
            if (label == signal_names[plot_i]):
                
                for i in range(len(lns)):
    
                    # Type of plot will change the way to treat it
                    ln_type = self.plots_type[plot_i][i]
                    if (ln_type == "plot"):
                        ln = lns[i]
                        ln.set_visible(not ln.get_visible())
                    elif (ln_type == "fill"):
                        ln = lns[i]
                        ln.set_visible(not ln.get_visible())
                    elif (ln_type == "bar"):
                        ln = lns[i]
                        for rect in ln:
                            rect.align = "center"
    #                        print self.Data_list[plot_i][i]
                            rect.set_visible(not rect.get_visible())
        plt.draw()
        
    ## We make it a local variable so that it does not disapear
    hide.plots_affected = plots_affected
    hide.signal_names = names
    
    check.on_clicked(hide)
    self.widget_list.append(check)
    
def add_slider(self, plots_affected = [], 
               name = "slidy",
               func = "timeSlide",
               args = {}):
    ## This function adds a slider related to one of the plots
    ## Indicated by the index
    # plot_i = Indexes of the plot associated to the slider.
    # func = Set of functions that we want the slider to do.
    # Since we need to referenced inner things, we migh as well define the
    # updating functions inside this funciton.
             
    
    ## Shirnk the main axes. TODO, move this to another general func
#    plt.subplots_adjust(left = 0.2, bottom=0.2)

    ## Limits of value of the slider.
    ## If nothig specified, they are the index of the selected plot
    
    if (len(plots_affected) == 0):
        plots_affected = range(len(self.plots_list))
    
    if (func == "timeSlide"):
        wsize = args["wsize"]
        NpX, NcY = (self.Data_list[plots_affected[0]][0]).shape
    
        valMin = 0
        valMax = NpX - wsize
        valInit = NpX - wsize
    
    # Create the Slider canvas
    axcolor = 'lightgoldenrodyellow'
    SmarginX = 0.05
    SmarginY = 0.05
    Sheight = 0.03
    Swidth = 0.8 - 2*SmarginX
    
    axpos = plt.axes([SmarginX, SmarginY,  Swidth, Sheight],
                     axisbg=axcolor)
    sliderBar = Slider(axpos, name, valMin, valMax,  valinit = valInit)

    def slideTime(val):
        ## Function to slide the time signal through time
        val = int(val)  # Init index of the time series.
        fig = self.fig
#        print "FRGR"
        ## This will clear the axis !!
#        plt.cla()
        # Now we iterate over all the necesary plots
#        print "In bitch"
        for plot_i in plots_affected:
            lns = self.plots_list[plot_i]  # The reference to the plot
            # We might have more than one signal plotter though
            # We change the Y data shown
#            print "In bitch 2"
#            print self.plots_type[plot_i]
            for i in range(len(lns)):
#                print "In bitch 3"
                # Type of plot will change the way to treat it
                ln_type = self.plots_type[plot_i][i]
                if (ln_type == "plot"):
                    ln = lns[i]
                    ln.set_ydata(self.Data_list[plot_i][1][val:val + wsize,i])
                    ln.set_xdata (self.Data_list[plot_i][0][val:val + wsize])
                
                elif(ln_type == "fill"):
                    ## We will remove and redraw !!
                    ln = lns[i]
#                    print ln
#                    print type(ln)
                    ln.remove()
#                    x = self.Data_list[plot_i][0][val:val + wsize]
#                    y1 = self.Data_list[plot_i][1][val:val + wsize]
#                    y2 = self.Data_list[plot_i][2]
                    
                    x,y1,y2, where, ax, alpha,color, args, kwargs = self.Data_list[plot_i]
#                    ln.set_xdata (self.Data_list[plot_i][0][val:val + wsize])
                    
                    # TODO me estas puto diciendo que solo port seleccionar esto me jodes ?
                    x = x[val:val + wsize]
                    y1 = y1[val:val + wsize]
                    
                    if where is None:
                        pass
                    else:
                        where = where[val:val + wsize]
#                    print len(x), len(y1), y2
#                    print type(x), type(y1)
#                    print kwargs
                    
                    # We also need to resize "where" vector if needed.
#                    if "where" in kwargs
                    ln = ax.fill_between(x = x, y1 = y1, y2 = y2, where = where,
                                     color = color, alpha = 0.3,*args, **kwargs) 
                   
                    self.plots_list[plot_i][i] = ln
#                    print XX
#                    ln.set_ydata(self.Data_list[plot_i][1][val:val + wsize,i])
#                    ln.set_xdata (self.Data_list[plot_i][0][val:val + wsize])
                     
                elif (ln_type == "bar"):
                    ln = lns[i]
                    j = 0
                    for rect in ln:
                        rect.align = "center"
#                        print self.Data_list[plot_i][i]
                        rect.set_height(self.Data_list[plot_i][1][val + j -1])
                        rect.set_x(self.Data_list[plot_i][0][val + j-1])
                        rect.set_y(self.Data_list[plot_i][2][val + j-1])
#                        rect.align = "center"
#                        print plt.getp(rect, "width")
                        j += 1
                elif(ln_type == "candlestick"):
                    lines = self.plots_list[plot_i][0][0]
                    rects = self.plots_list[plot_i][0][1]
                    
                    axes_candlestick = self.Data_list[plot_i][1]
                    data = self.Data_list[plot_i][0]
                    
                    for line in lines:
                        line.remove()
                    for rects in rects:
                        rects.remove()
                    
                    plotting = candlestick_ohlc(axes_candlestick,data[val:val + wsize,:], width =0.7)
                    self.plots_list[plot_i][0] = plotting

                    
#                    print "d2"
#            ln.set_color("red")
                #    ax.set_title(frame)
        
        # Set the new limits of Y axes
        
        self.format_axis(val = val, wsize = wsize)
        for ax in self.axes_list:
            ax.relim()
            ax.autoscale_view()
#            # Draw the new signal
#            plt.draw()
        fig.canvas.draw_idle()
        
    ## Set the slider
    if (func == "timeSlide"):
        sliderBar.on_changed(slideTime)
        
    sliderBar.reset()
    slideTime(valInit)
    self.widget_list.append(sliderBar)
    ########################################################
    ###### Buttons for sliding as well #####################
    ##########################################################
    
    Bheight = Sheight
    Bwidth = 0.03
    
    BmarginXleft = SmarginX - Bwidth -0.001
    BmarginXright = SmarginX + Swidth + 0.001
    BmarginY = SmarginY

    BleftAx = plt.axes([BmarginXleft, BmarginY, Bwidth, Bheight])
    BrightAx = plt.axes([BmarginXright, BmarginY, Bwidth, Bheight])
    # Set the bottom into the axes, with some properties
    Bleft = Button2(BleftAx, '<', color = axcolor, hovercolor='0.975')
    Bright = Button2(BrightAx, '>', color = axcolor, hovercolor='0.975')
 
    # Create the function of the object
    def bleft_func(event,caca):
#        print caca
        sliderBar.set_val (sliderBar.val - caca) # Calls the on_changed func with the init value
    
    def bright_func(event,caca):
        sliderBar.set_val (sliderBar.val + caca) # Calls the on_changed func with the init value
    
    Bleft.on_clicked(bleft_func)
    Bright.on_clicked(bright_func)
 

    self.widget_list.append(Bleft)
    self.widget_list.append(Bright)
def slide_axis(val):
    pos = int(val)
    ## We move it by... plotting first everything and then we just 
    ## move the X limits of the Axis, not the best way bur ir is something

#    self.axes.set_xlim(self.X[val], self.X[val + wsize - 1])
#    self.axes.set_ylim(min(self.Y), max(self.Y))
#    self.fig.canvas.draw_idle()

#!/usr/bin/env python


def add_selector(self, listing):
    # We will be able to select X-frames and its boundaries
    # will be stored in the given list

    
    """
    The SpanSelector is a mouse widget to select a xmin/xmax range and plot the
    detail view of the selected region in the lower axes
    """

    def onselect(xmin, xmax):
#        indmin, indmax = np.searchsorted(x, (xmin, xmax))
#        indmax = min(len(x)-1, indmax)
        indmin = xmin
        indmax = xmax
        onselect.listing.append([indmin, indmax])
        print (onselect.listing)
    
    onselect.listing = listing
    
    # set useblit True on gtkagg for enhanced performance
    ax = self.axes
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
    
    self.widget_list.append(span)
    
def plot_wid(self, X = [],Y = [],  # X-Y points in the graph.
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
        loc = 1,
        scrolling = -1,  # Size of the window to visualize
        
        ### Super Special Shit !!
        fill = 0  #  0 = No fill, 1 = Fill and line, 2 = Only fill
        ):         

    ## Preprocess the data given so that it meets the right format
    self.preprocess_data(X,Y)
    X = self.X
    Y = self.Y
    
    NpX, NcX = X.shape
    NpY, NcY = Y.shape
    
    # Management of the figure
    self.figure_management(nf, na, labels, fontsize)
    
    plt.subplots_adjust(left = 0.05, bottom=0.1)
    ##################################################################
    ############### CALL PLOTTING FUNCTION ###########################
    ##################################################################
    ## TODO. Second case where NcY = NcX !!


        ## This function is the encharged of the plotting itself !!
        ## We must have an "array" with the different plot scenarios that we make
    

    ax = self.axes
    fig = self.fig
    wsize = scrolling
    val = NpX - wsize
    for i in range(NcY):  # We plot once for every line to plot
        self.zorder = self.zorder + 1
        colorFinal = self.get_color(color)
        if (i >= len(legend)):
            ln, = plt.plot(X[val:val + wsize],Y[val:val + wsize,i], lw = lw, alpha = alpha, 
                     color = colorFinal, zorder = self.zorder)
        else:
#            print X.shape
#            print Y[:,i].shape
            ln, = plt.plot(X[val:val + wsize],Y[val:val + wsize:,i], lw = lw, alpha = alpha, color = colorFinal,
                     label = legend[i], zorder = self.zorder)
        
        if (fill == 1):  ## Fill this shit !!
            self.filler(X[val:val + wsize],Y[val:val + wsize:,i],colorFinal,alpha)

    axcolor = 'lightgoldenrodyellow'
    # The canvas is small in the inferior part
    SmarginX = 0.05
    SmarginY = 0.05
    Sheight = 0.03
    axpos = plt.axes([SmarginX, SmarginY, 1.0 - 2*SmarginX , Sheight], axisbg=axcolor)
    sliderBar = Slider(axpos, 'Pos', 0, NpX - wsize,  valinit = NpX - wsize)
    
    
    ## Init, graph
    ln.set_ydata(Y[val:val + wsize])
    ln.set_xdata(X[val:val + wsize])
    def slide(val):
        
#            print "Pene"
        val = int(val)  # Init index of the time series.
        # We change the Y data shown
        ln.set_ydata(Y[val:val + wsize])
        ln.set_xdata(X[val:val + wsize])
#            ln.set_color("red")
    #    ax.set_title(frame)
        
        # Set the new limits of Y axes
        ax.relim()
        ax.autoscale_view()
#            # Draw the new signal
#            plt.draw()
        fig.canvas.draw_idle()
        
    sliderBar.on_changed(slide)
    sliderBar.reset()
        
    self.slider = sliderBar
    self.update_legend(legend,NcY,loc = loc)
    self.format_axis(nf, fontsize = fontsizeA)
    

    if (na == 1 or nf == 1):
        self.format_plot()
    
    return 0

def pene():
    # Create figure
    fig, ax = plt.subplots()
    # Displace the figure plotting space 25% left and 25% bottom
    # So it its shortened 25% left and 25% bottom. 
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Set the signal to plot
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0*np.sin(2*np.pi*f0*t)
    
    ## Plot the initial signal and we can reference this plot by l.
    l, = plt.plot(t, s, lw=2, color='red')
    
    # Set the axis limits for X and Y.
    plt.axis([0, 1, -10, 10])
    
    
    ## First we define axes that will be affected by the widgets.
    # A widget is associated to an axes of action. 
    
    ###########################################
    ################## SLIDERS #################
    ###########################################
    axcolor = 'lightgoldenrodyellow'
    
    # The Sliders will be plotted in another axes, these are those axes.
    # The axes is where we plot shit.
    # But this axes are where the Sliders will be.
    # plt.axes([x_init_left, y_init_bottom, width, height]
    axfreq = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
    axamp = plt.axes([0.25, 0.10, 0.65, 0.10], axisbg=axcolor)
    
    # Here we define sliders. We assign a different axes to them
    # We also define the range of values and the init value.    
    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
    
    ## Function to be used for the Widget. We will asociate it to an event
    # On the widget, it will be triggered with that event ocurrs.
    def update(val):
        amp = samp.val
        freq = sfreq.val
        # Reassign the Y values, the Xvalues remain the same
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        # Draw the new values
        fig.canvas.draw_idle()
    
    ## Set listener function to the Slider widgets
    sfreq.on_changed(update)
    samp.on_changed(update)
    
    ###################################################
    ################## Reset Buttom #################
    ###################################################
    # We define a new axes for it.
    resetax = plt.axes([0.05, 0.2, 0.1, 0.04])
    # Set the bottom into the axes, with some properties
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    # Create the function of the object
    def reset(event):
        sfreq.reset() # Calls the on_changed func with the init value
        samp.reset()
    
    button.on_clicked(reset)
    
    ###################################################
    ################## Radio buttons #################
    ###################################################
    # We define a new axes for the radio bottoms
    rax = plt.axes([0.05, 0.5, 0.15, 0.15], axisbg=axcolor)
    # We define the RadioButtons
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    
    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)
    
    plt.show()
    
    ###################################################
    ################## Dissapearing #################
    ###################################################
    # We define a new axes for it.
    resetax = plt.axes([0.05, 0.025, 0.1, 0.04])
    # Set the bottom into the axes, with some properties
    button_2 = Button(resetax, 'Hide', color=axcolor, hovercolor='0.975')
    
    # Create the function of the object
    def hide(event):
        # Static variable of the func
        hide.state = hide.state ^ 1  # Conmutate state
    #        cur_axes = l.gca()
        ## Reference the current axes
        cur_axes = ax
        cur_axes.axes.get_xaxis().set_visible(hide.state)
        cur_axes.axes.get_yaxis().set_visible(hide.state)
        l.set_visible(hide.state)
    hide.state = True
    
    button_2.on_clicked(hide)


def add_onKeyPress(self, k = "k"):
    ax = self.axes
    fig = self.fig
    def press(event):
        pressedK =  event.key
        sys.stdout.flush()
        if pressedK == 'k':
            visible = ax.get_visible()
            ax.set_visible(not visible)
            fig.canvas.draw()
            
    def release(event):
        pressedK =  event.key
        sys.stdout.flush()
        if pressedK == 'k':
            visible = ax.get_visible()
            ax.set_visible(not visible)
            fig.canvas.draw()
            
    fig.canvas.mpl_connect('key_press_event', press)
    fig.canvas.mpl_connect('key_release_event', release)


import six
class Button2(AxesWidget):
    """
    A GUI neutral button.

    For the button to remain responsive you must keep a reference to it.

    The following attributes are accessible

      *ax*
        The :class:`matplotlib.axes.Axes` the button renders into.

      *label*
        A :class:`matplotlib.text.Text` instance.

      *color*
        The color of the button when not hovering.

      *hovercolor*
        The color of the button when hovering.

    Call :meth:`on_clicked` to connect to the button
    """

    def __init__(self, ax, label, image=None,
                 color='0.85', hovercolor='0.95'):
        """
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The :class:`matplotlib.axes.Axes` instance the button
            will be placed into.

        label : str
            The button text. Accepts string.

        image : array, mpl image, Pillow Image
            The image to place in the button, if not *None*.
            Can be any legal arg to imshow (numpy array,
            matplotlib Image instance, or Pillow Image).

        color : color
            The color of the button when not activated

        hovercolor : color
            The color of the button when the mouse is over it
        """
        AxesWidget.__init__(self, ax)

        if image is not None:
            ax.imshow(image)
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        self.cnt = 0
        self.observers = {}

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        ax.set_navigate(False)
        ax.set_axis_bgcolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.color = color
        self.hovercolor = hovercolor

        self._lastcolor = color

        self.rts = []  # Threads to do thins automatically
    def _click(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        
        ## Exucute all the actions
        for cid, func in six.iteritems(self.observers):
            func(event,5)
#            print "Pen2"
#            rt = tal.RepeatedTimer(1,func, event, 5)
#            self.rts.append(rt)
            
    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)
        if not self.eventson:
            return
        if event.inaxes != self.ax:
            return
        
#        for rt in self.rts:
##            rt.stop()
#        self.rts = []
        
    def _motion(self, event):
        if self.ignore(event):
            return
        if event.inaxes == self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            if self.drawon:
                self.ax.figure.canvas.draw()

    def on_clicked(self, func):
        """
        When the button is clicked, call this *func* with event.

        A connection id is returned. It can be used to disconnect
        the button from its callback.
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass


