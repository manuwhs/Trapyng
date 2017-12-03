import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities_lib as ul

import graph_basic as grba
import graph_plots as grpl
import graph_advanced as grad
import graph_3D as gr3D

import graph_GUI as grGUI
import trading_graphs as trgr

class CGraph ():
    
    def __init__(self,w = 20, h = 12, lw = 2):
        self.init_variables(w = w, h = h, lw = lw)
    
    init_variables = grba.init_variables
    savefig = grba.savefig
    set_labels = grba.set_labels
    init_figure = grba.init_figure
    update_legend = grba.update_legend
    format_axis = grba.format_axis
    format_axis2 = grba.format_axis2
    color_axis = grba.color_axis
    
    subplots_adjust =  grba.subplots_adjust
    hide_xaxis =  grba.hide_xaxis
    
    format_plot = grba.format_plot
    preprocess_data = grba.preprocess_data
    get_color = grba.get_color
    figure_management = grba.figure_management
    subplot2grid = grba.subplot2grid
    
    manage_axes = grba.manage_axes
    create_axes = grba.create_axes
    twin_axes = grba.twin_axes
    get_axes = grba.get_axes
    set_xlim = grba.set_xlim
    set_ylim = grba.set_ylim
    
    get_barwidth = grba.get_barwidth
    
    set_subplots = grba.set_subplots
    next_subplot = grba.next_subplot
    
    plot = grpl.plot
    plot_filled = grpl.plot_filled
    scatter = grpl.scatter
    bar = grpl.bar
    stem = grpl.stem
    step = grpl.step
    fill_between = grpl.fill_between

    preproces_data_3D = gr3D.preproces_data_3D
    format_axis_3D = gr3D.format_axis_3D
    plot_3D = gr3D.plot_3D
    bar_3D = gr3D.bar_3D
    scatter_3D = gr3D.scatter_3D
    add_text = grba.add_text
    ###### Advanced  #####
    candlestick = grad.candlestick
    histogram = grad.histogram
    Velero_graph = grad.Velero_graph
    Heiken_Ashi_graph = grad.Heiken_Ashi_graph
    plot_timeSeriesRange = grad.plot_timeSeriesRange
    plot_timeRegression = grad.plot_timeRegression
    ###### Widgets ######
    add_slider = grGUI.add_slider
    add_hidebox = grGUI.add_hidebox
    plot_wid = grGUI.plot_wid
    add_selector = grGUI.add_selector
    add_onKeyPress = grGUI.add_onKeyPress
    
    #### Trading ######
    tradingPlatform = trgr.tradingPlatform
    tradingPV = trgr.tradingPV
    tradingOcillator = trgr.tradingOcillator
    plotMACD = trgr.plotMACD
    plot_indicator = trgr.plot_indicator
    add_indicator = trgr.add_indicator
gl = CGraph()

#import numpy as np
#import matplotlib.pyplot as plt
##from matplotlib.widgets import TextBox
#fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)
#t = np.arange(-2.0, 2.0, 0.001)
#s = t ** 2
#initial_text = "t ** 2"
#l, = plt.plot(t, s, lw=2)
#
#
#def submit(text):
#    ydata = eval(text)
#    l.set_ydata(ydata)
#    ax.set_ylim(np.min(ydata), np.max(ydata))
#    plt.draw()
#
#axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
#text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
#text_box.on_submit(submit)
#
#plt.show()