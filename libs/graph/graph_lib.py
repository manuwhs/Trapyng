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
    close = grba.close
    
    update_legend = grba.update_legend
    
    # Axis functions !!
    format_xaxis = grba.format_xaxis
    format_yaxis = grba.format_yaxis
    
    format_axis2 = grba.format_axis2
    color_axis = grba.color_axis
    hide_xaxis =  grba.hide_xaxis
    hide_yaxis =  grba.hide_yaxis
    
    # Setting functions !
    format_legend = grba.format_legend
    set_textRotations = grba.set_textRotations
    set_fontSizes = grba.set_fontSizes
    apply_style = grba.apply_style
    preprocess_data = grba.preprocess_data
    get_color = grba.get_color
    
    # Basic functions
    figure_management = grba.figure_management

    
    # Axes functions !
    manage_axes = grba.manage_axes
    create_axes = grba.create_axes
    twin_axes = grba.twin_axes
    get_axes = grba.get_axes
    set_xlim = grba.set_xlim
    set_ylim = grba.set_ylim
    set_zoom = grba.set_zoom
    store_WidgetData = grba.store_WidgetData
    init_WidgetData = grba.init_WidgetData
    get_barwidth = grba.get_barwidth
    
    # Subplots functions 
    set_subplots = grba.set_subplots
    next_subplot = grba.next_subplot
    subplots_adjust =  grba.subplots_adjust
    subplot2grid = grba.subplot2grid
    
    # Basic graph functions
    plot = grpl.plot
    scatter = grpl.scatter
    stem = grpl.stem
    
    bar = grpl.bar
    step = grpl.step
    
    plot_filled = grpl.plot_filled
    fill_between = grpl.fill_between
    
    add_hlines = grpl.add_hlines
    add_vlines = grpl.add_vlines

    # 3D functions
    preproces_data_3D = gr3D.preproces_data_3D
    format_axis_3D = gr3D.format_axis_3D
    plot_3D = gr3D.plot_3D
    bar_3D = gr3D.bar_3D
    scatter_3D = gr3D.scatter_3D
    add_text = grba.add_text
    ###### Advanced  #####
    barchart = grad.barchart
    candlestick = grad.candlestick

    Velero_graph = grad.Velero_graph
    Heiken_Ashi_graph = grad.Heiken_Ashi_graph
    plot_timeSeriesRange = grad.plot_timeSeriesRange

    
    ######## Specific Math graphs ##########
    plot_timeRegression = grad.plot_timeRegression
    histogram = grad.histogram
    
    ###### Widgets ######
    add_slider = grGUI.add_slider
    add_hidebox = grGUI.add_hidebox
    plot_wid = grGUI.plot_wid
    add_selector = grGUI.add_selector
    add_onKeyPress = grGUI.add_onKeyPress
    
    #### Trading ######
    tradingLineChart = trgr.tradingLineChart
    tradingVolume =  trgr.tradingVolume
    
    tradingBarChart = trgr.tradingBarChart
    tradingBarChart = trgr.tradingBarChart
    tradingCandleStickChart = trgr.tradingCandleStickChart
    
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