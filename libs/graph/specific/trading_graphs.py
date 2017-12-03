

##!/usr/bin/env python
#import matplotlib.pyplot as plt
#from matplotlib.dates import DateFormatter, WeekdayLocator,\
#    DayLocator, MONDAY
#from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
#
#
## (Year, month, day) tuples suffice as args for quotes_historical_yahoo
#date1 = (2004, 2, 1)
#date2 = (2004, 4, 12)
#
#
#mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
#alldays = DayLocator()              # minor ticks on the days
#weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
#dayFormatter = DateFormatter('%d')      # e.g., 12
#
#quotes = quotes_historical_yahoo_ohlc('INTC', date1, date2)
#if len(quotes) == 0:
#    raise SystemExit
#
#fig, ax = plt.subplots()
#fig.subplots_adjust(bottom=0.2)
#ax.xaxis.set_major_locator(mondays)
#ax.xaxis.set_minor_locator(alldays)
#ax.xaxis.set_major_formatter(weekFormatter)
##ax.xaxis.set_minor_formatter(dayFormatter)
#
##plot_day_summary(ax, quotes, ticksize=3)
#candlestick_ohlc(ax, quotes, width=0.6)
#
#ax.xaxis_date()
#ax.autoscale_view()
#plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
#
#plt.show()


# THIS VERSION IS FOR PYTHON 2 #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
candlestick = candlestick_ohlc
import matplotlib
import pylab
import datetime as dt
import utilities_lib as ul
from matplotlib import collections  as mc
#from graph_lib import gl
# We will say that gl = self

#matplotlib.rcParams.update({'font.size': 9})
import warnings

# TODO: Add legend somehow
def tradingLineChart(self, timeData, seriesName = "Close", *args, **kwargs):         
    gl = self
    price = timeData.get_timeSeries([seriesName]);
    dates = timeData.get_dates()
    ax = gl.plot(dates,price, *args, **kwargs)
    return ax

def tradingVolume(self, timeData, seriesName = "Volume", # X-Y points in the graph.
      *args, **kwargs):         
    gl = self
    volume = timeData.get_timeSeries(["Volume"]);
    dates = timeData.get_dates()
    ax = gl.stem(dates,volume, *args, **kwargs)
    return ax

# TODO: Add legend somehow
def tradingBarChart(self, timeData, # X-Y points in the graph.
              *args, **kwargs):         
    gl = self
    dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
    dates = timeData.get_dates()
    ax = gl.barchart(dates, dataHLOC,*args, **kwargs)

    return ax

def tradingCandleStickChart(self, timeData, # X-Y points in the graph.
              *args, **kwargs):         
    gl = self
    dataOCHL = timeData.get_timeSeries(["Open","Close","High","Low"])
    dates = timeData.get_dates()
    
    ax = gl.candlestick(dates, dataOCHL,*args, **kwargs)
#    ax = gl.barchart(dates, dataHLOC,*args, **kwargs)

    return ax
    
    
def tradingPV(self, timeData, ax = None,
              width = 0.6, color_mode = 0,
              fontsize = -1, Ny = 6, Nx = 20,
              volumeFactor = 3):
                  
                
    ## This thing plots the trading price, value.
    ## If an axes if given to it, it plots it there,
    ## otherwise it does it on the last

    if (color_mode == 0):
        col_spines = "#5998ff"
        bg_color = '#07000d';col_axis = 'w'
        colorup='#53c156'; colordown='#ff1717'
        colorVolume = '#00ffe8'
        
    if (color_mode == 1):
        col_spines = "#5998ff"
        bg_color = '#07000d';col_axis = 'k'
        colorup='#53c156'; colordown='#ff1717'
        colorVolume = '#00ffe8'
        
    gl = self
    stockTD = timeData.get_timeData()
    date = stockTD.index.values
    date = ul.fnp(date)
    date = ul.preprocess_dates(date)

#    return 1
#    print date[0:5]
    openp = stockTD["Open"]
    closep = stockTD["Close"]
    highp = stockTD["High"]
    lowp = stockTD["Low"]
    volume = stockTD["Volume"]
    
    x = 0
    y = len(date)
    newAr = []
    while x < y:
        appendLine = date[x],openp[x],highp[x],lowp[x],closep[x],volume[x]
        newAr.append(appendLine)
        x+=1
    
    newAr = ul.fnp(newAr).T
    

    # We will divide everything in 6x4 and this will take the 4x4 in the middle
#    print "rev %f" % width
    width = gl.get_barwidth(date, width)
#    print width
    
    if (ax is None):
        ax = self.axes

    gl.candlestick(newAr, ax = ax, width = width,colorup=colorup, colordown=colordown)

#    return 1
#    pylab.setp(ax1 , axis_bgcolor = bg_color)
    ax.set_axis_bgcolor(bg_color)

    ### Format the axis color :) !!
    gl.color_axis(ax, col_spines, col_axis)
    ### Format the axis locators ! :) !!
    gl.format_axis2(ax, Nx = Nx, Ny = Ny, fontsize = fontsize)
    ax.set_ylabel(timeData.symbol, fontsize = fontsize)
    
    ## Volume AXES, the same as the price one !
    volumeMin = 0
    ax1v = gl.plot(date, volume,  
                   fill = 1,  alpha=.4, lw = 0.1, color = colorVolume,  nf = 0, na =1)
    #    ax1v = gl.step(date, volume,  
    #                   fill = 1,  alpha=.4, lw = 0.1, color = '#00ffe8',  nf = 0, na =1)
    #    ax1v = gl.bar(date, volume,  
    #                  alpha=.4,  color = '#00ffe8',  nf = 0, na =1)
    
    ax.grid(True, color="w")
    ax1v.grid(False)
    gl.color_axis(ax1v, col_spines, col_axis)
    gl.format_axis2(ax1v, Nx = Nx, Ny = Ny, fontsize = fontsize)
    ## Remove the volume labels !!!
    ax1v.yaxis.set_ticklabels([])
    
    ## Set limits !
    ax1v.set_ylim(0,volumeFactor * max(volume))
    ax.set_xlim(date[0] -width, date[-1] + width)
    
def tradingOcillator(self, timeData, osc, osc_name = "OSC", ax = None, color_mode = 0,
                     lowline = 30, highline = 70, fontsize = 15):
    # This function formats the axes to that to see an oscillator
    # timeData it the 

    if (color_mode == 0):
        indCol = '#c1f9f7'
        posCol = '#386d13'; negCol = '#8f2020'
        bg_color = '#07000d'; col_axis = 'w'
        col_spines = "#5998ff"
        
    if (color_mode == 1):
        indCol = '#c1f9f7'
        posCol = '#386d13'; negCol = '#8f2020'
        bg_color = '#07000d'; col_axis = 'k'
        col_spines = "#5998ff"
        
    gl = self
    
    date = timeData.get_dates()
    gl.plot(date, osc, 
            color = indCol, lw=1.5, nf = 0)
    
    if (ax == None):
        ax = self.axes
    
    ax.set_axis_bgcolor(bg_color)
    # Draw some lines ! 
    ax.axhline(highline, color=negCol)
    ax.axhline(lowline, color=posCol)
    
    # Fill between the lines !
    # Since rsi has Nan, the inequalities will give a warning, but that is it.
    warnings.filterwarnings("ignore")
    gl.fill_between(x = date, y1 = osc, y2 = highline, alpha=0.99, where=(osc>=highline), facecolor=negCol, edgecolor=negCol)
    gl.fill_between(x = date, y1 = osc, y2 = lowline, alpha=0.99, where=(osc<=lowline), facecolor=posCol, edgecolor=posCol)
    warnings.filterwarnings("always")
    
    ## Format color and so on.
    ax.set_yticks([lowline,highline])
    ax.yaxis.label.set_color(col_axis)
    gl.color_axis(ax, col_spines, col_axis)
    ax.set_ylabel(osc_name, fontsize = fontsize)

def plotMACD(self, timeData, nslow = 26, nfast = 12, nema = 9, 
             ax = None, color_mode = 1):
    gl = self
    if (color_mode == 0):
        fillcolor = '#00ffe8'
        indCol = '#c1f9f7'
        posCol = '#386d13'; negCol = '#8f2020'
        bg_color = '#07000d'; col_axis = 'w'
        col_spines = "#5998ff"
    elif (color_mode == 1):
        fillcolor = '#00ffe8'
        indCol = '#c1f9f7'
        posCol = '#386d13'; negCol = '#8f2020'
        bg_color = '#07000d'; col_axis = 'k'
        col_spines = "#5998ff"
        
    date = timeData.get_dates()
    ## Calculate the MACD and the MACD averaged
    MACD_things = timeData.MACD(n_slow = nslow, n_fast = nfast, n_smooth = nema)
    
    MACD = MACD_things[:,0]
    MACDsign= MACD_things[:,1]
    MACDdiff= MACD_things[:,2]
    emaslow = timeData.EMA(n = nslow)
    emafast = timeData.EMA(n = nfast)

    if (self.axes == None):
        ax = gl.plot(date, MACD, color='#4ee6fd', lw=2, nf = 0)
    else:
        gl.plot(date, MACD, ax=ax,color='#4ee6fd', lw=2, nf = 0)
        
    gl.plot(date, MACDsign, ax=ax, color='#e1edf9', lw=1, nf = 0)
    gl.fill_between(date, MACDdiff, ax=ax, y2 = 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

    ## Format the axis
    gl.color_axis(ax, col_spines, col_axis)
    ax.set_ylabel('MACD', color='k', fontsize = 15)
    gl.format_axis2(ax, Nx = 10, Ny = 5, fontsize = -1, rotation = 45)

def plot_indicator(self, timeData, Ndiv = 4, HPV = 2, color_mode = 1):
    # Ndiv is the number of horizontal divisions
    # HPV is the length of the Price-Volume Graph

    ## This will be a trial function where we put the price and then
    # windows to put the indicators.

    gl = self
    gl.init_figure()
    
    gl.subplot2grid((Ndiv,4), (0,0), rowspan=HPV, colspan=4)
    
    gl.tradingPV(timeData, color_mode = color_mode, fontsize = 15)
    all_axes = gl.get_axes()
    
    ax = all_axes[-1]
    gl.format_axis2(ax, Nx = 10, Ny = 5, fontsize = -1, rotation = 45)

    for i in range(len(all_axes)):
        ax = all_axes[i]
        plt.setp(ax.get_xticklabels(), visible=False)
        
    #plt.setp(ax.get_xticklabels(), visible=True)
 
    plt.suptitle("Indicators Station",color='k', fontsize = 20)

def add_indicator(self, ind, name = "Indicator", inprice = 0, pos = 1):
    # pos is the 1st or second position for indicators
    gl = self
    if (inprice == 0):
        gl.subplot2grid((4,4), (2 + pos,0), rowspan=1, colspan=4, sharex = ax)
        gl.plot(ind)
    else:
        gl.plot(ind, na = 1)
        
def tradingPlatform(self, timeData, MA1 = 26, MA2 = 12,
                    volumeFactor = 3, color_mode = 1):
    
    gl = self 
    
    ## TODO, it used to be this all the way:
     ## [-SP:]
#    SP = len(date[MA2-1:])
    
    if (color_mode == 1):
        col_spines = "#5998ff"
        col_axis = 'k'
        bg_color =  '#555555'  # '#07000d'
    
    stockTD = timeData.get_timeData()
    date = stockTD.index.values
    date = ul.fnp(date)
    date = ul.preprocess_dates(date)

#    print pd.to_datetime(date[0])
#    mdates.strpdate2num('%Y%m%d')
#    date = mdates.date2num(date)
    
#    date = range(len(date))
    openp = stockTD["Open"]
    closep = stockTD["Close"]
    highp = stockTD["High"]
    lowp = stockTD["Low"]
    volume = stockTD["Volume"]
    
    x = 0
    y = len(date)
    newAr = []
    while x < y:
        appendLine = date[x],openp[x],highp[x],lowp[x],closep[x],volume[x]
        newAr.append(appendLine)
        x+=1
    
#    print newAr[0:4]
    newAr = ul.fnp(newAr).T
#    print newAr.shape
    ## Plotting !!
#    fig = plt.figure(facecolor='#07000d')

    ###########################################################
    ################# PLOTTING THE CANDLESTICK ################
    ###########################################################
    # We will divide everything in 6x4 and this will take the 4x4 in the middle
    
    # Obtain data to plot
    Av1 = timeData.SMA( n = MA1)
    Av2 = timeData.SMA(n = MA2)
    
    ax1 = gl.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg= bg_color)
    gl.plot(date,Av1, nf = 0,
            color = '#e1edf9',legend=[str(MA1)+' SMA'], lw = 1.5)
    gl.plot(date,Av2, nf = 0,
            color = '#4ee6fd',legend=[str(MA2)+' SMA'], lw = 1.5)
    ax1.grid(True, color='w')
    
    gl.tradingPV(timeData, color_mode = 1, fontsize = 15, volumeFactor = volumeFactor)
    
    ## Format the legend !!
    maLeg = ax1.legend(loc=9, ncol=2, prop={'size':7},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = ax1.get_legend().get_texts()
    pylab.setp(textEd[0:5], color = 'w')
        
    #####################################################################
    ############## In the upper part plot RSI ###########################
    #####################################################################
    rsi = timeData.RSI(n = 14)
#    print rsi[0:30]
    ax0 = gl.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    tradingOcillator(self, timeData, rsi, osc_name = "RSI", ax = None, color_mode = 1,
                     lowline = 30, highline = 70)
    
    #####################################################################
    ############## MACD Axes ###########################
    #####################################################################

    ax2 = gl.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')

    gl.plotMACD(timeData, ax = ax2)
    ## Remove the xticklabels of the other axes !!
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ## Final touches !!! 
    plt.suptitle("Trasing Station",color='k', fontsize = 20)
    ### Do some stupid anotation !!!!
#    ax1.annotate('Big news!',(date[510],Av1[510]),
#        xytext=(0.8, 0.9), textcoords='axes fraction',
#        arrowprops=dict(facecolor='white', shrink=0.05),
#        fontsize=14, color = 'w',
#        horizontalalignment='right', verticalalignment='bottom')


    ## Smaller overall image ! So that we can put more stuff !!
    plt.subplots_adjust(left=.09, bottom=.14, right=.70, top=.95, wspace=.20, hspace=0)
    plt.show()

#    ws = 100
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
 
