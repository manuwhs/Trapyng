

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
import urllib2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
candlestick = candlestick_ohlc
import matplotlib
import pylab

import graph_basic as grba
import utilities_lib as ul

from graph_lib import gl

matplotlib.rcParams.update({'font.size': 9})

eachStock = 'EBAY','TSLA','AAPL'
plt.close("all")
def rsiFunc(prices, n=14):
#    n = float(n)
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow

def graphData(stockTD,MA1,MA2):
    '''
        Use this to dynamically pull a stock:
    '''
    try:
        print 'Currently Pulling',stock
    except Exception, e:
        print str(e), 'failed to organize pulled data.'

    ## Now we use the data
#    try:
        ## Basic data obtaining
    date = stockTD.index.values
    date = ul.fnp(date)
    date = grba.preprocess_dates(date)
    print date[0]
    print np.array(date[0])
#    print pd.to_datetime(date[0])
#    mdates.strpdate2num('%Y%m%d')
    date = mdates.date2num(date)
    
#    date = range(len(date))
    openp = stockTD["Open"]
    closep = stockTD["Close"]
    highp = stockTD["High"]
    lowp = stockTD["Low"]
    volume = stockTD["Volume"]
    
    # Obtain data to plot
    Av1 = movingaverage(closep, MA1)
    Av2 = movingaverage(closep, MA2)
    SP = len(date[MA2-1:])
    
    x = 0
    y = len(date)
    newAr = []
    while x < y:
        appendLine = date[x],openp[x],closep[x],highp[x],lowp[x],volume[x]
        newAr.append(appendLine)
        x+=1
    ## Plotting !!
#    fig = plt.figure(facecolor='#07000d')

    ###########################################################
    ################# PLOTTING THE CANDLESTICK ################
    ###########################################################
    # We will divide everything in 6x4 and this will take the 4x4 in the middle
    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
    candlestick(ax1, newAr[-SP:], width=.6, colorup='#53c156', colordown='#ff1717')
#    gl.Velero_graph(stockTD)
    # Plot the means :)
    Label1 = str(MA1)+' SMA'
    Label2 = str(MA2)+' SMA'
    ax1.plot(date[-SP:],Av1[-SP:],'#e1edf9',label=Label1, linewidth=1.5)
    ax1.plot(date[-SP:],Av2[-SP:],'#4ee6fd',label=Label2, linewidth=1.5)
    
    ax1.grid(True, color='w')
    
    ### Format the axis  :) !!
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("w")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price and Volume')

    maLeg = plt.legend(loc=9, ncol=2, prop={'size':7},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color = 'w')

    ## Volume AXES, the same as the price one !
    volumeMin = 0
    ax1v = ax1.twinx()
    ax1v.fill_between(date[-SP:],volumeMin, volume[-SP:], facecolor='#00ffe8', alpha=.4)
    
    ### Set ylim so that the volume will be seen below
    
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ax1v.set_ylim(0, 3*volume.max())
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')
    
    #####################################################################
    ############## In the upper part plot RSI ###########################
    #####################################################################

    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    rsi = rsiFunc(closep)
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'
    
    ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
    
    # Draw some lines ! 
    ax0.axhline(70, color=negCol)
    ax0.axhline(30, color=posCol)
    
    # Fill between the lines !
    ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    
    ## Format color and so on.
    ax0.set_yticks([30,70])
    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')

    #####################################################################
    ############## MACD Axes ###########################
    #####################################################################

    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    fillcolor = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9
    
    ## Calculate the MACD and the MACD averaged
    emaslow, emafast, macd = computeMACD(closep, slow = nslow, fast = nfast)
    ema9 = ExpMovingAverage(macd, nema)
    
    ax2.plot(date[-SP:], macd[-SP:], color='#4ee6fd', lw=2)
    ax2.plot(date[-SP:], ema9[-SP:], color='#e1edf9', lw=1)
    ax2.fill_between(date[-SP:], macd[-SP:]-ema9[-SP:], 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

    ## Format the axis
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('MACD', color='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)


    plt.suptitle("CACA",color='w')

    ## Remove the xticklabels of the other axes !!
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ### Do some stupid anotation !!!!
    ax1.annotate('Big news!',(date[510],Av1[510]),
        xytext=(0.8, 0.9), textcoords='axes fraction',
        arrowprops=dict(facecolor='white', shrink=0.05),
        fontsize=14, color = 'w',
        horizontalalignment='right', verticalalignment='bottom')


    ## Smaller overall image ! So that we can put more stuff !!
    plt.subplots_adjust(left=.09, bottom=.14, right=.70, top=.95, wspace=.20, hspace=0)
    plt.show()
    fig.savefig('example.png',facecolor=fig.get_facecolor())
       
#    except Exception,e:
#        print 'main loop',str(e)


graphData(timeData.get_timeData(),10,50)