
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr
import matplotlib.colors as ColCon
import copy

import indicators_lib as indl
def get_BollingerBand (timeSeries, L = 20):
    ## Get the Bollinger Bands !!
    
    MA = indl.get_SMA(timeSeries, L = L)
    diff = np.power(timeSeries - MA, 2);# Get the difference to the square

#    print MA.shape
#    print price.shape
#    print diff.shape

    diff_SMA = indl.get_SMA(diff, L)  # Get the avera STD over L periods
    diff_SMA = 2 * np.sqrt(diff_SMA)
    
    # Now we apply a MA over this shit
    
    return diff_SMA