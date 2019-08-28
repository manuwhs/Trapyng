# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:55:37 2016

@author: montoya
"""
#
import import_folders
import pandas as pd
import numpy as np
import DDBB_lib as DBl
import CCAPM as CCAPM

from graph_lib import gl
import bond_math as boma
import CBond as CBond
import matplotlib.pyplot as plt
plt.close("all")


## Generate a list of bonds:
bond_names = []
myBond = CBond.CBOND( name = "hola", freq = 2, coupon = 5.75, par = 100.)
# Set some properties 
myBond.set_price(95.0428)
myBond.set_timeToMaturity(2.5)



### All the info of the symbols
#symbols = ['AMD', 'BAC', 'MSFT', 'TXN']
try_lib = 1
if (try_lib):
    myBond = CBond.CBOND( name = "hola", freq = 2, coupon = 5.75, par = 100.)
    # Set some properties 
    myBond.set_price(95.0428)
    myBond.set_timeToMaturity(2.5)
    myBond.set_ytm(0.10)
    
    # Get the yield to maturity
    print myBond.get_ytm()
    print myBond.get_ytm(price = 96)
    print myBond.get_ytm(price = 96, T = 3.5)
    
    # Get the price
    print myBond.get_price()
    print myBond.get_price(ytm = 0.10)
    print myBond.get_price(ytm = 0.11, T = 5.5)
    
    print myBond.get_mduration(price = 96, T = 5.5, dy=0.01)
    print myBond.get_convexity(price = 96, T = 5.5, dy=0.01)
    
    ########################################################
    #################### Cool graphs ######################
    #######################################################

