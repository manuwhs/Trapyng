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

import datetime as dt
import numpy as np
import CTimeData as CTD
import copy as copy
import utilities_lib as ul
from yahoo_finance import Share
import CPortfolio as CPfl
import utilities_lib as ul
import matplotlib.pyplot as plt

import scipy.optimize as optimize

from graph_lib import gl
plt.close("all")

##########################################################
############### Functions to operate with Bond ###############
##########################################################

def spot_interest(R,T):
     # This function calculates the compound return of an investment 
     # with simple interest rate "R" over T periods of investment.
     si = np.exp(R*T) - 1 # R + R(1+R) + R(1 + R + R(1+R)) + ....
     return si
     
def compound_interest(R,T):
     # This function calculates the compound return of an investment 
     # with simple interest rate "R" over T periods of investment.
     ci = (1 + R)**T - 1 # R + R(1+R) + R(1 + R + R(1+R)) + ....
     return ci

def simple_interest(R,T):
     # This function calculates the compound return of an investment 
     # with simple interest rate "R" over T periods of investment.
    si = R*T
    return si

def bond_ytm(price, par, T, coupon, freq = 2, guess = 0.05):
    ## This function gets the yield to maturity of a bond.
    ## It calculates the internal interest rate that the bold applies
    ## Price: Price of the bond
    # T: Number of years of the bond left. (Usually measured in years
    # with the frequency being its lowest resolution (like  half a year = 0.5))
    # coup: Amount paid every period
    # par: Principal. Huge amount of money we will get paid at the end.
    # ytm: Yield to maturity. Interest rate of the bond applied to all the cupons
    # and principal.
    freq = float(freq)
    periods = T*freq
    coupon = coupon/100.*par/freq
    deltat = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda(y): \
    sum([coupon/(1+y/freq)**(freq*t) for t in deltat]) + \
    par/(1+y/freq)**(freq*t) - price

    return optimize.newton(ytm_func, guess)

def bond_price(par, T, ytm, coupon, freq = 2):
    # This function calculates the prices of a bond.
    # freq: Number of times a year that the bond gives money
    # T: Number of years of the bond left. (Usually measured in years
    # with the frequency being its lowest resolution (like  half a year = 0.5))
    # ytm: Yield of the bond.
    # coup: Amount paid every period
    # par: Principal. Huge amount of money we will get paid at the end.
    # ytm: Yield to maturity. Interest rate of the bond applied to all the cupons
    # and principal.

    freq = float(freq)
    periods = T*freq
    
    coupon = coupon/100.*par/freq
    
    # Calculate 
    deltat = [(i+1)/freq for i in range(int(periods))]
    
    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in deltat]) + \
            par/(1+ytm/freq)**(freq*T)
            
    return price

def bond_mduration(price, par, T, coupon, freq, dy=0.01):
    # Modified duration of a bond
    #The modified duration of a bond can be thought of as the first derivative of the
    #relationship between price and yield
    #The higher the duration of a bond, the more sensitive it is to yield changes.
    #Conversely, the lower the duration of a bond, the less sensitive it is to yield changes.

    # Derivative divided by price !!!!!

    # Calculate the ytm of the bond now.
    ytm = bond_ytm(price, par, T, coupon, freq)
    
    # Get the points for the derivative
    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coupon, freq)
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coupon, freq)
    
    # Perform discrete derivative.
    firstDerivative = (price_minus-price_plus)/(2*dy)
    mduration = firstDerivative/price
    return mduration
    
def bond_convexity(price, par, T, coupon, freq, dy=0.01):
     # Second derivarive =
#    Convexity is the sensitivity measure of the duration of a bond to yield changes.
#    If the yield 
#    Think of convexity as the second derivative of the relationship between the price
#    and yield:
    
    # We calculate the curent yield of the bond
    ytm = bond_ytm(price, par, T, coupon, freq)
    # Now we have a point in the curve.
    # To get its derivative we get a point to the left and a point to 
    # the right to get two point that will give as a line which it the derivative
    # of the yield curve at that point
    
    # Same... second derivative divided by price !!
    ytm_minus = ytm - dy   
    ytm_plus = ytm + dy
    
    price_minus = bond_price(par, T, ytm_minus, coupon, freq)
    price_plus = bond_price(par, T, ytm_plus, coupon, freq)
    
    secondDerivative = (price_minus+price_plus-2*price)/dy**2
    convexity = secondDerivative/price
    return convexity
    
    
    
def calculate_forward_rate(T1,R1, R2, T2):
    # T1 is the time to maturity of the 
    # Here, r 1 andr2 are the continuously compounded annual interest rates at time period
    # T1 and T2 respectively.
    forward_rate = (R2*T2 - R1*T1)/(T2 - T1)
    return forward_rate
    
#plot_compound_understanding()
def plot_compound_understanding():
    years = 50  # Total duration of the investment
    years = range(1,years+1)
    frequencies = [0.1, 0.5, 1.0, 2, 12, 52, 365]
    R = 0.05  # Annual Rate
    retruns = []
    
    flag_plot = 1
    for freq in frequencies:
        returns_i = []
        for year in years:
            returns_i.append(compound_interest(R/freq,year*freq))
        
        gl.plot(years,returns_i, 
                labels = ["Bond compound Return", "Years","Return"],
                legend = ["Frequency %f" % freq],
                nf = flag_plot,
                loc = 2)
        flag_plot = 0
    
    # We also plot the one with simple interest
    returns_i = []
    for year in years:
        returns_i.append(simple_interest(R/freq,year*freq))
    
    gl.plot(years,returns_i, 
            labels = ["Bond compound Return", "Years","Return"],
            legend = ["Simple interest"],
            lw = 5,
            nf = flag_plot,
            loc = 2)

    # We also plot the one with spot interest
    returns_i = []
    for year in years:
        returns_i.append(spot_interest(R/freq,year*freq))
    
    gl.plot(years,returns_i, 
            legend = ["Spot interest"],
            lw = 5,
            nf = flag_plot,
            loc = 2)

    ## The limit is the Spot Rate !! When we can do it continuously
    ## In this case the return is e^(RT)