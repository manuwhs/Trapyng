# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:55:37 2016

@author: montoya
"""
#
import import_folders
import pandas as pd
import numpy as np
from graph_lib import gl
import bond_math as boma
import utilities_lib as ul
##########################################################
############### Functions to operate with Bond ###############
##########################################################

def spot_interest(self):
     R = self.ytm
     T = self.timeToMaturity * self.freq
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

def get_ytm(self, price = -1, T = -1, guess = 0.05):

    if (price == -1):
        price = self.price
    if (T == -1):
        T = self.timeToMaturity
        
    ytm = boma.bond_ytm(
                  par = self.par,
                  coupon = self.coupon,
                  freq = self.freq,
                  T = T, price = price, guess = guess)
    return ytm
    
def get_price(self, ytm = -1, T = -1):
    if (ytm == -1):
        ytm = self.ytm
    if (T == -1):
        T = self.timeToMaturity
        
    price = boma.bond_price(
                  par = self.par,
                  coupon = self.coupon,
                  freq = self.freq,
                  T = T, ytm = ytm)
    return price
    
def get_mduration(self, price = -1, T = -1, dy=0.01):
    if (price == -1):
        price = self.price
    if (T == -1):
        T = self.timeToMaturity
        
    mduration = boma.bond_mduration(price = price, dy=dy, 
                            par = self.par,
                            coupon = self.coupon,
                            freq = self.freq,
                            T = T)
    return mduration
    
def get_convexity(self, price = -1, T = -1,dy=0.01):
    if (price == -1):
        price = self.price
    if (T == -1):
        T = self.timeToMaturity
    convexity = boma.bond_convexity(price = price, dy = dy,
                            par = self.par,
                            coupon = self.coupon,
                            freq = self.freq,
                            T = T)

    return convexity

def estimate_price_DC(self, price, dytm = 0.01, dy = 0.01):
    ## This function estimates the new price if there is a change
    ## in the ytm by using the mduration and convexity, which are just
    ## first and second order derivaitons. Taylor decomposition
    D = self.get_mduration(price = price, dy = dy)
    C = self.get_convexity(price = price, dy = dy)

    ## - because the mduration is the derivative negated
    deltaPrice =  price * (-dytm * D  + C * dytm**2 /2);
    price_estimation = price + deltaPrice
    
    return price_estimation
    
def calculate_forward_rate(T1,R1, R2, T2):
    # T1 is the time to maturity of the 
    # Here, r 1 andr2 are the continuously compounded annual interest rates at time period
    # T1 and T2 respectively.
    forward_rate = (R2*T2 - R1*T1)/(T2 - T1)
    return forward_rate
    
def yieldPriceStudy(self, initial_price = 80):
    # The initial price is for the aproximation of the
    # funciton with a cuadratic equation in one point.

    #### Obtain the yield-price curve from the structure    
    Np = 100
    ytm_list = np.linspace(0.001, 0.40, Np)
    
    prices = []
    mdurations = []
    convexities = []
    for ytm in ytm_list:
        price = self.get_price(ytm = ytm)
        mduration = self.get_mduration(price = price)
        convexity = self.get_convexity(price = price)
        
        prices.append(self.get_price(ytm = ytm))
        mdurations.append(mduration)
        convexities.append(convexity)
    
    gl.set_subplots(2,2)
    gl.plot(ytm_list,prices, 
            labels = ["Yield curve", "Yield to maturity" ,"Price of Bond"],
            legend = ["Yield curve"], loc = 3)
    
    gl.plot(ytm_list,prices, 
            labels = ["Duration and Yield", "Yield to maturity" ,"Price of Bond"],
            legend = ["Yield curve"], loc = 3)
    gl.plot(ytm_list,mdurations, na = 1, nf = 0,
            legend = ["Duration"], loc = 1)
    
    gl.plot(ytm_list,prices, 
            labels = ["Convexity and Yield","Yield to maturity" ,"Price of Bond"],
            legend = ["Yield curve"], loc = 3)
    gl.plot(ytm_list,convexities, na = 1, nf = 0,
            legend = ["Convexity"], loc = 1)
            
    ### Estimation of the yield courve around a point using the 
    ## Duration and convexity. 
    
    price = initial_price
    ytm = self.get_ytm(price)
    dytmList = np.linspace(-0.10, 0.10, 100)
    
    ## Obtain estimations
    estimations = []
    for dytm in dytmList:
        eprice = self.estimate_price_DC(price = price, dytm = dytm, dy = 0.01)
        estimations.append(eprice)
    
    ## Obtain real values
    rael_values = []
    for dytm in dytmList:
        rprice = self.get_price(ytm = ytm + dytm)
        rael_values.append(rprice)
    
    # Calculate the error !
    rael_values = ul.fnp(rael_values)
    estimations = ul.fnp(estimations)
    error = np.abs(np.power(rael_values - estimations, 1))
    
    gl.plot(ytm_list,prices, 
            labels = ["Convexity and Yield","Yield to maturity" ,"Price of Bond"],
            legend = ["Yield curve"], loc = 3, lw = 4, color = "r")
    
    gl.scatter([ytm],[price], nf = 0, 
            legend = ["Initial price"], loc = 3, lw = 4, color = "g")
    
    gl.plot(dytmList + ytm,estimations,  nf = 0,
            legend = ["Estimation"], loc = 3, lw = 2, color = "b")
    
    gl.plot(dytmList + ytm,error,  nf = 0, na = 1,
            legend = ["Error"], loc = 1, lw = 2, color = "b")


    ## The limit is the Spot Rate !! When we can do it continuously
    ## In this case the return is e^(RT)