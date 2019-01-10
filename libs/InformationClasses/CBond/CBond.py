
import pandas as pd
import numpy as np
import datetime as dt
import Bond_core as Boco
import Bond_estimation as Boes

class CBOND:
    # THE CAPM class takes a portfolio as input and operates.
    ## Can we maxime the return subject to that the total beta is 0 ?
    # Or minimizing beta ? Can we do that ? 

    def __init__(self, name = "", freq = 2, coupon = 100, par = 1000):
        
        ## Properties of the structure of the bond
        self.name = name
        self.freq = freq    # Frequency of the 
        self.coupon = coupon  # Amount of money we will get paid at every preriod
        self.par = par    # Par value, face. What we will be paid at the finish
                         # of the bond time
        
        ### Properties that depend on the time.
        self.timeToMaturity = -1 # Time left to the end of the bond
        self.Byield = -1  # The current yield of the bond (coup/Price)
        self.ytm = -1     # Yield to maturity
        self.price = -1   # Price of the bond at the current time.
  
        ############################
  
    set_name = Boco.set_name
    set_freq = Boco.set_freq
    set_coupon = Boco.set_coupon
    set_par = Boco.set_par
    
    set_price = Boco.set_price
    set_timeToMaturity = Boco.set_timeToMaturity
    set_ytm = Boco.set_ytm
            
    get_price = Boes.get_price
    get_ytm = Boes.get_ytm
    get_mduration = Boes.get_mduration
    get_convexity = Boes.get_convexity
    
    estimate_price_DC = Boes.estimate_price_DC
    yieldPriceStudy = Boes.yieldPriceStudy
    
    