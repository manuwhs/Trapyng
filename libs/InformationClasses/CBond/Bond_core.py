
import pandas as pd
import numpy as np
import datetime as dt



#################################################
############### Setting functions ###############
#################################################
def set_name(self, name = ""):
    self.name = name
    
def set_freq(self, freq = 2):
    self.freq = freq
def set_coupon(self, coupon = 100):
    self.coupon = coupon
def set_par(self, par = 1000):
    self.par = par

def set_price(self, price = 1000):
    self.price = price
def set_timeToMaturity(self, timeToMaturity = 1.0):
    self.timeToMaturity = timeToMaturity
def set_ytm(self, ytm = 0.10):
    self.ytm = ytm
            
