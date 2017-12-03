""" Advance Analysis. BL, Timing, Distributions """
import os
os.chdir("../")
import import_folders
# Classical Libraries
import copy as copy
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
# Own graphical library
from graph_lib import gl 
# Data Structures Data
import CPortfolio as CPfl
import CCAPM as CCAPM

# Import functions independent of DataStructure
import utilities_lib as ul

plt.close("all")
##########################################################
############### CAPM model !!! ######################
##########################################################
## Load the porfolio in the CAPM model
periods = [43200]   # 43200 1440
source = "Yahoo" # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = source)
 
#symbols = ["XAUUSD", "XAGUSD", "NYMEX.CL"]
symbols =  ["GE", "HPQ","XOM"]
#symbols = ["Mad.ELE","Mad.ENAG","Mad.ENC","Mad.EZE","Mad.FER","Mad.GAM"]
#symbols = ["USA.JPM","USA.KO","USA.LLY","USA.MCD","USA.MMM","USA.MO","USA.MON",
#           "USA.MRK","USA.OXY","USA.PEP","USA.PFE"]
################## Date info ###################
sdate_str = "01-01-2010"
edate_str = "01-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

####### LOAD SYMBOLS AND SET Properties   ###################
Cartera = CPfl.Portfolio(symbols, periods)   # Set the symbols and periods to load
Cartera.load_symbols_csv(storage_folder)

########## Set the CAPM model object ##########
CAPMillo = CCAPM.CAPM(Cartera, periods[0])
CAPMillo.set_allocation([])  # Initial allocation of the porfolio
CAPMillo.set_Rf(0.0)  # Risk-free rate
CAPMillo.set_seriesNames(["Close"])  # Adj Close 
CAPMillo.set_index('XAUUSD')  # Set the index commodity

CAPMillo.pf.set_interval(sdate,edate)

########## FILLING THE DATA ##########
print "Filling Data"
CAPMillo.pf.fill_data()
print "Data Filled"

dates = CAPMillo.pf.symbols[symbols[0]].TDs[periods[0]].dates
print "Market Timing "
print CAPMillo.get_portfolio_ab()

Black_litterman = 0
if (Black_litterman == 1):
    # Take the values from He & Litterman, 1999.
    
    # Calculate initial portfolio from the market capitalization
    mc = [317957301, 556812224, 532667693, 1703857627, 989410674]
    weq = [0.15,0.25,0.1,0.28,0.22]

    Sigma = [[0.04881,	0.03311,	0.03080,	0.04534,	0.03998],
            [0.03311,	0.04129,	0.02939,	0.03777,	0.03321],
            [0.03080,	0.02939,	0.03406,	0.03804,	0.03348],
            [0.04534,	0.03777,	0.03804,	0.06446,	0.04618],
            [0.03998,	0.03321,	0.03348,	0.04618,	0.04839]]
    
    # Study !!!
    # We are assuming efficiency. The amount of money in a place
    # accounts for its risk. So the weights have the return..

    # Risk aversion of the market 
    delta = 2  # Names gamma of A by the professor.
    # Almost increases linearly the Posterior return.
    # It also depends in the prior Q and Omega
    # It is close to the fucking sharpe ratio.
    # delta = (Rp - Rf)/Sigma_p^2.
    # If we multipy w by a constant (we like more risk).
    # And the delta increases.
    
    Sigma_shit  = np.sqrt(np.dot(np.dot(weq,Sigma),weq))
    
    # Coefficient of uncertainty in the prior estimate of the mean
    tau = 0.3
    ### Prior of out Views !!!
    P1 = [[ -1,	1,	0,	0,	0],
          [ 0, -1	,1,	0,	0]]
    P1 = ul.fnp(P1)
    
    # If we invert P1 and Q1 at the same time we get the same
    Q1 = [-0.02, -0.01]
    Q1 = ul.fnp(Q1)
    
    Omega1 = np.dot(np.dot(P1,Sigma),P1.T) * np.eye(Q1.shape[0])
    
    res = CAPMillo.BlackLitterman(weq, Sigma, delta, # Prior portfolio variables
                   tau,              # Uncertainty coefficient of the porfolio priors
                   P1, Q1, Omega1)   # Prior views variables
    
    # Reference returns of the portfolio of the market
    # They can just be calculated using the portfolio
    refPi = delta * np.dot(Sigma, weq)              
    
    print refPi  # Expected returns wihout priors
    print res[0] # Expected returns after priors
    
    # A priory the expected return Posteriori does not have to be bigger
    # Just more accuarate to reality if our views are right :)
    Ereturn = np.dot(refPi,weq)
    EreturnPost = np.dot(res[0],res[1])
    
    print Ereturn
    print EreturnPost
#    display('View 1',assets,res)
    
