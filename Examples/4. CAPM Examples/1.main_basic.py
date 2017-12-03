""" BASIC USE OF THE LIBRARY FOR INTRODUCTORY ANALYSIS """
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

# We get the dates we work on.
# TODO: Define what are the dates of a portfolio, or well, it depends on the period
dates = CAPMillo.pf.symbols[symbols[0]].TDs[periods[0]].get_dates()

## Get some efficient frontier and simulate the portfolio
efficient_frontiers = 1
efficient_frontiers2 = 0
Black_litterman = 0
CAPM_model = 1


if (CAPM_model == 1):
    CAPMillo.set_allocation([])
    CAPMillo.set_Rf(0.0)
    
    CAPMillo.set_index(symbols[0])
    param = CAPMillo.get_symbol_ab(symbols[1])
    print param
    
    params = CAPMillo.get_all_symbols_ab()
    print params
    
    param = CAPMillo.get_portfolio_ab(mode = "normal")    
    print param
    
    param = CAPMillo.get_portfolio_ab(mode = "gaussian")    
    print param
    
    JensenAlpha = CAPMillo.get_portfolio_JensenAlpha()
    
    ## IDEA !! Maybe use the portfolio in the frontier that maximizes
    ## the alpha and minimizes the beta !!! Maybe minimizing beta is not as important
    ## In the CAMP we already have the total Exp and risk.
    ## Alpha and beta say: Does out portolio perform better than the market ?
    ## If we just follow the market, investing everything on the index,
    ## Thus investing in everything proportionally to their capital.
    ## Then we have alpha = 0 and beta = 1 
#    CAPMillo.test_symbol_ab(symbols[2])
    CAPMillo.test_Jensens_Alpha()
    
    # Do it again with an optimal portolio
    w = CAPMillo.TangentPortfolio(Rf = 0.0)
    CAPMillo.set_allocation(w)
    CAPMillo.test_Jensens_Alpha()
    
    print "Market Timing "
    print CAPMillo.get_portfolio_ab()


if (efficient_frontiers == 1):
    gl.set_subplots(2,2)
    Nalloc = 10000
    #1
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "yes", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8, legend = ["Normal alloc"])
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Tangent", max_exp = 100.0)
    CAPMillo.plot_allocations(portfolios, legend = ["Normal Eff"], nf = 0)
    CAPMillo.scatter_allocations(np.eye(CAPMillo.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)

    #2
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "Lintner", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8,nf = 1, legend = ["Lintner alloc"])
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Lintner")
    CAPMillo.plot_allocations(portfolios, legend = ["Lintner Eff"], nf = 0)
    CAPMillo.scatter_allocations(np.eye(CAPMillo.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)

    #3
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "no", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8,nf = 1, legend = ["Markovitz alloc"])
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Markowitz")
    CAPMillo.plot_allocations(portfolios, legend = ["Markowitz Eff"], nf = 0)
    CAPMillo.scatter_allocations(np.eye(CAPMillo.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)

    #4
    # WARNING !! The calculation of the efficient frontier this way could be
    # wrong if the covariance matrix is not good enough.
#    ## Frontier and portfolios when we allow short sales.
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "yes", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8, legend = ["Normal alloc"])
    
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Tangent", max_exp = 10.0)
    CAPMillo.plot_allocations(portfolios, legend = ["Normal Eff"], nf = 0)

    ## Frontier and portfolios when we allow short sales but constained in 
    ## in the sum of the absolute values.
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "Lintner", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8,nf = 0, legend = ["Lintner alloc"])
    
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Lintner")
    CAPMillo.plot_allocations(portfolios, legend = ["Lintner Eff"], nf = 0)
#    
    # Get the efficient frontier where we cannot borrow or lend money
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "no", mode = "gaussian")
    CAPMillo.scatter_allocations(alloc, alpha = 0.8,nf = 0, legend = ["Markovitz alloc"])
    
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Markowitz")
    CAPMillo.plot_allocations(portfolios, legend = ["Markowitz Eff"], nf = 0)

    # Scatter Assets
    CAPMillo.scatter_allocations(np.eye(CAPMillo.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)
    

if (efficient_frontiers2 == 1):
    # Other way for finding efficient frontier
    # Scatter random porfolio so that the sum of all the allocation is 1

#    portfolios = CAPMillo.TangenPortfolioFrontier(norm = "none", maxRf = 0.0032)
#    CAPMillo.scatter_allocations(portfolios, nf = 1)

#    portfolios = CAPMillo.TangenPortfolioFrontier2(norm = "none", maxRf = 0.01)
#    CAPMillo.scatter_allocations(portfolios, nf = 1)

    Nalloc = 100000
    alloc = CAPMillo.get_random_allocations(Nalloc, short = "yes", mode = "gaussian")
    
    CAPMillo.scatter_allocations(alloc, alpha = 0.8, legend = ["Normal alloc"])
    optimal, portfolios = CAPMillo.efficient_frontier(kind = "Tangent")
    CAPMillo.plot_allocations(portfolios, legend = ["Normal Efficient Eff"], nf = 0)

    # Scatter Assets
    CAPMillo.scatter_allocations(np.eye(CAPMillo.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)


####################################################
#### Examples model !!!
####################################################

examples_F = 0
if (examples_F == 1):
    
    # Do some examples with portfolio
    Rf = 0.001
    ## INIT Portfolio
    CAPMillo = CCAPM.CAPM(Cartera, period)
    CAPMillo.set_allocation([])
    CAPMillo.set_Rf(Rf)
    
    ## Simulate stupid portolio
    CAPMillo.simulate_Portfolio()

    ## Now lets do a proper portfolio
    # Get the optimal porfolio (short allowed)
    w = CAPMillo.TangentPortfolio(Rf = Rf)
    CAPMillo.set_allocation(w)
    CAPMillo.simulate_Portfolio()

    ## Now change the Rf
    Rf = 0.00
    CAPMillo.set_Rf(Rf)
    w = CAPMillo.TangentPortfolio(Rf = Rf)
    CAPMillo.set_allocation(w)
    CAPMillo.simulate_Portfolio()
    
