import CBond as CBond
import bond_math as ba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as ColCon
import scipy.stats as stats
import datetime as dt
import basicMathlib as bMl
import utilities_lib as ul
from graph_lib import gl
###############################################################
########### Introduction to Financial Enginierind DTU  ########
###############################################################

## Set of function badly done in order to meet the requests of
## the final project

folder_images = "./pics/"
def IFE_a(self, year_start = 1996, year_finish = 2016, window = 10):
    ## Basic, just look at the bloody graphs
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_finish,1,1))
    
    dates = self.get_dates()
    prices = self.pf.get_timeSeries(self.period)
    returns = self.get_Returns()
#    print returns.shape
    gl.plot(dates, prices,
            labels = ["Monthly price of Symbols", "Time (years)", "Price (dolar)"],
            legend = self.pf.symbols.keys(), loc = 2)   
    gl.savefig(folder_images +'pricesAll.png',
               dpi = 150, sizeInches = [2*8, 1.5*6])

    gl.plot(dates, returns,
            labels = ["Monthly return of the Symbols", "Time (years)", "Return (%)"],
            legend = self.pf.symbols.keys())   
    gl.savefig(folder_images +'returnsAll.png', 
               dpi = 150, sizeInches = [2*8, 1.5*6])

    ## Distribution obtaining
    gl.set_subplots(2,2)
    for i in range(4):
        gl.histogram(returns[:,i], labels = [self.symbol_names[i]])
    
    gl.savefig(folder_images +'returnDistribution.png',
               dpi = 150, sizeInches = [2*8, 1.5*6])

    ##############  Posible Transformations ##################

    ws = [3, 4, 6, 8]
    
    gl.set_subplots(2,2)
    for w in ws:
        means, ranges = bMl.get_meanRange(prices[:,1], w)
        gl.scatter(means, ranges, lw = 4,
                   labels = ["", "mean","range"],
                   legend = ["w = %i" %(w)])
                   
    gl.savefig(folder_images +'rangeMean.png',
               dpi = 150, sizeInches = [2*8, 1.5*6])

    
def IFE_b(self,year_start = 1996, year_finish = 2016, window = 10):
    ## Question b of the asqued thing
    
    all_returns = []
    all_covMatrices = []
    
    all_dates = []  # To store the dates of the estimation
    for year_test in range(year_start,year_finish - window + 1): # +1 !!
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
        
        ret = self.yearly_Return(self.get_MeanReturns())
        covMat = self.yearly_covMatrix(self.get_covMatrix())
    
        all_covMatrices.append(covMat)
        all_returns.append(ret)
        
        # Get the dates from any of the symbols of the portfolio
        dates = self.get_dates()
        all_dates.append(dates[-1])
        
    ## Plotting the returns
    all_returns = np.array(all_returns)

#    gl.plot(all_dates, all_returns[:,0],
#            labels = ["Returns", "Time", "Return"],
#            legend = [self.pf.symbols.keys()[0]])
#            
#    gl.plot(all_dates, all_returns[:,1],
#            legend = [self.pf.symbols.keys()[1]], nf = 0, na = 0)

    ## 1) Plot the returns of all of them together for the eleven windows
    gl.plot(all_dates, all_returns,
            labels = ["Average Return in 10 years", "Time (years)", "Anual return of Assets"],
            legend = self.symbol_names) 

    gl.savefig(folder_images +'returnsAveAll.png', 
               dpi = 150, sizeInches = [2*8, 1.5*6])

    ## 2) Plot the covariance matrix for 9 years
    gl.set_subplots(2,3)
    
    for i in range(6):
        gl.bar_3D(self.symbol_names, self.symbol_names, all_covMatrices[i],
                  labels = [str(year_start +window+i),"",""],
                   fontsize = 30, fontsizeA = 19)    

    gl.savefig(folder_images +'covsAveAll.png', 
               dpi = 80, sizeInches = [4*8, 3*6])


def IFE_c (self, Rf = 0,year_start = 1996, year_finish = 2016, window = 10):
    ## With monthly data, calculate the Efficient frontier
    ## year by year. So lets do it

    self.set_Rf(Rf)
    
    nf_flag = 1
    all_portfolios = []
    for year_test in range(year_start,year_finish - window + 1): # +1 !!
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
#        print self.get_Returns()[1]  # Check that it works

#        portfolios = self.Lintnerian_efficient_frontier(norm = "none", maxRf = 0.00031)
#        optimal, portfolios = self.efficient_frontier(kind = "Markowitz")
#        optimal, portfolios = self.efficient_frontier(kind = "Normal")
        optimal, portfolios = self.efficient_frontier(kind = "Tangent")
        all_portfolios.append(portfolios)
    
        self.plot_allocations(portfolios, labels = ["Efficient Frontiers", "Risk (std)", "Return (%)"],
                              legend = ["Frontier " + str(year_test + window)], nf = nf_flag)
        nf_flag = 0
    
    gl.savefig(folder_images +'effAll.png', 
               dpi = 150, sizeInches = [2*8, 2*6])


def IFE_d (self, Rf = 0.01, Rfs_list = [0], year_start = 1996, year_finish = 2016, window = 10):
    ### The official one can be done executing the exercise c with another Rf
    ## Just another graph to show that now we should not use all the money.
    ## The efficient frontier is not going to change.
    ## Only the market line. But we exexute IFE_c again with the new Rf 
    ## And plot some market lines !!
#    self.pf.set_interval(dt.datetime(1996,12,5),dt.datetime(2016,2,21))
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_finish,1,1))
    # Just plot some tangeny lines to the portfolio !!
    ## First plot some data !!
    Nalloc = 100000
    
    self.set_Rf(Rf)
    alloc = self.get_random_allocations(Nalloc, short = "yes", mode = "gaussian")
    self.scatter_allocations(alloc, alpha = 0.3,nf = 1)
    
    # Get upper limit of std to plot market lines
    w = self.TangentPortfolio(Rf = Rf)
    self.set_allocation(w)
    stdR = self.get_PortfolioStd()
    
    Optimal_portfolios = []
    for Rf in Rfs_list:
        bias, slope = self.Market_line (Rf = Rf)
        Optimal_portfolios.append(self.TangentPortfolio(Rf = Rf))
        gl.plot([0,4*stdR],[bias, bias + slope*4*stdR],
                legend = ["Mkt Line Rf: %0.3f, SR:%0.2f" % (Rf,slope)],
                nf = 0,loc = 2)


    optimal, portfolios = self.efficient_frontier(kind = "Tangent", max_exp = 20)
    self.plot_allocations(portfolios, nf = 0, lw = 4, color = "k", legend = ["Efficient Frontier"])

    self.scatter_allocations(np.eye(self.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)
    
    self.scatter_allocations(Optimal_portfolios, 
            legend = ["Optimal portfollios"], nf = 0, alpha = 1.0, lw = 5)
    
    gl.savefig(folder_images +'marketLines.png', 
               dpi = 150, sizeInches = [2*8, 2*6])


    ### Only one market line
    Rf = 0
    bias, slope = self.Market_line (Rf = Rf)
    gl.plot([0,4*stdR],[bias, bias + slope*4*stdR],
            legend = ["Mkt Line Rf: %0.3f, SR:%0.2f" % (Rf,slope)],
            nf = 1,loc = 2)
        
    optimal, portfolios = self.efficient_frontier(kind = "Tangent", max_exp = 20)
    self.plot_allocations(portfolios, nf = 0, lw = 4, color = "k", legend = ["Efficient Frontier"])

    self.scatter_allocations(np.eye(self.Nsym), 
            legend = ["Assets"], nf = 0, alpha = 1.0, lw = 5)
            
    self.scatter_allocations([Optimal_portfolios[2]],
            legend = ["Optimal portfollios"], nf = 0, alpha = 1.0, lw = 5)
    
    gl.savefig(folder_images +'marketLine.png', 
               dpi = 150, sizeInches = [2*8, 2*6])


def IFE_e (self, ObjectiveR = 0.003, Rf = 0.0, year_start = 1996, year_finish = 2016, window = 10):
    # Just, choose a desired return,
    # Using training Samples calculate using the market line
    # the optimal porfolio for that.
    # Then, using also the last year ( test), recalculate the portfolio needed
    # for that return, and the difference between is the turnover
    self.set_Rf(Rf)

    nf_flag = 1
    desired_Portfolios = []
    all_dates = []
    for year_test in range(year_start,year_finish - window + 1): # +1 !!
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
        
        # Obtain the market line !!
        w = self.TangentPortfolio(Rf = Rf) # Obtain allocation
        # Obtain the expected return and std when using all our money !
        self.set_allocation(w)
        expRet, stdRet = self.get_metrics (investRf = "no")
        param = bMl.obtain_equation_line(Rf, expRet, stdRet)
        bias, slope = param
    
        # Once we have the equation of the line, we obtain how much money
        # we need to use to reach the desired Expecred Return.
        # Rt = (1 - X)Rf + XRp with X = sum(w)
        # For a desired Rt we solve the X

        X = (ObjectiveR - Rf)/(expRet - Rf)
        
#        print X
        # So the desired porfolio is:
        wdesired = w*X
        desired_Portfolios.append(wdesired)
        
        gl.plot([0,1.3*abs(X*stdRet)],[bias, bias + 1.3*abs(slope*stdRet*X)],
            labels = ["Desired Portfolios", "Risk (std)", "Return (%)"],
            legend = ["%s, X: %0.3f" %((year_test + window ), X[0])],
            nf = nf_flag, loc = 2)
        nf_flag = 0
        gl.scatter([abs(X*stdRet)],[ObjectiveR],
            nf = 0)


        dates = self.get_dates()
        all_dates.append(dates[-1])
#        print wdesired

    gl.savefig(folder_images +'desiredPortfolios.png', 
               dpi = 150, sizeInches = [2*8, 2*6])

    # Now we calculate the turnovers 
    Turnovers = []
    prev_abs_alloc = []  # Previous, absolute allocation
    percentaje_changed = []
    Nport = len(desired_Portfolios)
    
    for i in range(Nport-1):
        to = bMl.get_TurnOver(desired_Portfolios[i], desired_Portfolios[i+1])
        Turnovers.append(to)
        prev_abs_alloc.append(np.sum(np.abs(desired_Portfolios[i])))
        percentaje_changed.append(Turnovers[-1]/prev_abs_alloc[-1])
        print Turnovers
    
    gl.set_subplots(1,3)
    
    gl.bar(all_dates[1:], Turnovers, color = "g",
           labels = ["Portfolio turnovers", "Year","Value"])
    
    gl.add_text([all_dates[1:][3],max(Turnovers)*0.80], 
                 "Mean: %0.2f" % np.mean(Turnovers), 30)

    gl.bar(all_dates[0:-1], prev_abs_alloc, color = "r",
           labels = ["Absolute allocations", "Year","Value"])
    
    gl.bar(all_dates[1:], percentaje_changed,  color = "b",
           labels = ["Percentage turnover", "Year","Value"])
    
    gl.add_text([all_dates[1:][3],max(percentaje_changed)*0.80], 
                 "Mean: %0.2f" % np.mean(percentaje_changed), 30)

    gl.savefig(folder_images +'turnovers.png', 
               dpi = 150, sizeInches = [2*8, 1*6])
               
def IFE_f (self, ObjectiveR = 0.003, Rf = 0.0, year_start = 1996, year_finish = 2016, window = 10):
    ### The official one can be done executing the exercise c with another Rf
    ## Just another graph to show that now we should not use all the data.

    # Just, choose a desired return,
    # Using training Samples calculate using the market line
    # the optimal porfolio for that.
    # Then calculate for the next year, the real return
    # for that portfolio. 
    # Do this for several years as well.
    self.set_Rf(Rf)
    
    nf_flag = 1
    
    All_stds = []
    PortfolioReturns = []
    IndexReturns = []
    all_dates = []
    for year_test in range(year_start,year_finish - window + 1 - 1): # +1 !!
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
        
        # Obtain the market line !!
        w = self.TangentPortfolio(Rf = Rf) # Obtain allocation
        self.set_allocation(w)
        # Obtain the expected return and std when using all our money !
        expRet, stdRet = self.get_metrics (investRf = "no")
        param = bMl.obtain_equation_line(Rf, expRet, stdRet)
        bias, slope = param
        X = (ObjectiveR - Rf)/(expRet - Rf)
        wdesired = w*X

        ## Check that the output of this portfolio is the desired one.
        self.set_allocation(wdesired)  # Set the allocation
        expRet, stdRet = self.get_metrics()  # Get the expected return for that year
       
#        print ret 
        ## Now that we have the desired w*X, we will calculate the resturn of
        ## the portfolio in the following year.
        # To do so, we set the dates, only to the next year, set the portfolio allocation
        # And calculate the yearly expected return !!

        # Set the dates to only the next year !!
        # Also, one month before in order to get the returns of the first month.
        self.pf.set_interval(dt.datetime(year_test + window,1,1),dt.datetime(year_test + window + 1,1,1))
        self.set_allocation(wdesired)  # Set the allocation
        expRet, stdRet = self.get_metrics()  # Get the expected return for that year
        PortfolioRet = self.yearly_Return(expRet)  # Get yearly returns
        PortfolioReturns.append(PortfolioRet)
        
        All_stds.append(self.yearly_covMatrix(stdRet))
        
        indexRet = self.get_indexMeanReturn()
        indexRet = self.yearly_Return(indexRet)
        IndexReturns.append(indexRet)
        
#        dates = self.get_dates()
        all_dates.append(year_test + window + 1)
        
        ## Graph with the evolutio of the portfolio price after the assignment
        gl.plot(range(1,13), np.cumsum(self.get_PortfolioReturn()),
                nf = nf_flag, 
                labels = ["Evolution of returns by month", "Months passed", "Cumulative Return"],
                legend = [str(year_test + window +1)])
        nf_flag = 0
#        print ret

    gl.savefig(folder_images +'returnsEvolMonth.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
    
    ## Graph with the desired, the obtained returns and the returns of the index
    gl.bar(all_dates[:], IndexReturns, 
            labels = ["Obtained returns", "Time (years)", "Return (%)"],
            legend = ["Index Return"],
            alpha = 0.8,
            nf = 1)
    gl.bar(all_dates[:], PortfolioReturns, 
           labels = ["Returns of year", "Year","Value"],
            legend = ["Porfolio Return"],
            alpha = 0.8,
            nf = 0)
            
    gl.scatter(all_dates[:], self.yearly_Return(ObjectiveR) * np.ones((len(all_dates[:]),1)), 
            legend = ["Objective Return"],
            nf = 0)

    gl.scatter(all_dates[:], All_stds, 
            legend = ["Std of the portfolio return"],
            nf = 0)
            
    gl.savefig(folder_images +'returnsEvolYears.png', 
               dpi = 150, sizeInches = [2*8, 2*6])

    #### Crazy idea !! Lets plot where the fucking efficient frontier went 
    nf_flag = 1
    PortfolioReturns = []
    IndexReturns = []
    all_dates = []
    gl.set_subplots(2,3)
    for year_test in range(year_start,year_start + 6): # +1 !!
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
        optimal, portfolios = self.efficient_frontier(kind = "Tangent")
        self.plot_allocations(portfolios, labels = ["Evolution of the efficient frontier"],
                              legend = ["Frontier " + str(year_test + window) + " before"], color = "k", nf = 1)
 
        self.pf.set_interval(dt.datetime(year_test + window,1,1),dt.datetime(year_test + window + 1,1,1))
        self.set_allocation(self.TangentPortfolio(Rf = Rf))
        self.plot_allocations(portfolios, legend = ["Frontier " + str(year_test + window) + " after"], color = "r",nf = 0)
        
    gl.savefig(folder_images +'effEvol.png', 
               dpi = 80, sizeInches = [4*8, 3*6])
def IFE_f3 (self, ObjectiveRlist = [0.003], Rf = 0.0, year_start = 1996, year_finish = 2016, window = 1):
    ### The official one can be done executing the exercise c with another Rf
    ## Just another graph to show that now we should not use all the data.

    # Just, choose a desired return,
    # Using training Samples calculate using the market line
    # the optimal porfolio for that.
    # Then calculate for the next year, the real return
    # for that portfolio. 
    # Do this for several years as well.

    self.set_Rf(Rf)
    
    All_returns  = []
    All_vars = []
    
    optimal, portfolios = self.efficient_frontier(kind = "Tangent")
    nport = len(portfolios)
    
    portlist = range(0,nport,nport/100)
    
    ObjectiveR = 0.03
    for iport in portlist:
        PortfolioReturns = []
        all_dates = []
        for year_test in range(year_start,year_finish - window + 1 - 1): # +1 !!
            # Set the dates
            self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
            
            # Obtain the market line !!
            w = portfolios[iport]
            self.set_allocation(w)
            # Obtain the expected return and std when using all our money !
            expRet, stdRet = self.get_metrics (investRf = "no")
            param = bMl.obtain_equation_line(Rf, expRet, stdRet)
            bias, slope = param
            X = (ObjectiveR - Rf)/(expRet - Rf)
            wdesired = w*X
    
            self.pf.set_interval(dt.datetime(year_test + window,1,1),dt.datetime(year_test + window + 1,1,1))
            self.set_allocation(wdesired)  # Set the allocation
            expRet, stdRet = self.get_metrics()  # Get the expected return for that year
            PortfolioRet = self.yearly_Return(expRet)  # Get yearly returns
            PortfolioReturns.append(PortfolioRet)
            
            dates = self.get_dates()
            all_dates.append(dates[0])
        
        All_returns.append(np.mean(PortfolioReturns))
        All_vars.append(np.std(PortfolioReturns)/np.sqrt(np.sqrt(12*12)))
#    All_returns = np.array(All_returns).reshape(len(ObjectiveRlist),10)
#    print All_returns
    All_means = All_returns
    print All_returns
#    All_means = np.mean(All_returns, axis = 1)
    print ul.fnp(All_returns).shape
    print All_means
#    print All_means - ObjectiveRlist
#    All_means = np.divide((All_means - ObjectiveRlist),ObjectiveRlist)
#    print All_means
    ## Graph with the desired, the obtained returns and the returns of the index
    gl.bar(portlist, All_means, 
            labels = ["Obtained returns", "Time (years)", "Return (%)"],
            legend = ["Index Return"],
            alpha = 0.8,
            nf = 1)

    gl.plot(portlist, All_vars, 
            labels = ["Obtained returns", "Time (years)", "Return (%)"],
            legend = ["Index Return"],
            alpha = 0.8,
            nf = 0)
            
    gl.savefig(folder_images +'best_Objective.png', 
               dpi = 150, sizeInches = [2*8, 2*6])


def IFE_f2 (self, ObjectiveRlist = [0.003], Rf = 0.0, year_start = 1996, year_finish = 2016, window = 10):
    ### The official one can be done executing the exercise c with another Rf
    ## Just another graph to show that now we should not use all the data.

    # Just, choose a desired return,
    # Using training Samples calculate using the market line
    # the optimal porfolio for that.
    # Then calculate for the next year, the real return
    # for that portfolio. 
    # Do this for several years as well.

    self.set_Rf(Rf)
    
    All_returns  = []
    All_vars = []
    
    windowslist = range(1,13)
    ObjectiveR = 0.03
    for window in windowslist:
        PortfolioReturns = []
        all_dates = []
        for year_test in range(year_start,year_finish - window + 1 - 1): # +1 !!
            # Set the dates
            self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
            
            # Obtain the market line !!
            w = self.TangentPortfolio(Rf = Rf) # Obtain allocation
            self.set_allocation(w)
            # Obtain the expected return and std when using all our money !
            expRet, stdRet = self.get_metrics (investRf = "no")
            param = bMl.obtain_equation_line(Rf, expRet, stdRet)
            bias, slope = param
            X = (ObjectiveR - Rf)/(expRet - Rf)
            wdesired = w*X
    
            self.pf.set_interval(dt.datetime(year_test + window,1,1),dt.datetime(year_test + window + 1,1,1))
            self.set_allocation(wdesired)  # Set the allocation
            expRet, stdRet = self.get_metrics()  # Get the expected return for that year
            PortfolioRet = self.yearly_Return(expRet)  # Get yearly returns
            PortfolioReturns.append(PortfolioRet)
            
            dates = self.get_dates()
            all_dates.append(dates[0])
        
        All_returns.append(np.mean(PortfolioReturns))
        All_vars.append(np.std(PortfolioReturns)/np.sqrt(np.sqrt(12*12)))
#    All_returns = np.array(All_returns).reshape(len(ObjectiveRlist),10)
#    print All_returns
    All_means = All_returns
    print All_returns
#    All_means = np.mean(All_returns, axis = 1)
    print ul.fnp(All_returns).shape
    print All_means
#    print All_means - ObjectiveRlist
#    All_means = np.divide((All_means - ObjectiveRlist),ObjectiveRlist)
#    print All_means
    ## Graph with the desired, the obtained returns and the returns of the index
    gl.bar(windowslist, All_means, 
            labels = ["Obtained returns", "Time (years)", "Return (%)"],
            legend = ["Index Return"],
            alpha = 0.8,
            nf = 1)

    gl.plot(windowslist, All_vars, 
            labels = ["Obtained returns", "Time (years)", "Return (%)"],
            legend = ["Index Return"],
            alpha = 0.8,
            nf = 0)
            
    gl.savefig(folder_images +'best_Objective.png', 
               dpi = 150, sizeInches = [2*8, 2*6])

def IFE_g (self, Rf = 0, year_start = 1996, year_finish = 2016, window = 10):
    ## CAPM model question, calculate abs and doubt everything you know

    self.set_Rf(Rf)
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_finish,1,1))
    
    
   # Plot the correlation between some index and the stock
    gl.set_subplots(2,3)
    
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_start + window,1,1))
    for i in range(6):
        self.plot_corrab(self.symbol_names[i])    
        
    gl.savefig(folder_images +'SymbolAB.png', 
               dpi = 80, sizeInches = [2*8, 2*6])

   # Plot the jensen alpha of some of the stocks
    gl.set_subplots(2,3)
    
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_start + window,1,1))
    for i in range(6):
        JensenAlpha = self.get_symbol_JensenAlpha(self.symbol_names[i])
        gl.histogram(JensenAlpha, labels = [self.symbol_names[i]])  
        
    gl.savefig(folder_images +'JensenAlphasAll.png', 
               dpi = 80, sizeInches = [4*8, 3*6])

    
    ## We set a stupid initial portfolio (Everything equal)
    param = self.get_symbol_ab(self.symbol_names[1])
    print "Params of %s" % self.symbol_names[1]
    print param
    
    ########## TEST ONE SYMBOL ######
#    self.test_symbol_ab(self.symbol_names[1])
    # Print stupid portfolio
    # Param
    params = self.get_all_symbols_ab()
    print "All params"
    print params
    
    # Params of stupid porfolio
    print "Params of stupid portfolio"
    self.set_allocation([])
    param = self.get_portfolio_ab(mode = "normal")    # Obtained as definition
    print param
    param = self.get_portfolio_ab(mode = "gaussian")  # Obtained first getting the cov matrix
    print param
    
    ########## TEST Portfolio ######
    # Test the jensenAlpha of the portfolio
    JensenAlpha = self.get_portfolio_JensenAlpha()
    
    ## IDEA !! Maybe use the portfolio in the frontier that maximizes
    ## the alpha and minimizes the beta !!! Maybe minimizing beta is not as important
    ## In the CAMP we already have the total Exp and risk.
    ## Alpha and beta say: Does out portolio perform better than the market ?
    ## If we just follow the market, investing everything on the index,
    ## Thus investing in everything proportionally to their capital.
    ## Then we have alpha = 0 and beta = 1 
#    CAPMillo.test_symbol_ab(symbols[2])
 
#   Plot random porfolios correlation with the index
    alloc = self.get_random_allocations(100, short = "yes", mode = "gaussian")
    gl.set_subplots(2,3)
    for i in range(6):
        self.set_allocation(alloc[i])
        self.plot_portfoliocorrab( nf = 1)

    gl.savefig(folder_images +'randomPortCorr.png', 
               dpi = 150, sizeInches = [2*8, 2*6])

#   Plot Jesen Alpha for random portfolios      
    flag_nf = 1
    for i in range(5):
        self.set_allocation(alloc[i])
        self.test_Jensens_Alpha(nf = flag_nf)
        flag_nf = 0
        
    gl.savefig(folder_images +'randomPortJA.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
               
    ##############################################
    ########### ANALIZE 3 optimal portfolios #####
    ##############################################
    Rfs = [0,0.002, 0.0031]
    print "???????????????:?:::::::::::::::::::::::::::::::::::::::"
    flag_nf = 1
    for Rf in Rfs:
        # Do it again with an optimal portolio
        w = self.TangentPortfolio(Rf = Rf)
        self.set_allocation(w)
        self.test_Jensens_Alpha(nf = flag_nf)
        flag_nf = 0
        
    flag_nf = 1
    
    gl.savefig(folder_images +'optimalPortJA.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
    gl.set_subplots(1,3)
    for Rf in Rfs:
        # Do it again with an optimal portolio
        w = self.TangentPortfolio(Rf = Rf)
        self.set_allocation(w)
        self.plot_portfoliocorrab(nf = 1)
        flag_nf = 0

    gl.savefig(folder_images +'optimalPortCorr.png', 
               dpi = 150, sizeInches = [2*8, 1*6])
               
def IFE_h (self, Rf = 0, mktcap = [], year_start = 1996, year_finish = 2016, window = 10):
    ## Black litterman question !!
    # The optimal portolio, lets say is the one given by Markovitz
    # mktcap is a dicktionary with the market capitalizaion of the equities
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_finish,1,1))
    
    ## Get the actual stuff !!

    ExpRet = self.get_MeanReturns()
    Sigma = self.get_covMatrix()
    woptimal = self.TangentPortfolio()
    self.set_allocation(woptimal)
    R,S = self.get_metrics()
    delta = (R - self.Rf)/np.power(S,2)  # Optimal risk adversion
    
    
    ## Get the weights by the market capitalization
    if (len(mktcap) > 0):
        weq = []
        for sym in self.symbol_names:
            weq.append(mktcap[sym])
        
        weq = ul.fnp(weq) /np.sum(weq)
        weq = weq.T.tolist()[0]
#        print weq
    else:
        
        weq = woptimal  # Initial prior
    ############### PUT FECKING BL prior instead ##########
    # Calculate initial portfolio from the market capitalization
    # Risk aversion of the market. We say it is the one of the portfolio
    # The optimal portfolio is the market.
#    weq = np.ones((1,self.Nsym))/self.Nsym
#    weq = weq.tolist()[0]
    # Coefficient of uncertainty in the prior estimate of the mean
    tau = 10
    
    ### Prior of our Views !!!
    P1 = np.zeros((2,self.Nsym))
    P1[0,0] = -1; P1[0,1] =  1
    P1[1,1] = -1; P1[1,2] =  1
    P1 = ul.fnp(P1)
    
    # If we invert P1 and Q1 at the same time we get the same
    Q1 = [0.0002, 0.0001]
    Q1 = ul.fnp(Q1)
    
    Omega1 = np.dot(np.dot(P1,Sigma),P1.T) * np.eye(Q1.shape[0])
    
    postPi,weqpost = self.BlackLitterman(weq, Sigma, delta, # Prior portfolio variables
                   tau,              # Uncertainty coefficient of the porfolio priors
                   P1, Q1, Omega1)   # Prior views variables
    
    # Reference returns of the portfolio of the market
    # They can just be calculated using the portfolio
      
    # A priory the expected return Posteriori does not have to be bigger
    # Just more accuarate to reality if our views are right :)
    refPi = delta * np.dot(Sigma, weq)  
      
    Ereturn = np.dot(refPi,weq)
    EreturnPost = np.dot(postPi,weqpost)
    

    ## Plot the returns !!! 
    # We will plot the real w returns, the Pi Returns, and the Post- Returns
    gl.set_subplots(2,3)
    gl.bar(self.pf.symbols.keys(),ExpRet,
           labels = ["Optimal initial returns"])    
    gl.bar(self.pf.symbols.keys(),refPi,
             labels = ["Prior Returns"])
    gl.bar(self.pf.symbols.keys(),postPi,
             labels = ["Posterior Returns"])

#    gl.savefig(folder_images +'returnsBL.png', 
#               dpi = 150, sizeInches = [2*8, 2*6])
               
    ## Plot the weights !!! 
    # We will plot the real w returns, the Pi Returns, and the Post- Returns
#    gl.set_subplots(1,3)
    
    gl.bar(self.pf.symbols.keys(),woptimal,
           labels = ["Optimal intial weights"])
    gl.bar(self.pf.symbols.keys(),weq,
             labels = ["Prior Weights"])
    gl.bar(self.pf.symbols.keys(),weqpost,
             labels = ["Posterior Weights"])
             
#    gl.savefig(folder_images +'weightsBL.png', 
#               dpi = 150, sizeInches = [2*8, 2*6])
               
    gl.savefig(folder_images +'weightsreturnsBL.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
               
    pass

def IFE_i (self, Rf = 0.0,  year_start = 1996, year_finish = 2016, window = 10):
    ### Timing. Check if when the market had big return, we incresed the beta (higher return)
    ## And when the market had negative return, we have not so bad return
    ## The way to do this is to perform a cuatratic curve fit.
    self.pf.set_interval(dt.datetime(year_start,1,1),dt.datetime(year_finish,1,1))
    print self.marketTiming()
    
    gl.savefig(folder_images +'timingPosteriori.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
               
    self.set_Rf(Rf)
    
    obtained_returns = []
    index_returns = []
    
    for year_test in range(year_start,year_finish - window): # 
        # Set the dates
        self.pf.set_interval(dt.datetime(year_test,1,1),dt.datetime(year_test + window,1,1))
        # Obtain the market line !!
        w = self.TangentPortfolio(Rf = Rf) # Obtain allocation
        self.set_allocation(w)

        # Once the model is found, we obtain the returns of the next year
        self.pf.set_interval(dt.datetime(year_test + window,1,1),dt.datetime(year_test + window + 1,1,1))
#        self.pf.set_interval(dt.datetime(fin_year,1,1),dt.datetime(fin_year +1,1,1))
        
        returns = self.get_PortfolioReturn()  # Get the expected return for that year
#        dates =  self.get_dates()
#        print returns.shape
#        print returns.T.tolist()[0]
        obtained_returns.extend(returns.T.tolist()[0])
        index_returns.extend(self.get_indexReturns().T.tolist()[0])
        
    obtained_returns = np.array(obtained_returns)
    index_returns = np.array(index_returns)
    print self.marketTiming(obtained_returns, index_returns)
    
    gl.savefig(folder_images +'timingBacktest.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
    
def IFE_2a(self):  
    ### Bond question !! 
    myBond = CBond.CBOND( name = "hola", freq = 2, coupon = 5.75, par = 100.)
    # Set some properties 
    myBond.set_price(95.0428)
    myBond.set_timeToMaturity(2.5)
    myBond.set_ytm(0.10)
    
    # Plot the compound price understanding 
    ba.plot_compound_understanding()
    gl.savefig(folder_images +'compoundUnders.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
               
    myBond.yieldPriceStudy(90)
    gl.savefig(folder_images +'yieldCurve.png', 
               dpi = 150, sizeInches = [2*8, 2*6])    
def IFE_2b(self):  
    ### Calculate the convexity and duration of the bonds and then calculate 
    # the portfolio one by adding the weighted. (It is just weighted average
    # of prices. Nothing more, nothing less.)
    myBond = CBond.CBOND( name = "hola", freq = 2, coupon = 5.75, par = 100.)
    # Set some properties 
    myBond.set_price(95.0428)
    myBond.set_timeToMaturity(2.5)
    myBond.set_ytm(0.10)
    
    # Plot the compound price understanding 
    ba.plot_compound_understanding()
    gl.savefig(folder_images +'compoundUnders.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
               
               
def IFE_2c(self):  
    ### Bond question !! 
    myBond = CBond.CBOND( name = "hola", freq = 2, coupon = 5.75, par = 100.)
    # Set some properties 
    myBond.set_price(95.0428)
    myBond.set_timeToMaturity(2.5)
    myBond.set_ytm(0.10)
    
    # Plot the compound price understanding 
    ba.plot_compound_understanding()
    gl.savefig(folder_images +'compoundUnders.png', 
               dpi = 150, sizeInches = [2*8, 2*6])
    

    
    