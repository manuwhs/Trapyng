
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as ColCon
import scipy.stats as stats

import basicMathlib as bMl
from graph_lib import gl
import utilities_lib as ul

###################################################
########### Functions that have to with alpha beta #####################
########################################################

## Just another way to express the return and variance
## of all the symbols, related to one index.
def set_index(self, symbol_index = -1):
    ## Set the index of CAPM model
    if (type(symbol_index) == type(-1)):
        # If we are given nothing or a number
        # We just stablish the first one
        symbol_index = self.pf.symbols.keys()[0]
        
    self.Sindex = symbol_index
    
def get_indexReturns(self):
    index = self.Sindex  # The index
    ind_ret = self.pf.symbols[index].TDs[self.period].get_timeSeriesReturn()
    return ind_ret
    
def get_indexMeanReturn(self):
    ind_ret = self.get_indexReturns()
    ind_ret = np.mean(ind_ret)
    return ind_ret
    
def get_symbol_ab(self, symbol):
    ## This function outputs the alpha beta a symbol

    index = self.Sindex  # The index
    sym_ret = self.pf.symbols[symbol].TDs[self.period].get_timeSeriesReturn()
    ind_ret = self.get_indexReturns()

#    plt.scatter(ind_ret,sym_ret)
    coeff = bMl.get_linearRef(ind_ret, sym_ret)

    return coeff


def get_all_symbols_ab (self):
    
    symbols = self.pf.symbols.keys()
    coeffs = []
    for sym in symbols:
        coeffs.append(self.get_symbol_ab(sym))
    
    return coeffs
    
def get_portfolio_ab(self, mode = "normal"):
    ### This function gets the alpha beta for the portfolio
    index = self.Sindex
    if (mode == "normal"):
        # We calculate it in a gaussian way
        returns = self.get_PortfolioReturn()
        ind_ret = self.get_indexReturns()
        coeff = bMl.get_linearRef(ind_ret, returns)
        
    if (mode == "gaussian"):
        # We calculate by calculating the individual ones first.
        # The total coefficient is the sum of all coefficients        
        coeffs = np.array(self.get_all_symbols_ab())
        coeff = coeffs.T.dot(self.allocation)
        
    return coeff
    
def get_symbol_JensenAlpha(self, symbol, mode = "normal"):
    ### This function gets the Jensens Alpha of the portolio.
    ## Which is the alpha of the portfolio, taking into account
    ## The risk-free rate. Which is what is everything expected to
    # Grow.
    index = self.Sindex
    coeff = self.get_symbol_ab(symbol)
    beta = coeff[1]
    
#    print "beta = " + str(beta)
    returns = self.get_SymbolReturn(symbol)
    ind_ret = self.get_indexReturns()
    
    # It is the difference between what we obtain and the index
    # Sum of weighted alphas, taking into account the Riskfree Rate
    JensenAlpha = (returns - self.Rf) - beta*(ind_ret - self.Rf)
        
    return JensenAlpha
    
def get_portfolio_JensenAlpha(self, mode = "normal"):
    ### This function gets the Jensens Alpha of the portolio.
    ## Which is the alpha of the portfolio, taking into account
    ## The risk-free rate. Which is what is everything expected to
    # Grow.
    index = self.Sindex
    coeff = self.get_portfolio_ab(mode = mode)
    beta = coeff[1]
    
#    print "beta = " + str(beta)
    returns = self.get_PortfolioReturn()
    ind_ret = self.get_indexReturns()
    
    # It is the difference between what we obtain and the index
    # Sum of weighted alphas, taking into account the Riskfree Rate
    JensenAlpha = (returns - self.Rf) - beta*(ind_ret - self.Rf)
        
    return JensenAlpha

def test_Jensens_Alpha(self, nf = 1):
    # Test the gaussianity and confidence of the alpha.
    residual = self.get_portfolio_JensenAlpha()
    ttest = stats.ttest_1samp(a = residual,  # Sample data
                 popmean = 0)          # Pop mean
    
    print "TESTING PORFOLIO"
    print np.mean(residual), np.std(residual)
    print ttest
    ## Fit a gaussian and plot it
    gl.histogram(residual)

    
def test_symbol_ab(self,symbol, nf = 1):
    ## This function tests that the residuals behaves properly.
    ## That is, that the alpha (how we behave compared to the market)
    ## has a nice gaussian distribution.

    ## Slide 7

    index = self.Sindex  # The index
    sym_ret = self.pf.symbols[symbol].TDs[self.period].get_timeSeriesReturn()
    ind_ret = self.get_indexReturns()

    # Get coefficients for the symbol
    coeffs = self.get_symbol_ab(symbol)
    
    ##### GET THE RESIDUAL
    X = np.concatenate((np.ones((sym_ret.shape[0],1)),sym_ret),axis = 1) 
    pred = X.dot(np.array(coeffs))  # Pred = X * Phi
    pred = pred.reshape(pred.shape[0],1)

    residual = pred - ind_ret
    print "Mean of residual %f" % np.mean(residual)
    
    ### Now we test the residual
    print "Statistical test of residual"
    ttest = stats.ttest_1samp(a = residual,  # Sample data
                 popmean = 0)          # Pop mean
    print ttest
    ######## DOUBLE REGRESSION OF PAGE 7. Early empirical test
    Xres =  np.concatenate((ind_ret,np.power(residual,2)),axis = 1) 
    coeff = bMl.get_linearRef(Xres, sym_ret)
    print "Early empirical test of CAPM is wrong"
    print coeff

    hist, bin_edges = np.histogram(residual, density=True)
    gl.bar(bin_edges[:-1], hist, 
           labels = ["Distribution","Return", "Probability"],
           legend = [symbol],
           alpha = 0.5,
           nf = nf)
    
    ## Lets get some statistics using stats
    m, v, s, k = stats.t.stats(10, moments='mvsk')
    n, (smin, smax), sm, sv, ss, sk = stats.describe(residual)

    print "****** MORE STATISTIC ************"
    print "Mean " + str(sm)
    tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
    pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
    print 't-statistic = %6.3f pvalue = %6.4f' % (tt, pval)
    return coeff
    
def marketTiming(self,returns = [], ind_ret = [], mode = "Treynor-Mazuy"):
    # Investigate if the model is good. 
    # We put a cuatric term of the error.
    
    returns = ul.fnp(returns)
    ind_ret = ul.fnp(ind_ret)
    
    if (returns.size == 0):
        returns = self.get_PortfolioReturn()
    if (ind_ret.size == 0):
        ind_ret = self.get_indexReturns()
    
    # Instead of fitting a line, we fit a parabola, to try to see
    # if we do better than the market return. If when Rm is higher, we have 
    # higher beta, and if when Rm is lower, we have lower beta. So higher
    # and lowr return fitting a curve, cuatric, 

    gl.scatter(ind_ret, returns,
               labels = ["Treynor-Mazuy", "Index Return", "Portfolio Return"],
               legend = ["Returns"])
    
    ## Linear regression:
    Xres =  ind_ret
    coeffs = bMl.get_linearRef(Xres, returns)
    
    Npoints = 10000
    x_grid = np.array(range(Npoints))/float(Npoints)
    x_grid = x_grid*(max(ind_ret) - min(ind_ret)) +  min(ind_ret)
    x_grid = x_grid.reshape(Npoints,1)
    
    x_grid_2 = np.concatenate((np.ones((Npoints,1)),x_grid), axis = 1)
    y_grid = x_grid_2.dot(np.array(coeffs)) 
    
    gl.plot(x_grid, y_grid, legend = ["Linear Regression"], nf = 0)

    
    Xres =  np.concatenate((ind_ret,np.power(ind_ret,2)),axis = 1) 
    coeffs = bMl.get_linearRef(Xres, returns)
    
    x_grid_2 = np.concatenate((np.ones((Npoints,1)),x_grid,np.power(x_grid,2).reshape(Npoints,1) ),axis = 1) 
    y_grid = x_grid_2.dot(np.array(coeffs)) 
    
#    print y_grid.shape
    
    gl.plot(x_grid, y_grid, legend = ["Quadratic Regression"], nf = 0)
    
    print coeffs
    return 1

def get_residuals_ab(self):
    
    # For histogram
    import pylab 
    import scipy.stats as stats
    
    measurements = np.random.normal(loc = 20, scale = 5, size=100)   
    stats.probplot(measurements, dist="norm", plot=pylab)
    pylab.show()
    
def plot_portfoliocorrab(self, nf = 1):
    # This function plots the returns of a symbol compared
    # to the index, and computes the regresion and correlation parameters.
    
    index = self.Sindex  # The index
    sym_ret = self.get_PortfolioReturn()
    ind_ret = self.get_indexReturns()

    # Mean and covariance
    data = np.concatenate((sym_ret,ind_ret),axis = 1)
    means = np.mean(data, axis = 0)
    cov = np.cov(data)
    
    # Regression
    coeffs = bMl.get_linearRef(ind_ret, sym_ret)

    gl.scatter(ind_ret, sym_ret,
               labels = ["Gaussianity study", "Index: " + self.Sindex,"Porfolio"],
               legend = ["Returns"],
                nf = nf)
    
    ## Linear regression:
    Xres =  ind_ret
    coeffs = bMl.get_linearRef(Xres, sym_ret)
    
    Npoints = 10000
    x_grid = np.array(range(Npoints))/float(Npoints)
    x_grid = x_grid*(max(ind_ret) - min(ind_ret)) +  min(ind_ret)
    x_grid = x_grid.reshape(Npoints,1)
    
    x_grid_2 = np.concatenate((np.ones((Npoints,1)),x_grid), axis = 1)
    y_grid = x_grid_2.dot(np.array(coeffs)) 
    
    gl.plot(x_grid, y_grid, 
            legend = ["b: %.2f ,a: %.2f" % (coeffs[1], coeffs[0])], 
            nf = 0)
            
def plot_corrab(self, symbol, nf = 1):
    # This function plots the returns of a symbol compared
    # to the index, and computes the regresion and correlation parameters.
    
    index = self.Sindex  # The index
    sym_ret = self.pf.symbols[symbol].TDs[self.period].get_timeSeriesReturn()
    ind_ret = self.get_indexReturns()

    # Mean and covariance
    data = np.concatenate((sym_ret,ind_ret),axis = 1)
    means = np.mean(data, axis = 0)
    cov = np.cov(data)
    
    # Regression
    coeffs = bMl.get_linearRef(ind_ret, sym_ret)

    gl.scatter(ind_ret, sym_ret,
               labels = ["Gaussianity study", "Index: " + self.Sindex,symbol],
               legend = ["Returns"],
               nf = nf)
    
    ## Linear regression:
    Xres =  ind_ret
    coeffs = bMl.get_linearRef(Xres, sym_ret)
    
    Npoints = 10000
    x_grid = np.array(range(Npoints))/float(Npoints)
    x_grid = x_grid*(max(ind_ret) - min(ind_ret)) +  min(ind_ret)
    x_grid = x_grid.reshape(Npoints,1)
    
    x_grid_2 = np.concatenate((np.ones((Npoints,1)),x_grid), axis = 1)
    y_grid = x_grid_2.dot(np.array(coeffs)) 
    
    gl.plot(x_grid, y_grid, 
            legend = ["b: %.2f ,a: %.2f" % (coeffs[1], coeffs[0])], 
            nf = 0)
