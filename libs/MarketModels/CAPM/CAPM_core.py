
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graph_lib import gl
import matplotlib.colors as ColCon
import utilities_lib as ul
import basicMathlib as bMl

############################################
########### CORE FUNC #####################
###########################################

def set_seriesNames(self,seriesNames = ["AdjClose"]):
    # Function to set the seriesNames for all the timeSeries
    # of the symbols !!
    
    for symbol_n in self.pf.symbol_names:
        symbol = self.pf.symbols[symbol_n]
        symbol.TDs[self.period].set_seriesNames(seriesNames)
    
def set_allocation(self, allocation):
    self.allocation = allocation
    if (len(allocation) == 0):
        allo = np.ones((1,self.Nsym))/self.Nsym
        allo = allo.tolist()[0]
        self.allocation = allo

def set_interval(self,sdate,edate):
    self.pf.set_interval(sdate, edate)
    
def get_dates(self):
    dates = self.pf.get_dates(self.pf.symbols.keys()[0],self.period)
    return dates
    
def get_allocation(self):
    return self.allocation

def set_Rf(self, Rf = 0):
    self.Rf = Rf    # Risk_free rate
    
def initVariablesPorfolio(self, pf):
    self.returns = np.array([])
    self.covMatrix = np.array([])
    self.pf = pf
    self.symbol_names = self.pf.symbols.keys()
    self.Nsym = len(self.pf.symbols.keys())
    self.Rf = 0    # Risk_free rate
    self.set_index()
    
def get_Returns(self):
    # This function gets the returns of all the individual
    # symbols we have in stock

    # If we have not calculated the returns

    #if (self.returns.shape[0] == 0):
    self.returns = self.pf.get_Returns(self.period)
        
    return self.returns
def get_SymbolReturn(self, symbol):
    return self.pf.symbols[symbol].TDs[self.period].get_timeSeriesReturn();
    
def get_MeanReturns(self):
    # Gets the mean returns !!

    Mret = self.get_Returns()
    Mret = np.mean(Mret, axis = 0)
    return Mret

def get_covMatrix(self):
    #if (self.covMatrix.shape[0] == 0):
    returns = self.get_Returns()
    self.covMatrix = np.cov(returns.T)
    return self.covMatrix 

def get_corMatrix(self):
    returns = self.get_Returns()
    self.corMatrix = np.corrcoef(returns.T)

    return self.corMatrix
    
def get_PortfolioReturn(self ):
    ## This function outputs the retuern of the portfolio
    # Without taking into account lending or borrowing money.

    returns = self.get_Returns()
    allocation = self.allocation
    
    # Return of the Portfolio without taking into 
    # acount lending or borrowing
    WR = returns.dot(allocation)
    
    WR = WR.reshape((WR.size,1))  ## THIS MIGHT FUCK ME UP !!
#==============================================================================
#     Rf = self.Rf  # Risk free rate
#     
#     # Xc is how much money we are using 
#     # The one we dont use we lend it (or borrow it if negative)
# 
#     Xc = np.sum(allocation)  
#     ## Using the Risk free rate !!
#     TWR = (1 - Xc) * Rf + WR 
#     
#==============================================================================
    return WR
    

def get_PortfolioStd(self):
    allocation = np.array(self.allocation)
    covMatrix = self.get_covMatrix()
    WRet = np.sqrt(allocation.dot(covMatrix).dot(allocation))
    return WRet

def get_PortfolioMeanReturn(self):
    Pret = self.get_PortfolioReturn()
    Pret = np.mean(Pret, axis = 0)
    return Pret
    
def plot_retCorr(self,sym_x, sym_y):
    # Given two markets, this function plots the scatter data between then and also the alpha and beta
    
    labels = ["CAPM scatter",sym_x,sym_y]
    ret_x = self.pf.symbols[sym_x].TDs[self.period].get_timeSeries(["Average"]);
    ret_y = self.pf.symbols[sym_y].TDs[self.period].get_timeSeries(["Average"]);
    
#        print ret_x.flatten().shape, ret_y.shape

    gr.scatter_graph(ret_x, ret_y, labels, 1)
    
def get_metrics(self, mode = "gaussian", investRf = "yes"):
    # Gets the Average Return and Std by joining all returns
    # It does not take into account the gaussianity itself
    # It sums the random variables and applies the definition 
    if (mode == "normal"):
        returns = self.get_Returns()
        WeightRet = returns.dot(self.allocation) # Wheiged sum of the returns 
        expRet = np.mean(WeightRet)
        stdRet = np.std(WeightRet)
    
    if (mode == "gaussian"):
        # Gets the metrics but using the covariance matrix instead of the real values
        # This model assumes sum of gaussians model
        expRet = self.get_PortfolioMeanReturn()
        stdRet = self.get_PortfolioStd()
    
    # If we are going to lend or borrow money at Rf rate
    if (investRf == "yes"):
        ### In this part, we assume that we are borrowing or lending all the money
        ## That we are not using. So what we do is: 
        ##  - We calculate how much money are we using and add how much we have 
        ## to borrow or lend to the risk free rate.
        Rf = self.Rf
        Xc = np.sum(self.allocation)  
        # Proper equation is Rt = (1 - X)Rf + XRp with X = sum(w)
        # Where Rp is the return of the portolio, assuming that we use all our monet
        # into it. In this equation expRet is allready the return of the portofio
        # using the specific weights. So we do not need to weight it. It is already
        # the final portolio.
        
        expRet = (1 - Xc) * Rf + expRet
 
    return expRet, stdRet

def get_SharpR(self):
    # Gets the sharp ratio of the Porfolio
    expRet, stdRet = self.get_metrics()
    
    # The SharpR is the 
    return expRet/stdRet
    

def simulate_Portfolio(self, mode = "gaussian"):
    # This function gets several data from your portfolio

    expRet, stdRet = self.get_metrics(mode = mode)
    # expRet is how much money you finally get, no matter what the fuck you did.
    # stdRet is how much variance you finally get, no matter what the fuck you did.
    coeffs = self.get_portfolio_ab(mode = mode)
    
    print "-------------------------------------------------------"
    print "Portfolio Metrics (" + mode + ")"
    print "Rf: " + str(self.Rf)
    print "Allocation: " + str(self.pf.symbols.keys())
    print "Allocation: " + str(self.allocation)
    print "Sum Allocation: " + str(np.sum(self.allocation))
    print "Expected Return: " + str(expRet)
    print "Volatily (std Return): " + str(stdRet)
    print "Sharp Ratio: " + str((expRet - self.Rf)/stdRet)
    print "......................................................."
    print "CAPM Model stuff.  Index: " + str(self.Sindex)
    print " Alpha: " + str(coeffs[0]) +" Beta: " + str(coeffs[1])  
    print "-------------------------------------------------------"

def compute_allocations(self,allocations): 
    # This function performs a fast computing of the Return and STD
    # of a set of portfolios. No caring about Rf rate.
    allocations = ul.fnp(allocations)     
    Nalloc = len(allocations)
    fast_flag = 1
    
#    if (len(allocations) == 1):
#        fast_flag = 0;
    if (fast_flag == 0):
    ## Properway Using Library Function ##
        P_Ret_s = [];
        P_std_Return_s = []
        for i in range (Nalloc):
            self.set_allocation(allocations[i])
            expRet = self.get_PortfolioMeanReturn()
            stdRet = self.get_PortfolioStd()
            P_Ret_s.append(expRet)
            P_std_Return_s.append(stdRet)
    
    # Fast Way !##
    else:
        # problems with allocations and function fnp
    
        Allocations = ul.fnp(allocations) 
#        print Allocations.shape
#        Allocations = Allocations.T
        ## Check the correct way of the matrix
        if (Allocations.shape[0] == len(self.symbol_names)):
            Allocations = Allocations.T
        P_Ret_s = np.dot(Allocations,ul.fnp(self.get_MeanReturns()));
        P_std_Return_s = np.dot(Allocations,self.get_covMatrix()) * ul.fnp(Allocations)
        P_std_Return_s = np.sqrt(np.sum(P_std_Return_s, axis = 1))
    
    P_Ret_s = ul.fnp(P_Ret_s)
    P_std_Return_s = ul.fnp(P_std_Return_s)
    
    ## These are the expectd return and std for the differen portfolios
    return P_Ret_s, P_std_Return_s


def get_random_allocations(self,Nrandom, short = "no", mode = "guassian", nf = 1):
    # Mode is the form to calculate the final standard deviation
    # short is to allow being short.

    # Scatter different portolios !!

    Randoms = np.random.uniform(0,1,(Nrandom,self.pf.Nsym))
#    print np.sum(Randoms,1) 
    Allocations = Randoms/np.sum(Randoms,1).reshape((Nrandom,1))  # Normalize to 1
    if (short == "Lintner"):
        # The sum of the absolutes will be 1
        # We create a random -1 1 1 -1 sign for every allocation
        sign = np.sign(np.random.randn(Nrandom,self.pf.Nsym))
        Allocations = Allocations * sign
        
    if (short == "yes"):
        # The sum will be 1
        sign = np.sign(np.random.randn(Nrandom,self.pf.Nsym))
        Allocations = Allocations * sign
        sums = np.sum(Allocations, axis = 1).reshape(Nrandom,1)
        
        ## We filter the sums a bit, in case we fucked up and they add to 0
        Alloc_indx = np.where(np.abs(sums).flatten() > 0.1)
        Alloc_indx = Alloc_indx[0]
        
        Allocations = Allocations[Alloc_indx,]
#        print Allocations.shape
        sums = sums[Alloc_indx]
        Allocations = Allocations/sums
        Nrandom, Nsym = Allocations.shape
    
    return Allocations
    
def scatter_allocations(self,allocations, 
                        labels = ['Porfolios', "Risk (std)", "Return"],
                        legend = ["Portfolios"],
                        lw = 2, alpha = 0.5,
                        nf = 1):
    ## Given a set of allocations, this function
    # plots them into a graph.
    returns, risks = self.compute_allocations(allocations)
    ## Scatter the random portfolios
    gl.scatter(risks, returns,labels = labels, legend = legend, 
            nf = nf, lw = lw, alpha = alpha)
def plot_allocations(self,allocations, 
                        labels = ['Porfolios', "Risk (std)", "Return"],
                        legend = ["Portfolios"],
                        lw = 5, alpha = 1.0,
                        color = None,
                        nf = 1):
    ## Given a set of allocations, this function
    # plots them into a graph.
    returns, risks = self.compute_allocations(allocations)
    ## Scatter the random portfolios
    gl.plot(risks, returns,labels = labels, legend = legend, 
            nf = nf, lw = lw, alpha = alpha, color = color)


def yearly_Return(self, returns):
    # Function that gives you the yearly returns of the returns given
    period = self.period 
    year_period = ul.names_dic["Y1"]
    
    num_sum = year_period/period
    
    return returns * num_sum
    
    
def yearly_covMatrix(self, covMatrix):
    # Function that gives you the yearly returns of the returns given
    period = self.period 
    year_period = ul.names_dic["Y1"]
    
    num_sum = year_period/period
    
    return covMatrix * np.sqrt(num_sum)
    
    