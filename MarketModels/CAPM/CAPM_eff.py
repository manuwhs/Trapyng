
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr
import matplotlib.colors as ColCon
import basicMathlib as bMl

def randomly_optimize_Portfolio(self, Nrandom):
    # This function tries to find the best allocation randomly tring

    Randoms = np.random.uniform(0,1,(Nrandom,self.pf.Nsym))
    Allocations = Randoms.T/np.sum(Randoms,1)  # Normalize to 1
    Allocations = Allocations.T
    Sharps_R = [];
    
    for i in range (Nrandom):
         self.set_allocation(Allocations[i])
         expRet, stdRet = self.get_metrics()
         Sharps_R.append(expRet)
    
    Sharps_R = np.array(Sharps_R)
    argbest_allocatio = np.argmax(Sharps_R)
    
    b_alloc = Allocations[argbest_allocatio,:]
    self.set_allocation(b_alloc)
    return b_alloc


import cvxopt as opt
from cvxopt import blas, solvers

def efficient_frontier(self, kind = "Markowitz", N = 1000, max_exp = 5.0):
    ## Calculates the efficient frontier using convex optimization
    ## There are 3 types of frontiers:
    ##  - Markowitz:  No short sales allowed. Sum(w) = 1
    ##  - Lintner: Short sales allowed. Sum(abs(w)) <= 1. Assume Rf = 0
    ##  - Normal: Short sales allowed. Sum(w) = 1
    ##  - Tangent: Short sales allowed. Sum(w) = 1. 
    #              But we calculate using the Tangent function

    # N is the number of points of the efficient frontier we calculate.
    # max_exp is the exponent for the search

    if (kind == "Tangent"):
        allocations = self.TangenPortfolioFrontier(N = N, max_exp = max_exp)
#        print allocations
        only_eff = 1
        if (only_eff == 1):  # Only the real efficient frontier !
            rets, stds = self.compute_allocations(allocations)
            indexes = np.argwhere(np.diff(stds)  < 0).T[0]
            allocations = np.array(allocations)[indexes,:]
#            print allocations
#            print np.diff(stds)
#            print indexes
        optimal = self.TangentPortfolio(Rf = self.Rf)
    
        return optimal, allocations
    
    solvers.options['show_progress'] = False
    ## Number of symbols and their returns
    n = len(self.pf.symbols.keys())
    returns = self.get_Returns()
    
    ## Number of samples santard deviations for which
    ## we will calculate the highest return. 
    
#    N = 100  # Exponential search in the mean mu.
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(self.get_covMatrix())
    pbar = opt.matrix(np.mean(returns, axis=0))
    
    ####### Create constraint matrices
    ###### Gx <= b ####### 
    
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    
    if (kind == "Markowitz"):
        h = opt.matrix(0.0, (n ,1))  # X cannot be smaller than 0 
    
    elif (kind == "Normal"):
        h = opt.matrix(100.0, (n ,1)) # No contraint in the value (just too high)
    
    elif(kind == "Lintner"):
        N_constraints = 2**n
        G = opt.matrix(1.0, (N_constraints, n))  # The sum of weights is 1
        ## Create the +1 -1 arrays
           
        Gaux = np.ones((N_constraints, n))
        for i in range (n):
            period = [1]*(2**i)
            period.extend([-1]*(2**i))
            
#            print period
#            A[1::(2**i),i] = -1
            Gaux[:,i] = np.tile(period,2**(n-i-1))

#        print Gaux.shape
        G = opt.matrix(Gaux)
#        print G
        
        h = opt.matrix(1.0, (N_constraints,1))  # Sums to 1
    else:
        print "You fucked up"
        return -1
        
    #################
    #### Ax = b #####
    # Here we place the constraint in the weights

    A = opt.matrix(1.0, (1, n))  # The sum of weights is 1
    b = opt.matrix(1.0)
    
    if(kind == "Lintner"):
        # We remove this constraint
        A = opt.matrix(0.0,(1, n))  # The sum of weights is 1
        A[0,0] = 0.1
        b = opt.matrix(0.0)
    
    # Calculate efficient frontier weights using quadratic programming
    if(kind == "Lintner"):
        # No linear constraints
        portfolios = [solvers.qp(mu*S, -pbar, G, h)['x'] 
                      for mu in mus]
    else:    
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A,b)['x'] 
                      for mu in mus]                  
    # Portfolios contains now the list of optimal allocations               
    ## Transform the cvxopt.base.matrix allocations to nparrays
    
    allocations = []
    for port in portfolios:
        allocations.append(np.array(port))
        
    ###################################################
    ######## CALCULATE Optimal Portfolio  #############
    ###################################################
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER with Rf = 0
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt).T, allocations
    
### Allowing risk free borrowing and lending, this 
# function give us the optimal w for the market.

def TangentPortfolio(self, Rf = 0, norm = "sum",):
    # This function finds the optimal portfolio when
    # there is a risk-free rate and short sales are allowed.
    # You can borrow and lens as much money as you want
 
    returns = self.get_Returns()
    mu = np.mean(returns, axis = 0)
    C = self.get_covMatrix()
    
    nr,nr = C.shape
    w = np.dot(np.linalg.inv(C),mu-Rf)  # Nope

    ## PLACE IN THE FUCKING RIGHT PLACE (Within the posibilities)
    ## This will place it in the position where we do not fucking
    ## use the risk-free component 

    ### Now we normilize the result so that we are not borrowing
    ## or lending any money.
    w = w/np.sum(w)   
    
    ## TODO: This is not useful
    if (norm == "abs"):
        # Now that it does not have risk-free component we normalize it
        w = w/np.sum(np.abs(w))
        
    return w

def TangenPortfolioFrontier(self, N = 100, maxRf = 0.05, max_exp = 1, norm = "sum"):
    # Using the tangent portfolio this calculates the frontier
    # We get two points in the frontier and do linear combination
    # Pf = k(P1) + (1-k)P2
    # max_exp is for knowing how much we can combine them

    kgrid = np.array(range(-N,N))*max_exp/float(N)
    
    P1 = self.TangentPortfolio(norm = norm, Rf = maxRf)
    P2 = self.TangentPortfolio(norm = norm, Rf = -maxRf)
    
    portfolios = []
    for k in kgrid:
        port = np.array(P1)*k + (1-k)*(np.array(P2))
        portfolios.append(port.tolist())
    
#    print portfolios
    return portfolios
    
def TangenPortfolioFrontier2(self, maxRf = 0.05, norm = "sum"):
    # Using the tangen portfolio this calculates the frontier
    # by varying the number of the risk-free rate
    # MaxRf is for the linear space of the risk-free rate
    N = 100
    mus =  np.array(range(-N,N))/float(N)
    mus = mus * maxRf
#    print mus
    portfolios = []
    for mu in mus:
        port = self.TangentPortfolio(norm = norm, Rf = mu)
        portfolios.append(port)
    
    return portfolios

    
def Market_line (self, alloc = [], Rf = 0):
    # This function gets the market line points for a given
    # allocation and Risk_free ratio
    
    # First we get the optimal porfolio
    if (len(alloc) == 0):
        alloc = self.TangentPortfolio(Rf = Rf, norm = "sum")
    
    self.set_allocation(alloc)
    self.set_Rf(Rf)
    
    # Obtain its Return and Std
    expRet, stdRet = self.get_metrics()  # expRet, stdRet
    
    param = bMl.obtain_equation_line(Rf, expRet, stdRet)
    bias,slope = param
    
    if (slope < 0):
        slope = -slope
        
#    ## Plotting this shit
#    N = 10 
#    xgrid = np.array(range(0,N))
#    xgrid = xgrid * (2*stdRet/N)
#    
#    ygrid = xgrid * slope + bias
#    
#    plt.plot(xgrid,ygrid)
    
    return bias, slope

def obtain_Porfolio_by_Ret(self, Ret = 0):
    ## This function obtains the optimal porfolio for
    # a given return, moving thourgh the Market line.
    bias, slope
    
def plot_MarketLine(self):
    pass
    
#import scipy.optimize as scopt
#
#def obtain_efficient_frontier(returns):
#    result_means = []
#    result_stds = []
#    result_weights = []
#    
#    means = returns.mean()
#    min_mean, max_mean = means.min(), means.max()
#    
#    nstocks = returns.columns.size
#    
#    for r in np.linspace(min_mean, max_mean, 100):
#        weights = np.ones(nstocks)/nstocks
#        bounds = [(0,1) for i in np.arange(nstocks)]
#        constraints = ({'type': 'eq', 
#                        'fun': lambda W: np.sum(W) - 1})
#        results = scopt.minimize(objfun, weights, (returns, r), 
#                                 method='SLSQP', 
#                                 constraints = constraints,
#                                 bounds = bounds)
#        if not results.success: # handle error
#            raise Exception(result.message)
#        result_means.append(np.round(r,4)) # 4 decimal places
#        std_=np.round(np.std(np.sum(returns*results.x,axis=1)),6)
#        result_stds.append(std_)
#        
#        result_weights.append(np.round(results.x, 5))
#    return {'Means': result_means, 
#            'Stds': result_stds, 
#            'Weights': result_weights}

