
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr
import matplotlib.colors as ColCon
import utilities_lib as ul
import basicMathlib as bMl

###################################################
########### Functions that have to with alpha beta #####################
########################################################

## Just another way to express the return and variance
## of all the symbols, related to one index.


import numpy as np
from scipy import linalg


def BlackLitterman(self, weq, Sigma, delta, # Prior portfolio variables
                   tau,              # Uncertainty coefficient of the porfolio priors
                   P, Q, Omega):       # Prior views variables

# blacklitterman
#   This function performs the Black-Litterman blending of the prior
#   and the views into a new posterior estimate of the returns.

# Inputs

## Prior Portfolio:
# We are going to give a prior portfolio. This is another gaussian variable
# where the value of weights is the mean, and we need to specify a covariance
# matrix for them. Usually we specify the covariance Matrix of the symbols.
# Since these are mainly dependent on it.
# This could be calculated using for example Portfolio Theory, efficient frontier.
# Or using Black Litterman thing of using the market capitalization.

#   weq    - Weights of the assets in the equilibrium portfolio
#   sigma  - Prior covariance matrix of the equilibrium portfolio
#   delta  - Risk tolerance from the equilibrium portfolio. The price of risk.
#           How much does variance cost in the equation U = w.T*T -1/2*A*w.T*S*w
#           A = delta. Usually it is the Exess return of the market divided by
#           the variance of the market A = (E[rm] - rf)/sigma^2_market.
#           We might use an index to find out A. 

### Prior Views:
# These are our personal views on how good will a symbol perform compared
# to other symbols. They are of the type S1 > S2 by X%.
# Matematically speaking this can be writen as: 
# S1 - S2 = N where N is GausianNoise with mean X and variance given by out confidence.
# The Gaussian Noise is our prior and has mean given in Q and variance
# given by Omega (how confident we are our prior)
# The matrix P is the selection Matrix, which multiplied by the symbols vector
# will give the right hand side of the equation (S1 - S2).

#   P      - Pick matrix for the view(s). NxM   (N views, M symobols)
#   Q      - Vector of view returns  (Nx1)
#   Omega  - Matrix of variance of the views (diagonal) (N x N)

#### Other coefficients
#   tau    - Coefficiet of uncertainty in the prior estimate of the mean (pi)
#           Regulates how sure we are about out prior portfolio.
#           Parameter to tune.

#           - It is usually 1 or 0.25
# Outputs
#   Er     - Posterior estimate of the mean returns
#   w      - Unconstrained weights computed given the Posterior estimates
#            of the mean and covariance of returns.
#   lambda - A measure of the impact of each view on the posterior estimates.
#

##########################################################

## Shape data so that it meets the python numpy needs:
    weq = ul.fnp(weq)
    Sigma = ul.fnp(Sigma)
    P = ul.fnp(P)
    Q = ul.fnp(Q) 
    Omega = ul.fnp(Omega)
    
    inv = np.linalg.inv # Change the name
    
    # Compute some inverses that are used frequently
    tauSigmaInv = inv(tau * Sigma)
    OmegaInv = inv(Omega)
# First we calculate the prior returns ! 
# These are the optimal returns of the prior portfolio.
# When maximizing utility function, this equation holds.
    pi = delta * np.dot(Sigma, weq) 

# The BL formula is:
# First multiplication factor [(tau*Sigma)^-1 + P.t*Omega^-1*P]^-1
# This is the uncertainty of the posterior returns as a combination
# of both priors (model portfolio and views)
    PostReturnsVariance = inv(tauSigmaInv + np.dot(np.dot(P.T,OmegaInv),P))
    
#   Now lets calculate the mean of the posterior returns
#   Compute posterior estimate of the uncertainty in the mean

    PostReturnsMean = np.dot(tauSigmaInv,pi) + np.dot(np.dot(P.T,OmegaInv),Q)
    PostReturnsMean = np.dot(PostReturnsMean.T, PostReturnsVariance)

#    The returns can also be obtained as:
#    PostReturnsMean = pi + np.dot(np.dot(np.dot(tau*Sigma,P),PostReturnsVariance),Q - pi)

    # Compute posterior weights based on uncertainty in mean
    # Basically the same equation as before (E = w*S). Now we use
    # The posterior Returns and posterior S
    w2 = PostReturnsMean.dot(inv(delta * PostReturnsVariance)).T
    w2 = np.dot(inv(delta * Sigma),PostReturnsMean.T)
  
  # Compute lambda value
#    lmbda = np.dot(linalg.pinv(P).T,(w2.T * (1 + tau) - weq).T)
  
    return [PostReturnsMean, w2]


