import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
# Data Structures Data
import CPortfolio as CPfl
import CSymbol as CSy
# Own graphical library
from graph_lib import gl 
# Import functions independent of DataStructure
import utilities_lib as ul
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import pandas as pd
from sklearn import linear_model
import basicMathlib as bMA
import MultivariateStatLib as MSlib
import indicators_lib as indl
import DDBB_lib as DBl
plt.close("all")

folder_images = "../pics/Trapying/MultivariateStat/"
##############################################
########## FLAGS ############################



##########################################################################
################# DATA OBTAINING ######################################
##########################################################################

mus = np.array([-0.5,-1,1.5])
stds = np.array([1,1.5,2])
Nsam = 1000
Nx = 10

if (Nx >3):
    mus = np.random.randn(Nx)
    stds = np.random.randn(Nx)
    
X = []
for i in range(Nx):
    X_i = np.random.randn(Nsam,1)*stds[i] + mus[i]
    X.append(X_i)

X = np.concatenate((X),axis = 1)

##########################################################################
################# EXPERIMENTS !! ######################################
##########################################################################

### Degrees of freedom of the eigenvalues test

D = 12;
Nsam = 1000;
Nlast_equal = 8;
m = D - Nlast_equal

df = (1.0/2)* (D-m +2)*(D-m-1)
print "Degrees of freedom eigenvalues %f"%df

## Perform the test statistic for the smallest eigenvalues 
lambdas = [9.07, 1.63, 0.87, 0.15, 0.086, 0.078, 0.061, 0.034, 0.033, 0.022,0.008, 0.005]
lambdas = np.array(lambdas)

prod_lambdas = 1
for i in range (m,D):
    prod_lambdas = prod_lambdas*lambdas[i]
    
lambda_Est = np.sum(lambdas[m:D])/(D-m)

n1 = Nsam - m - (1.0/6)*(2*(D-m) + 1 + 2.0/(D-m))

Z1 = -n1 * np.log(prod_lambdas/np.power(lambda_Est, D-m))

print "D: %i, m: %i, Nlast_equal: %i"%(D, m , Nlast_equal)
print "n1: %f" % (n1)
print "Z1: %f" % (Z1)
############################################################################
## Partial correlation V[Y|X] if if has to be computed by hand
Sigma_YY = np.array([[1,1],[1,4]])
Sigma_XY = np.array([[1.0/4,1.0/8],[1,1]])
Sigma_XX = np.array([[1,1],[1,4]])

Var_Y_X = Sigma_YY - Sigma_XY.T.dot(np.linalg.inv(Sigma_XX)).dot(Sigma_XY)
R2 = Var_Y_X[0,1]/(np.sqrt(Var_Y_X[1,1]* Var_Y_X[0,0]))
R2 = R2* R2;

## Coefficients of the Expected mean E[Y|X]
Sigma_YY = np.array([[4]])
Sigma_XY = np.array([[1.0/4,1.0/8]]).T
Coeff = Sigma_XY.T.dot(np.linalg.inv(Sigma_XX));

## Multiple Correlation if computed by hand

Sigma_YX = np.array([[1,1]])
Sigma_XX = np.array([[1,1],[1,4]])
sigma_Y = 4

rho_Y_XX  = Sigma_YX.dot(np.linalg.inv(Sigma_XX)).dot(Sigma_YX.T)/sigma_Y
##### Test between Linear Models

SS_M = 167.56
SS_H = 207.7

D_M = 6  # Number of param of model M
D_H = 4  # Number of param of model H
N = 12 # Number of samples


F_value = ((SS_H - SS_M)/(D_M- D_H))/(SS_M/(N-D_M))
df = [ D_M- D_H , N-D_M]

#### LCA : Compute F-statistic from Malhanovis distance:
    
n1 = 20
n2 = 20
D = 3
d = 1.311

Fstat = (float((n1 + n2 - D -1))/(D*(n1+n2-2)))* ((n1*n2)/(n1 + n2)) * d

### Wilks Lambda Test:
N = 178 # Number of samples
k = 3   # Number of dimensions of X (In one way MANOVA is the number of classes)
p = 4   # Number of output classes. In MANOVA it is the number of continuous variables

# Selection variables
s = 4   # Number of output we select
r = 2   # Number of dimensions of theta we select


Udf = [s,r,N-k]

# Now we change the nomenclature
p,q,r = Udf

if (p*p + q*q == 5):
    t = 1
else:
    t = np.sqrt(float((p*p*q*q -4))/(p*p + q*q -5))

v = (2*r + q -p -1)/2

Fdf = [p*q,v*t +1 -p*q/2] 


### Computing the constant of the LDA:
p1 = 2.0/3
p2 = 1.0/3
d = np.log(p2/p1)


## Fucking GLMs model
y = np.array([[-3,-1,1,0,1,2,2]]).T # Nsam x Ndim
x1 = np.array([[-3,-2,-1,0,1,2,3]]).T # Nsam x Ndim
x0 = np.ones((x1.size,1))
X = np.concatenate((x0,x1),axis = 1)
Nsam, Ndim = X.shape

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
res  = (y - X.dot(theta))
sigma_e = res.T.dot(res)/(Nsam- Ndim)
cov_theta = sigma_e* np.linalg.inv(X.T.dot(X))

# Estimation of new sample:

xnew = np.array([[1,0]])  # Nsam x Ndim

Var_newsam = xnew.dot(np.linalg.inv(X.T.dot(X))).dot(xnew.T)


