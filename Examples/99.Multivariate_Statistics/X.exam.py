import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy

# Own graphical library
from graph_lib import gl 
# Import functions independent of DataStructure
import utilities_lib as ul
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models

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


Diff_mean_MANOVA_or_LDA = 1
eigenvalue_test = 0
corr_flag = 0
Diff_GLM = 0
Wilks_Lambda = 0
LDA_constant = 0
GLM_params = 0


if (eigenvalue_test):
    D = 9;
    Nsam = 176;
    Nlast_equal = 3;
    m = D - Nlast_equal
    
    df = (1.0/2)* (D-m +2)*(D-m-1)
    print ("Degrees of freedom eigenvalues %f"%df)
    
    ## Perform the test statistic for the smallest eigenvalues 
    lambdas = [9.07, 1.63, 0.87, 0.15, 0.086, 0.078, 0.061, 0.034, 0.033, 0.022,0.008, 0.005]
    lambdas = np.array(lambdas)
    
    prod_lambdas = 1
    for i in range (m,D):
        prod_lambdas = prod_lambdas*lambdas[i]
        
    lambda_Est = np.sum(lambdas[m:D])/(D-m)
    
    n1 = Nsam - m - (1.0/6)*(2*(D-m) + 1 + 2.0/(D-m))
    
    Z1 = -n1 * np.log(prod_lambdas/np.power(lambda_Est, D-m))
    
    print ("D: %i, m: %i, Nlast_equal: %i"%(D, m , Nlast_equal))
    print ("n1: %f" % (n1))
    print ("Z1: %f" % (Z1))

if (corr_flag):
    """
    Operations Related To Correlations, Partial Correlation, Multiple Correlations
    """
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

if (Diff_GLM):
    SS_M = 167.56
    SS_H = 207.7
    
    D_M = 4  # Number of param of model M
    D_H = 2  # Number of param of model H
    N = 81 # Number of samples
    
    
    F_value = ((SS_H - SS_M)/(D_M- D_H))/(SS_M/(N-D_M))
    df = [ D_M- D_H , N-D_M]


if (Wilks_Lambda):
    ### Wilks Lambda Test:
    N = 57           # Number of samples
    k = 3           # Number of dimensions of X (In one way MANOVA is the number of classes)
    p = 2           # Number of output classes. In MANOVA it is the number of continuous variables
    
    # Selection variables
    s = 2   # Number of output we select
    r = 2   # Number of dimensions of theta we select
    
    
    Udf = [s,r,N-k]
    
    print (["Degrees freedom U:", Udf])
    # Now we change the nomenclature
    p,q,r = Udf
    
    if (p*p + q*q == 5):
        t = 1
    else:
        t = np.sqrt(float((p*p*q*q -4))/(p*p + q*q -5))
    
    v = (2*r + q -p -1)/2
    
    Fdf = [p*q,v*t +1 -p*q/2] 
    
    print (["Degrees freedom F:", Fdf])
   
if (LDA_constant):
    ### Computing the constant of the LDA:
    p1 = 2.0/3
    p2 = 1.0/3
    d = np.log(p2/p1)

if (Diff_mean_MANOVA_or_LDA):
    ### Computing the constant of the LDA:
    n1 = 16   # Number of samples of one class
    n2 = 21   # Number of smaples of the other
    D = 2     # Dimensionality of the continuous space
    d = 0.3 
    
    df = [D, n1 + n2 - D -1]
    
    n1 = float(n1)  # As to obtain float values
    k = ((n1 + n2 - D - 1)/ (D*(n1 + n2 -2))) * (n1*n2)/(n1+n2)
    
    Fstat = k * d
    print (["Degrees of freedom: ", df])
    print (["Konstant: ", k])
    
    print (["Fstat: ", Fstat])
    

## Fucking GLMs model

if (GLM_params):
    y = np.array([[2,1,3,4,0]]).T # Nsam x Ndim
    x1 = np.array([[-2,-1,0,1,2]]).T # Nsam x Ndim
    
    x0 = np.ones((x1.size,1))
    
    X = np.concatenate((x0,x1),axis = 1)
    Nsam, Ndim = X.shape
    
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    res  = (y - X.dot(theta))
    sigma_e = res.T.dot(res)/(Nsam- Ndim)
    cov_theta = sigma_e* np.linalg.inv(X.T.dot(X))
    
    print ("Parameters theta")
    print (theta)

    print ("Covariance of the parameters theta")
    print (cov_theta)
    
    # Estimation of new sample:
    
    xnew = np.array([[1,0]])  # Nsam x Ndim
    Var_newsam = xnew.dot(np.linalg.inv(X.T.dot(X))).dot(xnew.T)


A = np.array([[0,1,0],[0,0,1]])
B = np.array([[0,1,0],[0,0,1]]).T

C = np.array(range(1,10)).reshape((3,3))

Sel = A.dot(C).dot(B)

    