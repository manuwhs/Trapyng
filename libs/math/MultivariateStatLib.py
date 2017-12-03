#########################################################3
############### BASIC MATH ##############################
##########################################################
## Library with basic mathematical functions 
# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import matplotlib.colors as ColCon
from scipy import spatial
import datetime as dt
from sklearn import linear_model
import utilities_lib as ul

from sklearn.decomposition import PCA    
import math



def get_explained_var_PCA(X, pca = None):
    if (type(pca) != type(None)):
        nSamples, nFeatures = X.shape
        pca = PCA()
        pca.fit(X)
    return pca.explained_variance_

def get_eigenValues():
    n_samples = X.shape[0]
    # We center the data and compute the sample covariance matrix.
    X -= np.mean(X, axis=0)
    cov_matrix = np.dot(X.T, X) / n_samples
    for eigenvector in pca.components_:
        print (np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    

    # Plot classification score vs number of components
#    nComponents = np.arange(1,nFeatures,8)
#    pcaScores = np.zeros((5,np.alen(nComponents)))
    
#    for i,n in enumerate(nComponents):   
#        pca = PCA(n_components=n,whiten=False)
#        XtrainT = pca.fit_transform(Xtrain)
#        XtestT = pca.transform(Xtest)
#        pcaScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    
 #%% Plot data proyections for PCA
def get_components_PCA(X, n_components = 2, pca = None):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    uPCA = pca.components_
    
    return uPCA

def get_projected_PCA(X, n_components = 2, pca = None):
    pca = PCA(n_components=n_components)
    xtPCA = pca.fit_transform(X)
    return xtPCA

def get_cumvawr_PCA(X, pca = None):
    
  ######################################
  # Only left with 2 components to show the cahnge of PCA
    # Just to show the hyperplane of maximum varinace in 2 D, we transform the data into 2 D.
    
    # Training data with 2 dimensions
    pca = PCA(n_components=2)
    Xtrain_2d = Xtrain[:,:2]
    xtPCA = pca.fit_transform(Xtrain_2d)
    uPCA = pca.components_
        

    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(Xtrain_2d[labelsTrain==l,0],Xtrain_2d[labelsTrain==l,1],alpha=alpha_val,c=classColors[i])
    
    plt.quiver(uPCA[0,0],uPCA[0,1],color='k',edgecolor='k',lw=1,scale=5)
    plt.quiver(uPCA[1,0],uPCA[1,1],color='k',edgecolor='k',lw=1,scale=10)
    plt.title('Original Data and first 2 eigenvectors')
#    plt.xlim([-4,4])
#    plt.ylim([-4,4])
    plt.show()

    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=alpha_val,c=classColors[i])

    plt.title('Projected data over the components')
#    plt.xlim([-4,4])
#    plt.ylim([-4,4])
    plt.show()
    