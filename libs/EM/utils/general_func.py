# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:47:19 2017

@author: montoya
"""

import import_folders
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.optimize import newton
import numpy as np
import utilities_lib as ul
import copy
import pandas as pd

def remove_module(X):
#    X = np.atleast_2d(X)
    N,D = X.shape
    Module =   np.linalg.norm(X, axis = 1) #np.sqrt(np.sum(np.power(X,2), axis = 1))
    Module = Module.reshape(N,1)
#    print (Module)
    X= np.divide(X,Module)
    return X

def normalize_module(Xdata):
#    tol = 0.0000001
    # Expects a matrix (Nsamples, Ndim) and normalizes the values
#    print Xdata.shape
    Nsamples, Ndim = Xdata.shape
#    mean_of_time_instances = np.mean(Xdata, axis = 0).reshape(1,Ndim)
#    mean_of_channels = np.mean(Xdata, axis = 1).reshape(Nsamples,1)
#     Substract mean
#    Xdata = Xdata - 
    Xdata = Xdata  # - mean_of_channels# - mean_of_time_instances # - mean_of_time_instances# - mean_of_time_instances
    
#    print Xdata.shape
    # Normalice module
    Module = np.sqrt(np.sum(np.power(Xdata,2), axis = 1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
#    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    return Xdata

def accuracy (Y,T):
    N_samples = len(Y)
    score = 0
    for i in range (N_samples):
#            print predicted[i], Y[i]
        if (Y[i] == T[i]):
            score += 1;
    return 100*score/float(N_samples)
    

def draw_HMM_indexes(pi, A, Nchains = 10, Nsamples = 30):
    # If Nsamples is a number then all the chains have the same length
    # If it is a list, then each one can have different length
    K = pi.size  # Number of clusters
    Chains_list = []
    
    Cluster_index = range(K)
    
    Nsamples = ul.fnp(Nsamples)
    if(Nsamples.size == 1):  # If we only have one sample 
        Nsamples = [int(Nsamples)]*Nchains
        
    for nc in range(Nchains):
        Chains_list.append([])
        sample_indx = np.random.choice(Cluster_index, 1, p = pi)
        Chains_list[nc].append(int(sample_indx))
        
        for isam in range(1,Nsamples[nc]):
            # Draw a sample according to the previous state
            sample_indx = np.random.choice(Cluster_index, 1, 
                                           p = A[sample_indx,:].flatten())
            Chains_list[nc].append(int(sample_indx))
    
    return Chains_list

def draw_HMM_samples(Chains_list, Samples_clusters):
    # We take the indexes of the chain and then draw samples from a pregenerated set
    # Samples_clusters is a list where every element is an array of sampled of the
    # i-th cluster
    
    K = len(Samples_clusters)

    Nchains = len(Chains_list)
    HMM_chains = [];
    
    counter_Clusters = np.zeros((K,1))
    for nc in range(Nchains):
        Nsamples = len(Chains_list[nc])
        HMM_chains.append([])
        
        for isam in range(0,Nsamples):
            K_index = Chains_list[nc][isam]
            Sam_index = int(counter_Clusters[K_index])
  
            sample = Samples_clusters[K_index][Sam_index,:]
            counter_Clusters[K_index] = counter_Clusters[K_index] +1
            HMM_chains[nc].append(sample)
    
        HMM_chains[nc] = np.array(HMM_chains[nc])
    return HMM_chains

def get_EM_data_from_HMM(HMM_list, Nchains_load = -1):
    # If Nchains_load = -1, it loads them all
    # Load first dataset
    k = 0 # For the initial
    for Xdata_chain in HMM_list:
        
        Xdata_chain = np.array(Xdata_chain)
#        gl.scatter_3D(Xdata_chain[:,0], Xdata_chain[:,1],Xdata_chain[:,2], nf = 0, na = 0)
    
    #    print Xdata_k.shape
        if (k == 0):
            Xdata = copy.deepcopy(Xdata_chain)
        else:
            Xdata = np.concatenate((Xdata, copy.deepcopy(Xdata_chain)), axis = 0)
        k += 1
        
        if k == Nchains_load: # Control how many samples we use
            break
        
    return Xdata
    

def sum_logs(log_vector, byRow = False):
    # This functions sums a vector of logarithms
    # alfa[i,n,t] = np.logaddexp(aux, alfa[i,n,t])

    if (byRow == False):
        # We just add all the components
        log_vector = np.array(log_vector).flatten()
        log_vector = np.sort(log_vector) # Sorted min to max
                 
        a0 = float(log_vector[-1])
        others = np.array(log_vector[:-1]).flatten()
        N = 1
    else:
        # log_vector = (Nrows, NvaluestoAdd)
        N = log_vector.shape[0]
        log_vector = np.sort(log_vector, axis = 1) # Sorted min to max
        a0 = log_vector[:,[-1]]
        others = log_vector[:,:-1]
        
        if (0):
            log_vector = np.array(log_vector)
            N = log_vector.shape[0]
            result = []
            for i in range(log_vector.shape[0]):
                result.append(sum_logs(log_vector[i,:]))
            
            result = np.array(result).flatten().reshape(N,1)
            
            return result

#    print np.exp(others - a0).shape
    if (byRow == True):
        caca = np.sum(np.exp(others - a0),axis = 1)
        caca = caca.reshape(N,1)
    else:
        try:
            caca = np.sum(np.exp(others - a0))
        except AttributeError:
            print (a0)
            print (others)
            print (type(others))
            print (log_vector)
#    print caca.shape
    result = a0 + np.log(1 + caca)
    
    if (byRow == True):
#        print result.shape
        result = result.flatten().reshape(N,1)
    return result



############### LOADING FUNCTIONS ############
# TODO: probably move from here
def load_real_data(path = 'data/',file_name='face_scrambling_ERP.mat'):
    import scipy.io as sio
    mat = sio.loadmat(path+file_name)
    data = mat['X_erp']
    conditions = [x[0] for x in  mat['CONDITIONS'][0]]
    return data,conditions

def load_fake_data(path = 'data/'):
    from os import listdir
    csv = [x for x in listdir(path) if x.find('.csv')!=-1]
    data = [np.array(pd.read_csv(path+X,sep=','))  for X in csv]
    X = np.concatenate(data,axis=0)
    return X

def normalize_data(data):
    for i,row in enumerate(data):
        for j,x in enumerate(row):
            data[i,j] = normalize_subject(x)
    return data

def normalize_subject(data):
    d,N = data.shape
    for i in range(N):
        # TODO: Not normalized anymore
#        data[:,i] = data[:,i]/np.linalg.norm(data[:,i])
        pass
    return data.T
