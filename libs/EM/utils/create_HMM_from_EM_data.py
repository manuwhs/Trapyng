
# Official libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own libraries
import import_folders
from graph_lib import gl
import sampler_lib as sl
import EM_lib as EMl
import pickle_lib as pkl

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf
import copy


K = 3
gl.scatter_3D(0, 0,0, nf = 1, na = 0)
N = 10000

Xdata = []  # List will all the generated data
K = 3
#gl.scatter_3D([0,1,1,1,1,-1,-1,-1,-1], [0,1,1,-1,-1,1,1,-1,-1],[0,1,-1,1,-1,1,-1,1,-1], nf = 1, na = 0)
gl.scatter_3D(0, 0,0, nf = 1, na = 0)
kflag = 0
for k in range(1,K+1):
    folder = "./EM_data/"
    folder = "./test_data/"
    
    filedir = folder + "Wdata_"+ str(k)+".csv"
    Xdata_k = np.array(pd.read_csv(filedir, sep = ",", header = None))
    Xdata_k = Xdata_k[:,:]
    print Xdata_k.shape
#    Xdata_param = pkl.load_pickle( folder + "Wdata_"+ str(k)+".pkl",1)
#    mu = Xdata_param[0]
#    kappa = Xdata_param[1]
    
#    print "Real: ", mu,kappa
    # Generate and plot the data
    gl.scatter_3D(Xdata_k[:,0], Xdata_k[:,1],Xdata_k[:,2], nf = 0, na = 0)

    mu_est2, kappa_est2 = Wae.get_Watson_muKappa_ML(Xdata_k)
    print "ReEstimated: ", mu_est2,kappa_est2
    
    Xdata.append(copy.deepcopy(Xdata_k))

################################################################
######## Generate HMM data and store it ###########################
################################################################

## Now we define the parameters of the HMM
pi = np.array([0.2,0.3,0.5])
A = np.array([[0.4, 0.1, 0.5],
              [0.4, 0.5, 0.1],
              [0.7, 0.1, 0.2]])

## For every chain, we draw a sample according to pi and then
## We keep drawing samples according to A

Nchains = 20  # Number of chains
Nsamples = 50 + np.random.rand(Nchains) * 20  # Number of samples per chain
Nsamples = Nsamples.astype(int)  # Number of samples per chain

# For training
Chains_list = gf.draw_HMM_indexes(pi, A, Nchains, Nsamples)
HMM_list = gf.draw_HMM_samples(Chains_list, Xdata)

## For validation !!!

Nsa = Xdata[0].shape[1]
Xdata2 = []
for Xdata_k in Xdata:
    Xdata2.append(Xdata_k[int(Nsa/2):,:])
    
Chains_list2 = gf.draw_HMM_indexes(pi, A, Nchains, Nsamples)
HMM_list2 = gf.draw_HMM_samples(Chains_list2, Xdata2)

gl.scatter_3D(0, 0, 0, nf = 1, na = 0)

for XdataChain in HMM_list:
    gl.scatter_3D(XdataChain[:,0], XdataChain[:,1],XdataChain[:,2], nf = 0, na = 0)


# We pickle the information
# This way we have the same samples for EM and HMM

folder = "./HMM_data/"
pkl.store_pickle(folder +"HMM_labels.pkl",Chains_list,1)
pkl.store_pickle(folder +"HMM_datapoints.pkl",HMM_list,1)
pkl.store_pickle(folder +"HMM_param.pkl",[pi,A],1)

pkl.store_pickle(folder +"HMM2_datapoints.pkl",HMM_list2,1)