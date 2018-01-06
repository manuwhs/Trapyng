# Change main directory to the main folder and import folders
import os

# Official libraries
import matplotlib.pyplot as plt

import pandas as pd

# Own libraries
from graph_lib import gl
import sampler_lib as sl
import EM_libfunc as EMlf
import EM_lib as EMl
import copy
import pickle_lib as pkl
import Gaussian_distribution as Gad
import Gaussian_estimators as Gae
import Watson_distribution as Wad
import Watson_estimators as Wae
import vonMisesFisher_distribution as vMFd
import vonMisesFisher_estimators as vMFe

import general_func as gf
import numpy as np
plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"


class CDistributionManager():
    """
    We want to create a more open EM algorithm that allows clusters with different 
    properties: Differente Distribution, or different hyperparameters, or different Constraints in the distribution..
    We would also like to be able to manage constrains between the different clusters,
    like for example, the mean of two clusters is the same.
    
    This class aims to contain a set of distributions, D, and each distribution has a 
    number of cluster K_d that are generated according to that distribution
    """
    
    def __init__(self, distribution_list = None):
        
        """ 
        The next dictionary holds the different distributions that the mixtures has.
        They are references by name
        """
        
        """
        Dictionaty with ["Distribution_name"] = Distribution_object 
        This distribution object has its own hyperparameters
        """
        self.Dname_to_distribution = dict()
        """
        Dictionaty with ["Distribution_name"] = [3,4,5]  
        The index of the clusters that belong to this distribution.
        """
        self.Dname_to_clusters = dict()
        """
        Dictionary with the number of clusters, where each element tells the
        Distribution ID to which the k-th cluster belongs to.
        It is redundant since the previous 2 are enough but it is easier 
        to program with this.
        [3]: Distribution_name.
        Making this a dictionary makes it easier.
        """
        self.clusterk_to_Dname = dict()
        
        """
        The clusters numbers "k" in the distribution could be anything.
        As the algo progresses, they could be deleted or added with new k.
        The thetas for now is a list, we need to match the thetas con the
        "k"s in this distribution. This is what this dict does.
        """
        self.clusterk_to_thetak = dict()
        
    def add_distribution(self, distribution, Kd_list):
        # Kd_list = [0,1,2]
        """ 
        Function to add a distribution and the number of clusters.
        We identify the distributions by the name given.
        """

        self.Dname_to_distribution[distribution.name] = distribution
        self.Dname_to_clusters[distribution.name] = []
        for k in Kd_list:
            self.add_cluster(k, distribution.name)
    
#        print self.Dname_to_clusters
#        print self.Dname_to_distribution
#        print self.clusterk_to_Dname
#        print self.clusterk_to_thetak
    def remove_cluster(self, k):
        """
        Remove the cluster from the data structures.
        TODO: What if a distribution ends with no clusters ?
        Will it crash somewhere ?
        """
        distribution_name = self.clusters_distribution_dict[k]
        self.Dname_to_clusters[distribution_name].remove(k)
        self.clusterk_to_Dname.pop(k, None)
        self.clusterk_to_thetak.pop(k, None)
        
    def add_cluster(self,k, distribution_name):
        """
        We can add a cluster to one of the asociations
        """
        K = len(self.clusterk_to_Dname.keys())
        self.Dname_to_clusters[distribution_name].append(k)
        self.clusterk_to_Dname[k] = distribution_name
        self.clusterk_to_thetak[k] = K 
        
    def init_params(self, X, theta_init):        
        """
        This function initilizes the parameters of the distributions
        using the dunction provided, or directly the theta_init provided.
        We provide with X in order to be able to use the samples for initialization
        """
        K = len(self.clusterk_to_Dname.keys())
        theta = []
        for k in range(K):
            theta.append(None)
        
#        print ("K",K)
        Dnames = self.Dname_to_distribution.keys()
        for Dname in Dnames:
            distribution = self.Dname_to_distribution[Dname]
            theta_indexes = self.Dname_to_clusters[Dname] 
#            print ("theta_indexes",theta_indexes)
            ## If we are given directly the init theta, we do nothing
            if (type(theta_init)!= type(None)):
                # Get the theta_k corresponding to the clusters of the distribution
                theta_dist = [theta_init[self.clusterk_to_thetak[i]] for i in theta_indexes]
            else:
                theta_dist = None
            ## Compute the init parameters for the Kd clusters of the distribution
            Kd = len(theta_indexes) # Clusters of the distribution
            theta_cluster = distribution.init_params(X, Kd, theta_dist, distribution.parameters)
            
            for i in range(len(theta_indexes)): # Apprend them to the global theta
                theta[self.clusterk_to_thetak[theta_indexes[i]]] = theta_cluster[i]
            
#            print ("Length theta:", len(theta))
        return theta 
    
    def get_Cs_log(self, theta):
        """
        This funciton computes the Normalization constant of the clusters.
        TODO: Ideally, we will not need it when we only compute the likelihoods once.
        For now we will use it
        """

        K = len(self.clusterk_to_thetak.keys())
        Cs = []
        for k in range(K):
            Cs.append(None)
            
        Dnames = self.Dname_to_distribution.keys()
        for Dname in Dnames:
            distribution = self.Dname_to_distribution[Dname]
            theta_indexes = self.Dname_to_clusters[Dname] 
            
            for k in theta_indexes:
                k_theta = self.clusterk_to_thetak[k]
                try:
                   C_k = distribution.get_Cs_log(theta[k_theta], parameters = distribution.parameters) # Parameters of the k-th cluster
                except RuntimeError as err:
        #            error_type = err.args[1]
        #            print err.args[0] % err.args[2]
                    print ("Cluster %i degenerated during computing normalization constant" %k)           ####### HANDLE THE DEGENERATED CLUSTER #############
                    C_k = None;
                Cs[k_theta] = C_k
        return Cs
    
        
    ###################  
    def pdf_log_K(self, data, theta, Cs_logs = None):
        """
        Returns the likelihood of the samples for each of the clusters. Not multiplied by pi or anything.
        It is independent of the model.
        
        If the the data is an (N,D) array then it computes it for it. It returns ll[N,K]
        
        if it is a list, it computes it for all of them separatelly. It returns ll[ll1[N1,K], ll2[N2,K],...]
        """
        
        if (type(data) == type([])):
            pass
            list_f = 1
        else:
            data = [data]
            list_f = 0
        
        K = len(theta)
#        print D,N,K
        if (type(Cs_logs) == type(None)):
            Cs_logs = self.get_Cs_log(theta)
        
        ll_chains = []
        for X in data:
            N,D = X.shape
            lls = np.zeros((N,K))
        
            Dnames = self.Dname_to_distribution.keys()
            for Dname in Dnames:
                distribution = self.Dname_to_distribution[Dname]
                theta_indexes = self.Dname_to_clusters[Dname] 
#                print (theta_indexes)
                theta_dist = [theta[self.clusterk_to_thetak[i]] for i in theta_indexes]
                
                if (type(Cs_logs) != type(None)):
                    Cs_logs_dist = [Cs_logs[self.clusterk_to_thetak[i]] for i in theta_indexes]
                else:
                    Cs_logs_dist = None
                
                lls_d = distribution.pdf_log_K(X.T,theta_dist, parameters = distribution.parameters, Cs_log = Cs_logs_dist )
    #            print lls_d.shape
                for i in range(len(theta_indexes)):
                    lls[:,self.clusterk_to_thetak[theta_indexes[i]]] = lls_d[:,i]
            
                ## TODO: Variational inference ! Change in this !
                ## In this simple case since Gaussian is 1 dim less, we want to give it more importance !
                lls[:,0] =  lls[:,0] # *(D-1)/float(D) #(2/float(3)) # + np.log(2)  # Multiply by 2, just to try 
                
            ll_chains.append(lls)
            
        if (list_f == 0):
            ll_chains = ll_chains[0]
        return ll_chains
    
    def get_theta(self, X, r):
        N,D = X.shape
        N,K = r.shape
        
        theta = []
        for k in range(K):
            theta.append(None)
            
        Dnames = self.Dname_to_distribution.keys()
        for Dname in Dnames:
            distribution = self.Dname_to_distribution[Dname]
            theta_indexes = self.Dname_to_clusters[Dname] 
            
            for k in theta_indexes:
                k_theta = self.clusterk_to_thetak[k]
                rk = r[:,[k_theta]]  # Responsabilities of all the samples for that cluster
                try:
                   theta_k = distribution.theta_estimator(X, rk, parameters = distribution.parameters) # Parameters of the k-th cluster
                except RuntimeError as err:
        #            error_type = err.args[1]
        #            print err.args[0] % err.args[2]
                    print ("Cluster %i degenerated during estimation" %k)           ####### HANDLE THE DEGENERATED CLUSTER #############
                    theta_k = None;
                theta[k_theta] = theta_k
        return theta
    
    ################# Functions for managing cluster ############################

    def manage_clusters(self, X, r, theta_prev, theta_new):
        """ 
        Here we manage the clusters that fell into singlularities or that their
        pdf cannot be computed,usually because the normalization constant is
        not computable.
        
        In the process we will compute the likelihoods
        """
        clusters_change = 0
        K = len(self.clusterk_to_Dname.keys())
        ##################### Check for singularities (degenerated cluster) ##########
        if (type(theta_prev) != type(None)):  # Not run this part for the initialization one.
            for k in self.clusterk_to_thetak:
                k_theta = self.clusterk_to_thetak[k]
                if(type(theta_new[k_theta]) == type(None)):  # Degenerated cluster during estimation
                    print ("Cluster %i has degenerated during estimation (singularity) "%k_theta)
                    # TODO: Move only the following line inside the function below
                    print (" Xum responsability of samples r: %f"% (np.sum(r[:,[k_theta]]))) 
                    distribution = self.Dname_to_distribution[self.clusterk_to_Dname[k]]
                    theta_new[k_theta] = distribution.degenerated_estimation_handler(
                            X, rk = r[:,[k_theta]] , prev_theta_k = theta_prev[k], parameters = distribution.parameters )
                    
                    clusters_change = 1  # We changed a cluster !
                    
        ################# Check if the parameters are well defined ( we can compute pdf) ################
        for k in self.clusterk_to_thetak:
            k_theta = self.clusterk_to_thetak[k]
            # Checking for the clusters that we are not gonna remove due to degenerated estimation.  
            if(type(theta_new[k_theta]) != type(None)):  
                # We estimate the likelihood of a sample for the cluster, if the result is None
                # Then we know we cannot do it.
                distribution = self.Dname_to_distribution[self.clusterk_to_Dname[k]]
                if (type(distribution.pdf_log_K(X[[0],:].T,[theta_new[k_theta]],parameters = distribution.parameters)) == type(None)):
                    print ("Cluster %i has degenerated parameters "%k)
                    if (type(r) == type(None)): # We do not have rk if this is the first initialization
                        rk = None
                    else:
                        rk = r[:,[k_theta]]
                    # We give the current theta to this one !!! 
                    theta_new[k_theta] = distribution.degenerated_params_handler(
                            X, rk = rk , prev_theta_k = theta_new[k_theta], parameters = distribution.parameters )
                    
                    clusters_change = 1; # We changed a cluster !
                    
        return theta_new, clusters_change
    
class CDistribution ():
    """ This is the distribution object that is given to the program in order to run the algorithm.
    The template is as follows !!
    
    Each distribution has a set of parameters, called "theta" in this case. 
    In the EM we will have "K" clusters, and the distributions are "D" dimensional.
    
    Theta is given as a list where inside it has the parameters. 
    For example for the Gaussian distribution theta = [mus, Sigma]
    Where mus would be a K x D matix and Sigma another DxD matrix.
    
    The probabilities or densities have be given in log() since they could be very small.
    
    TODO: I want the calling of funcitons in a way that when called externally, 
    the parameters are given automatically ?
    Right now, simply:
        - Store the parameters as a dict in the object
        - Externally when calling the funcitons, pass them the dict,
          the functions should have as input the dict
    
    """
    
    def __init__(self, name = "Distribution"):
        
        self.name = name;
        self.pdf_log = None;
        self.pdf_log_K = None;
        self.init_theta = None;
        self.theta_estimator = None;
        self.sampler = None;
        
        ## If we are gonna somehow set new parameters for the clusters according
        ## to some rule, instead of deleting them.
        
        self.degenerated_estimation_handler = None
        self.degenerated_params_handler = None
        self.check_degenerated_params = None

        # Function to use at the end of the iteration to modify the cluster parameters
        self.use_chageOfClusters = None
        
        ########### Hyperparameters ###############
        """
        We can store the hyperparameters in this distribution.
        Examples are:
            - Parameters for initialization of clusters
            - Parameters for singularities
            - Number of iterations of Newton for the parameter estimation
            - Constraints in the Estimation (Only diagonal Gaussian matrix)
        """
        self.parameters = dict()
        
    def set_distribution(self,distribution, parameters = None):
        """ We have a set of preconfigured distributions ready to use """
        if (type(distribution) != type(None)):
            if (distribution == "Watson"):
                # For initialization, likelihood and weighted parameter estimamtion
                self.init_params = Wad.init_params
                self.pdf_log_K = Wad.Watson_K_pdf_log
                self.theta_estimator = Wae.get_Watson_muKappa_ML 
                
                ## For degeneration
                self.degenerated_estimation_handler = Wad.degenerated_estimation_handler 
                self.degenerated_params_handler = Wad.degenerated_params_handler 
                
                ## For optimization
                self.get_Cs_log = Wad.get_Cs_log
                
                ## Optional for more complex processing
                self.use_chageOfClusters = Wad.avoid_change_sign_centroids
                
                if(type(parameters) == type(None)):
                    self.parameters["Num_Newton_iterations"] = 5
                    self.parameters["Allow_negative_kappa"] = "no"
                    self.parameters["Kappa_min_init"] = 0
                    self.parameters["Kappa_max_init"] = 100
                    self.parameters["Kappa_max_singularity"] =1000
                    self.parameters["Kappa_max_pdf"] = 1000
                else:
                    self.parameters = parameters
                
            elif(distribution == "Gaussian"):
                self.init_params = Gad.init_params
                self.pdf_log_K = Gad.Gaussian_K_pdf_log
                self.theta_estimator = Gae.get_Gaussian_muSigma_ML 
                
                ## For degeneration
                self.degenerated_estimation_handler = Gad.degenerated_estimation_handler 
                self.degenerated_params_handler = Gad.degenerated_params_handler 
                
                ## For optimization
                self.get_Cs_log = Gad.get_Cs_log
                
                ## Optional for more complex processing
                self.use_chageOfClusters = None
    
                if(type(parameters) == type(None)):
                    self.parameters["mu_variance"] = 1
                    self.parameters["Sigma_min_init"] = 1
                    self.parameters["Sigma_max_init"] = 15
                    self.parameters["Sigma_min_singularity"] = 0.1
                    self.parameters["Sigma_min_pdf"] = 0.1
                    self.parameters["Sigma"] = "diagonal" # "full", "diagonal"
                else:
                    self.parameters = parameters
        

            elif (distribution == "vonMisesFisher"):
                # For initialization, likelihood and weighted parameter estimamtion
                self.init_params = vMFd.init_params
                self.pdf_log_K = vMFd.vonMisesFisher_K_pdf_log
                self.theta_estimator = vMFe.get_vonMissesFisher_muKappa_ML 
                
                ## For degeneration
                self.degenerated_estimation_handler = vMFd.degenerated_estimation_handler 
                self.degenerated_params_handler = vMFd.degenerated_params_handler 
                
                ## For optimization
                self.get_Cs_log = vMFd.get_Cs_log
                
                ## Optional for more complex processing
#                self.use_chageOfClusters = Wad.avoid_change_sign_centroids
                
                if(type(parameters) == type(None)):
                    self.parameters["Num_Newton_iterations"] = 2
                    self.parameters["Kappa_min_init"] = 0
                    self.parameters["Kappa_max_init"] = 100
                    self.parameters["Kappa_max_singularity"] =1000
                    self.parameters["Kappa_max_pdf"] = 1000
                else:
                    self.parameters = parameters
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    """
    ################################################################################
    ##################### TEMPLATE FUNCTIONS #######################################
    ################################################################################
    """        
            
    
        ### Funcitons and shit 
    def pdf_log_K(X, theta, Cs_log = None):
        return None
    """
    This function returns the pdf in logarithmic units of the distribution.
    It accepts any number of samples N and any number of set of parameters K
    (number of clusters) simultaneuously. For optimizaiton purposes, if the 
    normalization constants are precomputed, they can be given as input.
    
    Inputs:
        - X: (DxN) numpy array with the samples you want to compute the pdf for.
        - theta: This is the set of parameters of all the clusters. 
                 Theta is a list with the set of parameters of each cluster
                     
                 theta = [theta_1, theta_2,..., theta_K]

                 The format in which the parameters of each cluster theta_i are
                 indicated are a design choice of the programmer. 
                 We recomment using another list for each of ther parameters.
                 For example for the gaussian distribution:
                     
                     theta_i = [mu_i, Sigma_i]
    
                 Where mu_i would be a Dx1 vector and Sigma_i a DxD matrix
                 
        - Cs_log: Some distributions have normalization constant that is 
        expensive to compute, you can optionally give it as input if you
        already computed somewhere else to save computation.
        Note: If you provide the computing function, this optimization is done 
        automatically within the implementation.
        
    Outputs:
        - log_pdf: numpy 2D array (NxK) with the log(pdf) of each of the samples
                    for each of the cluters
    """
        

    def get_Cs_log(theta_k, parameters = None):
        return None
    """
    This function will compute the normalization constant of a cluster.
    Usually, the normalization constant is a computaitonal bottelneck since it
    could take quite some time to compute. In an iteration of the EM algoeithm,
    it should only be computed once per cluster. If given, the computation
    of such constants can be minimized.
    
    Input:
       - theta_k: The parameters of the cluster in the previous iteration.
       
    Output:
        - Cs_log: Normalization constant of the cluster
    
    """
        
    def init_params(X,K, theta_init = None, parameters = None):
        return None
    """
    This function initializes the parameters of the K clusters if no inital 
    theta has been provided. If a initial theta "theta_init" is specified when 
    calling the "fit()" function then that initialization will be used instead. 
    The minimum parameters to be provided are the number of cluster K
    and the dimensionality of the data D. 
    In order to add expressivity we can use parameters of the dictionary

    
    Inputs:
        - K: Number of clusters to initialize
        - X: The input samples to use. We can use them to initilize.
        
        - theta_init: Optional parameter. If we specified externally an initialization
          then this theta will bypass the function. It needs to be specified in the interface
          to be used internally by the algorithm.
        - parameters: Dictionary with parameters that can be useful for the initialization.
          It is up to the programmers how to use it. An example of use in the Gaussian Distribution,
          is setting the maximum variance that the initialized clusters can have.
          
    Outputs:
        - theta_init: This is the set of parameters of all the clusters. 
                    Its format is the same as previously stated.
    """

    
    def theta_estimator(X, rk = None, parameters = None):
        return None
    """
    This function estimates the parameters of a given cluster k, k =1,...,K
    given the datapoint X and the responsability vector rk of the cluster.
    
    Input:
       - X: (DxN) numpy array with the samples.
       - rk: Nx1 numpy vector with the responsibility of the cluster to each sample
       - parameters: Dictionary with parameters useful foe the algorithm.
    Output:
        - theta_k: The parameters of the cluster. This is distribution dependent
           and its format must the coherent with the rest of the functions.
    
    Note: It is likely for this function to fail due to degenetate clusters. For example
         trying to compute the variance of one point. 
         The user should handle such exeptions, catching the possible numerical errors,
         if the computation is not possible, then this funciton should return None
         and the logic will be handled later.
         
         A recommended way of doing this is calling try - execpt XXXXXX
    """
        
       
    
    def degenerated_estimation_handler(X, rk , prev_theta_k , parameters = None):
        return None
    """
    If during the estimation of the parameters there was a numerical error and
    the computation is not possible, this function will try to solve the situation.
    Some common solutions are to use the previous parameters theta_k of the cluster
    or reinitilizite it using other hyperperameters.
    
    Input:
       - X: (DxN) numpy array with the samples.
       - rk: Nx1 numpy vector with the responsibility of the cluster to each sample
       
       - prev_theta_k: The parameters of the cluster in the previous iteration.
         It can be used to replace the current one.
       - parameters: Dictionary with hyperparameters if needed in order to
         compute the new parameters of the cluster
    
    Output:
        - theta_k: The parameters of the cluster. Distribution dependent.
                   If no recomputation method is possible, then this function 
                   must return "None" in which case the cluster will be later removed.

    """
        
    
    
    def degenerated_params_handler(X, rk , prev_theta_k , parameters = None):
        return None
    """
    It at some point the obtained parameters of the cluster makes it non-feasible
    to compute the pdf of the data, for example because the normalization constant
    is too big or too small, its inaccurate, or it takes too much time to compute then
    this function will attempt to recompute another set of parameters.
    
    Input:
       - X: (DxN) numpy array with the samples.
       - rk: Nx1 numpy vector with the responsibility of the cluster to each sample
       
       - prev_theta_k: The parameters of the cluster in the previous iteration.
         It can be used to replace the current one.
       - parameters: Dictionary with hyperparameters if needed in order to
         compute the new parameters of the cluster
    
    Output:
        - theta_k: The parameters of the cluster. Distribution dependent.
                   If no recomputation method is possible, then this function 
                   must return "None" in which case the cluster will be later removed.
    
    """
        
    
    
    def use_chageOfClusters(theta_new, theta_prev):
        """
        At the end of each iteration of the EM-algorithm we have updated the cluster parameters.
        We might want to do some operation on them depending on this change so this function
        will add some more capabilities to the algorithm.
        
        Input:
            - theta_new: The newly computed theta
            - theta_prev: The previously computed theta
            
        Output:
            - theta_new: The modified new parameters.
        
        """
        
    