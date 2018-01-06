import numpy as np
import copy 
import EM_libfunc as EMlf
import HMM_libfunc as HMMlf
import time
    
def EM(data, distributionsManager, clusters_relation = "independent",
       T = 30, delta_ll = 0.01,
       model_theta_init = None, theta_init = None, 
       verbose = 0, time_profiling = None):

    ################ Preprocess the input #############################
#    print (clusters_relation)
    if (clusters_relation == "independent"):
        X = EMlf.preprocess_data(data)
        
    elif (clusters_relation == "MarkovChain1"):
        ## If we are not ginve a list but a set of samples in a ndarray, we only have one chain
        if (type(data) != type([])):
            data = [data]
        N = len(data)         # Number of Realizations of the HMM
        N, D = data[0].shape; # Dimensionality of samples
    
        ######## THIS IS A CONCATENATED VERSION OF THE SAMPLES THAT WE NEED
        ## For some parts of the algorithm we need all the samples in a vector, for
        # example when computing the weighted estimatoin with responsability vector rk. 
        # We take into account different chains, so we also use data

        X = np.concatenate(data, axis = 0)
    else:
        print ("Wrong Cluster Relations")
            
    ##################  Create data structures for output at each iteration ####################################
    logl = []           # List where we store the likelihoods
    theta_list = []     # List with cluster parameters
    model_theta_list = []     # List with the model parameters
    
    ########################################################################
    ##################  INITIALIZATION ####################################
    ########################################################################
    K = len(distributionsManager.clusterk_to_Dname.keys())
    if (clusters_relation == "independent"):
        # model_theta_init = [pi]
        model_theta = EMlf.init_model_params(K,model_theta_init)
    elif(clusters_relation == "MarkovChain1"):
        # model_theta_init = [pi, A]
        model_theta= HMMlf.init_model_params(K,model_theta_init)
        
    theta = distributionsManager.init_params(X,theta_init)
    
    ########## Stability Check #############
    # TODO: Manage case where r = None and theta_prev = None since it is the first. Special shit may apply.
    
    if (clusters_relation == "independent"):
        # model_theta_init = [pi]
        model_theta = EMlf.init_model_params(K,model_theta_init)

        theta, model_theta, clusters_change = EMlf.manage_clusters(X,None, distributionsManager, 
                                                                   model_theta, theta_new = theta, theta_prev = None)
    
    elif(clusters_relation == "MarkovChain1"):
        # model_theta_init = [pi, A]
        model_theta= HMMlf.init_model_params(K,model_theta_init)
        theta, model_theta, clusters_change = HMMlf.manage_clusters(X,None, distributionsManager, 
                                                                   model_theta, theta_new = theta, theta_prev = None)

    ll = -1e500  # -inf

    theta_list.append(copy.deepcopy(theta))
    model_theta_list.append(copy.deepcopy(model_theta))
    
    for t in range(T):    #For every iteration of the EM algorithm
        # T is the maximum number of iterations, if we stop before due to convergence
        # we will break the loop
        if (verbose > 1):
            print ("Iteration %i"%t)
        
        """ 
        Compute the likelihood for each sample N to each cluster K.
        This is the only info we need about the samples later
        """
        
        ####### Compute the Loglikelihood of each sammple to each cluster ####

        #**********************************************************************************************  
        #*********** E Step ***************************************************************************
        #**********************************************************************************************
        if (clusters_relation == "independent"):
            loglike_samples = EMlf.get_samples_loglikelihood(X,theta,distributionsManager)
            r, new_ll = EMlf.get_r_and_ll(X,distributionsManager, theta,model_theta,loglike = loglike_samples)
            
        elif(clusters_relation == "MarkovChain1"):
            loglike_samples = HMMlf.get_samples_loglikelihood(data,theta,distributionsManager)
            gamma, fi, new_ll = HMMlf.get_r_and_ll(data,distributionsManager, theta,model_theta,loglike = loglike_samples)
        
#        print ("r.shape",r.shape)
        #*********************************************************************************************   
        #*********** M Step ***************************************************************************
        #*********************************************************************************************
        # In this step we calculate the next parameters of the mixture mdoel
        #Calculate new pimix and update
        if (clusters_relation == "independent"):
            model_theta = EMlf.get_model_theta(r)
#            model_theta = [pimix]
            
            theta_new = EMlf.get_theta(X, r,distributionsManager)
        elif(clusters_relation == "MarkovChain1"):
            model_theta = HMMlf.get_model_theta(gamma,fi)
#            model_theta = [pi, A]
            
            theta_new = HMMlf.get_theta(X, gamma, theta, distributionsManager)

        # Calculate new thetas parameters and update
        if (clusters_relation == "independent"):
            theta, model_theta, clusters_change = EMlf.manage_clusters(X,r, distributionsManager, 
                                                                       model_theta, theta_new = theta_new, theta_prev = theta)
        elif(clusters_relation == "MarkovChain1"):
            theta, model_theta, clusters_change = HMMlf.manage_clusters(X,gamma, distributionsManager, 
                                                                       model_theta, theta_new = theta_new, theta_prev = theta)
        
#        print clusters_change
        #************************************************************************************************* 
        #****** Calculate Incomplete log-likelihood  *****************************************************
        #*************************************************************************************************
        
#        Now this is merged into a single function in Responsability obtaining
#        We actually do one iteration more !! We detect it one iteration delayed
#        new_ll = EMlf.get_EM_Incomloglike_log(theta,pimix,X)

        if (verbose > 1):        
            print ("Loglk: %f"%(new_ll))

        #***************************************************** 
        #****** Convergence Checking *************************
        #*****************************************************
        
        logl.append(new_ll) # This is for the previous one
        theta_list.append(copy.deepcopy(theta))
        model_theta_list.append(copy.deepcopy(model_theta))
    
        if (t == T-1):  # If we are done with this
#            new_r, new_ll = EMlf.get_r_and_ll(X,distributionsManager, theta,model_theta)
            logl.append(new_ll)
            break
        
        # If we changed the clusters but the likelihoods are the same, then we stop
        # Because it is not going to evolve
        if (clusters_change == 0 or (ll == new_ll)): # If we did not have to delete clusters
            if(np.abs(new_ll-ll) <= delta_ll):

                # Compute the last Loglikelihood
#                new_r, new_ll = EMlf.get_r_and_ll(X,distributionsManager, theta,model_theta)
                logl.append(new_ll)
                break;
            else:
                ll = new_ll;
        else:
            ll = new_ll
#        print'  R:'
#        print r;
    if (verbose > 0):
        print ("Final ll: %f"% (logl[-1]))
    return logl,theta_list,model_theta_list

# Runs the ME N times and chooses the realization with the least loglihood
def run_several_EM(data, distribution, clusters_relation = "independent", 
                   T = 30, Ninit = 5,  delta_ll = 0.01,
                   model_theta_init = None, theta_init = None, 
                   verbose = 0, time_profiling = None):
    """
    Input:
        - data: Input data from which we will learn the distribution. The accepted formats are:
              - For "independent":
                  - np.ndarray(N,D): Bidimensional numpy array with dimensions
                    (N = Number of Samples) x (D = Dimensionality of input)
                  - List: Containing ndarrays of dimension (Ni,D). They will be concatenated
                  
              - For "MarkovChain":
                  - List: Containing ndarrays of dimension (Ni,D). Each element of the list
                    will be chain for the algorithm.
                    
        - distribution: Distribution manager with the different distributions of the mixture and their
           associated clusters.
        - clusters_relation: Dependency between the clusters:
              - For "independent": Independent clusters
              - For "MarkovChain1": MC order 1
        - delta_ll: If in an iteration the ll does not increase by this minimum value, the iterations stop-
        - T: Number maximum of iterations.
        - model_theta_init: Initial parameters of the model. If not given they will be initilized uniformly
          This is [pi] for "independent" and [pi, A] for "MarkovChain1".
        - theta_init: Initial parameters of the clusters. If not provided, the initializer provided in the
          distribution objects will be used.
        - Ninit: Number of random initilizations, for this puporse a random initilizer for the distributions
                 should be provided or all the initializations will have the same cluster parameters.
        - verbose: The higher the value, the more partial outputs will be printed in the screen.
        - time_profiling: If we want to see the times it takes for operations.
        
    Output:
        - logl: List of incomplete loglikelihoods asociated to each iteration.
          Notice it will have always the added initial loglikelihood asociated 
          to the initialization. So maximum it will be "T+1" elements.
        - theta_list: List of the cluster parameters for each of iterations.
        - model_theta_init: List of the model parameters for each of the iterations
    """

    if (verbose > 0):
        print ("EM number 1/%i" % Ninit)
# We make a first run of the HMM
    [logl,theta_list,model_theta_list] = EM(data,distribution,clusters_relation,
                                        T, delta_ll, model_theta_init,theta_init, verbose, time_profiling)
    best_logl = logl;   
    best_model_theta = model_theta_list;     
    best_theta = theta_list;     
    best_final_logll = logl[-1]
    
    if (Ninit > 1):
        for i in range(1,Ninit):
            if (verbose > 0):
                print ("EM number %i/%i" % (i+1,Ninit))
            [logl,theta_list,model_theta_list] = EM(data,distribution,clusters_relation,
                                    T, delta_ll, model_theta_init,theta_init, verbose, time_profiling)
            
            if (logl[-1] > best_final_logll):
                best_logl = logl;   
                best_model_theta = model_theta_list;     
                best_theta = theta_list;     
                best_final_logll = logl[-1]
    
    return [best_logl, best_theta, best_model_theta]
