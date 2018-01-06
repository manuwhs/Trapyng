import numpy as np
import HMM_libfunc as HMMlf
import copy
import time


def HMM(data, distribution,
       init_hyperparams = None, deged_est_params = None, deged_params = None, 
        I = 3,delta_ll = 0.01,R = 30,
        pi_init = None, A_init = None, theta_init = None,
        verbose = 0):
    
    perfrom_timing = 1
    
    #########################################################
    # data = list of realizations, every realization can have a different number of samples 


    N = len(data)         # Number of Realizations of the HMM
    D = data[0].shape[1]; # Dimensionality of samples

    ######## THIS IS A CONCATENATED VERSION OF THE SAMPLES THAT WE NEED
    ## For some parts of the algorithm we need all the samples in a vector, for
    # example when computing the weighted estimatoin with responsability vector rk. 
    # We take into account different chains, so we also use data
    X = data[0]
    for n in range(1,N):
        X = np.concatenate((X, data[n]), axis = 0)
                
    #*********** INITIALIZATION *****************************
    
    pi, A = HMMlf.init_HMM_params(D,I,pi_init, A_init)
    theta = distribution.init_params(I,D, theta_init, init_hyperparams = init_hyperparams)
    
    #*********************************************************
    #*********** ITERATIONS OF THE HMM ***********************
    #*********************************************************
    
    logl = []   # List where we store the likelihoos
    theta_list = []
    pi_list = []
    A_list = []
    
    pi_list.append(copy.deepcopy(pi))
    theta_list.append(copy.deepcopy(theta))
    A_list.append(copy.deepcopy(A))
    
    #### Timings dictionary #######
    # This list is suposed to obtain the 
    times_list =[]
    

    ### Initial check that the initialization clusters are fine
    theta, pi, A, clusters_change = HMMlf.manage_clusters(data,None, distribution, pi, A, theta_new = theta, theta_prev = None, 
                                                             deged_est_params = deged_est_params, deged_params = deged_params)
    for r in range(R):         # For every iteration of the EM algorithm
        if (verbose > 1):
            print "Iteration %i"%(r)
        
        time_dictionary = dict();

        #******************************************************  
        #*********** E Step ***********************************
        #******************************************************
        
        if(perfrom_timing):
            init_time = time.time()
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # In this step we calculate the alfas, betas, gammas and fis matrices
        # If we changes the clusters we have, we need to recompute alpha.
        # Otherwise we reuse it since we needed to compute the ll for the previous iteration
    
        if (clusters_change):
            alpha = HMMlf.get_alfa_matrix_log(data, pi, A,theta,distribution);

        if (r == 0):
            ## ALPHA is recomputed ar the end to 
            alpha = HMMlf.get_alfa_matrix_log(data, pi, A,theta,distribution)
            # Compute the initial incomplete-loglikelihood
            ll = HMMlf.get_HMM_Incomloglike(data, pi, A,theta, distribution, alpha = alpha)
            logl.append(ll)
            if (verbose > 1):
                print "Initial loglikelihood: %f" % ll

        

            
        if(perfrom_timing):
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FINISH TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            final_time = time.time()
            time_dictionary["E-Step"] = final_time - init_time
            print "E-Step: %f" %(final_time - init_time)
        
        #*****************************************************   
        #*********** M Step ***********************************
        #*****************************************************
        # In this step we calculate the next parameters of the HMM
        if(perfrom_timing):
            init_time = time.time()
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        pi = HMMlf.get_pi(gamma)
        A = HMMlf.get_A(fi)
        if(perfrom_timing):
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FINISH TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            final_time = time.time()
            time_dictionary["M-Step pi,A"] = final_time - init_time
            print "M-Step pi,A %f" %(final_time - init_time)
        if(perfrom_timing):
            init_time = time.time()
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        theta_new = HMMlf.get_theta(X, gamma, theta, distribution)
        if(perfrom_timing):
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FINISH TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            final_time = time.time()
            time_dictionary["M-Step theta"] = final_time - init_time
            print "M-Step theta %f" %(final_time - init_time)
        
        #*****************************************************************   
        #*********** Check that no degeneration happened  *****************
        #*****************************************************************
        # Some models can kind of die if the clusters converge badly.
        # For example if we have too many clusters, one of them could go to an outlier,
        # And the resposibility of that point would be 100% from that cluster
        # And the parameters of the cluster cannot be properly computed.
        # It is up to the stability function to modify the parameters.
        # Maybe removing the cluster or reininitializing the cluster randomly
        # TODO: Manage case where r = None and theta_prev = None since it is the first. Special shit may apply.
        if(perfrom_timing):
            init_time = time.time()
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        theta, pi, A, clusters_change = HMMlf.manage_clusters(data,gamma, distribution, pi,A, theta_new = theta_new, theta_prev = theta, 
                                                                 deged_est_params = deged_est_params, deged_params = deged_params)
        if(perfrom_timing):
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FINISH TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            final_time = time.time()
            time_dictionary["Manage_clusters"] = final_time - init_time
            print "Number of seconds cluster manage: %f" %(final_time - init_time)

        #********************************************************* 
        #****** Calculate Incomplete log-likelihood  *************
        #*********************************************************
        if(perfrom_timing):
            init_time = time.time()
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TIMING $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Calculate Incomplete log-likelihood with the Forward Algorithm
        alpha = HMMlf.get_alfa_matrix_log(data, pi, A,theta,distribution)
        new_ll = HMMlf.get_HMM_Incomloglike(data, pi, A,theta, distribution, alpha = alpha)
        if(perfrom_timing):
            final_time = time.time()
            time_dictionary["Compute final LL of iteration"] = final_time - init_time
            print "Compute final LL of iteration: %f" %(final_time - init_time)
        
        if (verbose > 1):
            print "Loglkelihood: %f " % new_ll
        #***************************************************** 
        #****** Convergence Checking *************************
        #*****************************************************
 
        logl.append(new_ll)
        pi_list.append(copy.deepcopy(pi))
        theta_list.append(copy.deepcopy(theta))
        A_list.append(copy.deepcopy(A))
    
        if (clusters_change == 0): # If we did not have to delete clusters
            if(new_ll-ll <= delta_ll): # Maybe abs
                break;
            else:
                ll = new_ll;
        else:
            ll = new_ll

    if (verbose > 0):
        print "Final Loglkelihood: %f " % new_ll
        
    return logl,theta_list,pi_list, A_list


def run_several_HMM(data, distribution,
       init_hyperparams = None, deged_est_params = None, deged_params = None, 
        I = 3,delta_ll = 0.01,R = 30,
        pi_init = None, A_init = None, theta_init = None,  Ninit = 5,
        verbose = 0):
    if (verbose > 0):
        print "HMM number 1/%i" % (Ninit)
# We make a first run of the HMM
    [logl,B_list,pi_list, A_list] = HMM(data, distribution,
       init_hyperparams , deged_est_params , deged_params , 
        I ,delta_ll ,R ,
        pi_init, A_init , theta_init ,
        verbose)
    best_logl = logl;   
    best_pi = pi_list;     
    best_B = B_list;   
    best_A = A_list
    best_final_logll = logl[-1]
    
    if (Ninit > 1):
        for i in range(1,Ninit):
            if (verbose > 0):
                print "HMM number %i/%i" % (i+1,Ninit)
            [logl,B_list,pi_list, A_list]  = HMM(data, distribution,
       init_hyperparams , deged_est_params , deged_params , 
        I ,delta_ll,R,
        pi_init, A_init , theta_init ,
        verbose)
            if (logl[-1] > best_final_logll):
                best_logl = logl;   
                best_pi = pi_list;     
                best_B = B_list;   
                best_A = A_list
                best_final_logll = logl[-1]
    
    return [best_logl, best_B, best_pi, best_A]
