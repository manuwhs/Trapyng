import general_func as gf
import copy
import numpy as np



def get_alfa_matrix_log(data, pi, A,theta,distribution, loglike = None):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    alfa = []
#    print(I,N,D,T)
    # Calculate first sample
    for n in range(N): # For every chain
        alfa.append(np.zeros((I,T[n])));
        for i in range(I):  # For every state
            alfa[n][i,0] = np.log(pi[:,i]) + loglike[n][[0], i]  # distribution.pdf_log_K(data[n][[0],:].T,theta,  [cp_logs[i]]);  # Maybe need to transpose

    # Calculate the rest of the alfas recursively
    for n in range(N):          # For every chain
        for t in range(1, T[n]):           # For every time instant
            aux_vec = np.log(A[:,:]) + alfa[n][:,[t-1]]
            alfa[n][:,[t]] = gf.sum_logs(aux_vec.T,  byRow = True)
#            print sum_logs(aux_vec.T,  byRow = True).shape
#            print alfa[n][:,[t]].shape
            alfa[n][:,[t]] +=  loglike[n][[t], :].T  # distribution.pdf_log_K(data[n][[t],:].T,theta, cp_logs).T
#                print np.log(Wad.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))
#                print    np.log(Wad.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))# alfa[i,n,t] 

#            for i in range(I):      # For every state
#                aux_vec = np.log(A[:,[i]]) + alfa[n][:,[t-1]]
#                alfa[n][i,t] = sum_logs(aux_vec)
#                alfa[n][i,t] =  Wad.Watson_pdf_log(data[n][[t],:].T, B[0][:,i], B[1][:,i], cp_log = cp_logs[i]) + alfa[n][i,t] ;
                
    return alfa
    
def  get_beta_matrix_log( data, pi, A,theta, distribution, loglike = None):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    beta = [];
    
    # Calculate the last sample
    for n in range(N): # For every chain
        beta.append(np.zeros((I,T[n])));
        Nsam, Nd = data[n].shape
#        pi_end = get_final_probabilities(pi,A,Nsam)
        
        for i in range(I):
            beta[n][i,-1] = 0 # np.log( pi_end[0,i])
#        print beta[n][:,-1]
#            beta[n][i,-1] = np.log( pi_end[0,i]) + Wad.Watson_pdf_log(data[n][-1,:], B[0][:,i], B[1][:,i], cp_log = cp_logs[i]);

#            aux_vec = []
#            for j in range(J):
#                aux_vec.append(pi_end[:,i])
#                np.log( pi_end[:,i]) + Wad.Watson_pdf_log(data[n][0,:], B[0][:,i], B[1][:,i], cp_log = cp_logs[i]);

    # Calculate the rest of the betas recursively
    for n in range(N):     # For every chain
        for t in range(T[n]-2,-1,-1):  # For every time instant backwards
            aux_vec = np.log(A[:,:]) +  beta[n][:,[t+1]].T + \
            loglike[n][[t+1], :]
            #distribution.pdf_log_K(data[n][[t+1],:].T, theta ,cp_logs)
            beta[n][:,[t]] = gf.sum_logs(aux_vec, byRow = True)
                
    return beta

def get_gamma_matrix_log( alpha,beta ):
    I = alpha[0].shape[0]
    N = len(alpha)
    T = [x.shape[1] for x in alpha]

    gamma = []
    
    for n in range(N):
        gamma.append(np.zeros((I,T[n])))
        for t in range (0, T[n]):
            gamma[n][:,t] = alpha[n][:,t] + beta[n][:,t];
    
    for n in range(N):
        for t in range(T[n]):
            #Normalize to get the actual gamma
            gamma[n][:,t] = gamma[n][:,t] - gf.sum_logs(gamma[n][:,t]);  

    return gamma

def  get_fi_matrix_log( data, A, theta, alpha,beta, distribution, loglike = None):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]

    fi = []
    
    for n in range(N):
        fi.append(np.zeros((I,I,T[n]-1)))

        zurullo = np.log(A[:,:])
        zurullo = zurullo.reshape(zurullo.shape[0],zurullo.shape[1],1)
        zurullo = np.repeat(zurullo,T[n]-1,axis = 2)
        
        mierda1 = beta[n][:,1:] 
        mierda1 = mierda1.reshape(1,mierda1.shape[0],mierda1.shape[1])
        mierda1 = np.repeat(mierda1,I,axis = 0)
        
        mierda2 = alpha[n][:,:-1] 
        mierda2 = mierda2.reshape(mierda2.shape[0],1,mierda2.shape[1])
        mierda2 = np.repeat(mierda2,I,axis = 1)
        
        # caca = distribution.pdf_log_K(data[n][1:,:].T, theta, cp_logs).T
        caca = loglike[n][1:,:].T
        caca = caca.reshape(1,caca.shape[0],caca.shape[1])
        caca = np.repeat(caca,I,axis = 0)
        
        fi[n][:,:,:] = zurullo + caca + mierda1 + mierda2
        
    for n in range(N):
        if(1):
            for t in range (0, T[n]-1):
                # Normalize to get the actual fi
                fi[n][:,:,t] = fi[n][:,:,t] - gf.sum_logs(fi[n][:,:,t]);  

    return fi

def get_samples_loglikelihood(data,theta,distribution, Cs_logs = None):
    """
    This function simply computes the likelihood for each sample, to each of the
    clusters 
    """
    loglike = distribution.pdf_log_K(data,theta)

    return loglike

def get_r_and_ll(data, distribution, theta, model_theta,loglike = None):
    N = len(data)
    pi = model_theta[0]
    A = model_theta[1]
    
    if type(loglike) != type(None):
        loglike = loglike
    else:
        cp_logs = distribution.get_Cs_log(theta)
        loglike = get_samples_loglikelihood(data, theta,distribution , Cs_logs = cp_logs)

    alpha = get_alfa_matrix_log(data, pi, A,theta,distribution, loglike = loglike)
    beta = get_beta_matrix_log(data, pi, A,theta, distribution, loglike = loglike);
    gamma = get_gamma_matrix_log(alpha,beta );
    fi = get_fi_matrix_log( data, A, theta, alpha,beta, distribution, loglike = loglike);
    
    ## Reconvert to natural units
    for n in range(N):
        gamma[n] = np.exp(gamma[n])
        fi[n] = np.exp(fi[n])
    
    new_ll = get_loglikelihood(data ,distribution,theta, model_theta , alpha = alpha)
    
    return gamma, fi, new_ll

def get_responsibilities(data, distribution, theta, model_theta,loglike = None):
    N = len(data)
    pi = model_theta[0]
    A = model_theta[1]
    
    if type(loglike) != type(None):
        loglike = loglike
    else:
        cp_logs = distribution.get_Cs_log(theta)
        loglike = get_samples_loglikelihood(data, theta,distribution , Cs_logs = cp_logs)
        
    alpha = get_alfa_matrix_log(data, pi, A,theta,distribution, loglike = loglike)
    beta = get_beta_matrix_log(data, pi, A,theta, distribution, loglike = loglike);
    gamma = get_gamma_matrix_log(alpha,beta );

    ## Reconvert to natural units
    for n in range(N):
        gamma[n] = np.exp(gamma[n])
        gamma[n]  =  gamma[n].T

    r = gamma
    
    return r

def get_model_theta(gamma,fi):
    pi = get_pi(gamma)
    A = get_A(fi)
    return [pi,A]

def get_pi(gamma):
    
    # Calculate new initial probabilities
    N = len(gamma)
    I = gamma[0].shape[0]

    pi = np.zeros((1,I))
    N_gamma = []
    for n in range(N):
        N_gamma.append (np.sum(gamma[n][:,0]));
    N_gamma = np.sum(N_gamma)
    
    for i in range(I):
        aux = []
        
        for n in range(N):
            aux.append(gamma[n][i,0])

        N_i_gamma = np.sum(aux)
        pi[0,i] = N_i_gamma/N_gamma;
        
    pi = pi.reshape(1,pi.size)
    return pi

def get_A(fi):

# Calculate transition probabilities A

    I = fi[0].shape[0]
    N = len(fi)
    A = -np.ones((I,I))

    for i in range(I):
#        print range(I)
        E_i_fi = []
        # Calculate vector ai = [ai1 ai2 ... aiJ]  sum(ai) = 1
        for n in range(N): 
            E_i_fi.append(np.sum(np.sum(fi[n][i,:,:])))
        E_i_fi = np.sum(E_i_fi) + 1e-200
        
        for j in range(I):
            E_ij_fi = []
            for n in range(N): 
                E_ij_fi.append(np.sum(fi[n][i,j,:]))
            E_ij_fi = np.sum(E_ij_fi)
            
            A[i,j] = E_ij_fi/E_i_fi;
            
#        print "A"
#        print A
  
    return A


def init_model_params(I,model_theta_init = None):
    # Here we will initialize the  parameters of the HMM structure, that is,the initial
    # probabilities of the state "pi", the transition probabilities "A".

    # Initial probabilities
    # We set the Initial probabilities with uniform discrete distribution, this
    # way, the a priori probability of any vector to belong to any component is
    # the same.
    if (type(model_theta_init) != type(None)):
        pi_init = model_theta_init[0]
        A_init = model_theta_init[1]
        
    if (type(model_theta_init) == type(None)): # If not given an initialization
        pi = np.ones((1,I));
        pi = pi*(1/float(I));
    else:
        pi = np.array(pi_init).reshape(1,I)
        
    # Transition probabilities "A"
    # We set the Transition probabilities with uniform discrete distribution, this
    # way, the a priori probability of going from a state i to a state j is
    # the same, no matter the j.

    if (type(model_theta_init) == type(None)): # If not given an initialization
        A = np.ones((I,I));   #A(i,j) = aij = P(st = j | st-1 = i)  sum(A(i,:)) = 1
        for i in range(I):
            A[i,:] =  A[i,:]*(1/float(I));
    else:
        A = A_init
        
    return [pi, A]


def get_theta(X, gamma, theta, distribution):
    """ This function aims to estimate the new theta values for all the clusters.
        For each cluster it will call the estimation function "distribution.theta_estimator(X, rk)".
        If it fails, it should create a RuntimeExeption, which is handled here by setting the parameters to None.
        This will be handled later.
    
    """
    # Cluster by Cluster it will estimate them, if it cannot, then it 
    
    # We only need the old mus for checking the change of sign
    # Maybe in the future get this out
    
    N = len(gamma)
    I = gamma[0].shape[0]
    D = X.shape[1]

    # Estimated theta
    r = []
    for i in range(I): # For every cluster
         # We compute the gamma normalization of the cluster
        # The contribution of every sample is weighted by gamma[i,n,t];
        # The total responsibility of the cluster for the samples is N_i_gamma
        rk = gamma[0][i,:]
        for n in range(1,N):
            rk = np.concatenate((rk, gamma[n][i,:]), axis = 0)
        rk = rk.reshape(rk.size,1)
        r.append(rk)
    
    r = np.concatenate(r, axis = 1)
    theta = distribution.get_theta(X, r)
        
    return theta

def get_final_probabilities(pi,A,N):
    Af = A
    for i in range(N-1):
        Af = Af.dot(A)
#        print Af
    pif = pi.dot(Af)

    return pif
    
def get_loglikelihood(data ,distribution,theta, model_theta,alpha = None):

    N = len(data)
    pi, A = model_theta
    I = pi.size
    # Check if we have been given alpha so we do not compute it
    if (type(alpha) == type(None)):
        cp_logs = distribution.get_Cs_log(theta)
        loglike = distribution.pdf_log_K(data, theta, Cs_logs = cp_logs)
        alpha = get_alfa_matrix_log(data, pi, A,theta,distribution, loglike = loglike)
#        print len(alpha),N
    new_ll = 0

    for n in range(N):    # For every HMM sequence
        ## Generate probablilities of being at the state qt = j in the last point
        # of the chainfs  
        Nsam, Nd = data[n].shape
#        pi_end = get_final_probabilities(pi,A,Nsam)
        all_val = []
#        print pi_end.shape
#        print np.log(pi_end)
#        print I

        for i in range(I):
            all_val.append(alpha[n][i,-1]) # + np.log(pi_end)[0,i]  + np.log(pi_end)[0,i]
#            print np.log(pi_end)[0,i]
#        print all_val
        new_ll = new_ll +  gf.sum_logs(all_val)#  sum_logs(alpha[n][:,-1] + np.log(pi_end).T);
        
    return new_ll

def get_HMM_Incomloglike_beta(A,B,pi,data, beta = []):

    N = len(data)
    I = pi.size
    # Check if we have been given alpha so we do not compute it
    if (len(beta) == 0):
        beta = get_beta_matrix_log(A,B,pi,data)
    new_ll = 0
    cp_logs = []
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    kappas = B[1]
    for i in range(I):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,i]))

    for n in range(N):    # For every HMM sequence
        ## Generate probablilities of being at the state qt = j in the last point
        # of the chainfs  
        Nsam, Nd = data[n].shape
        all_val = []
#        print pi_end.shape
#        print np.log(pi_end)
#        print I

        for i in range(I):
            all_val.append(beta[n][i,0] + np.log(pi)[0,i] + Wad.Watson_pdf_log(data[n][0,:], B[0][:,i], B[1][:,i], cp_log = cp_logs[i]))
#            print np.log(pi_end)[0,i]
#        print all_val
        new_ll = new_ll +  sum_logs(all_val)#  sum_logs(alpha[n][:,-1] + np.log(pi_end).T);
        
    return new_ll
    
def get_errorRate(real, pred):
    Nfails = 0
    Ntotal = 0
    for i in range(len(real)):
        T = np.array(real[i]).size
        
        for t in range(T):
    #        print decoded[i][t]
    #        print  HMM_list[i][t]
            if (int(real[i][t]) != int( pred[i][t])):
                Nfails += 1
            Ntotal += 1
    Failure = 100 * float(Nfails)/Ntotal
    return Failure

def remove_A(A, k):
    # Remove one of the states from A, we delete from both dimensions
    # and renormalize
    A = np.delete(A, k, axis = 1)
    A = np.delete(A, k, axis = 0)
    Asum = np.sum(A, axis = 1)
    A = A/ Asum
    return A
    
def remove_cluster( theta, model_theta, k):
    # This function removed the cluster k from the parameters
    pi,A = model_theta
    theta.pop(k)
    pi = np.delete(pi, k, axis = 1)
    pi = pi / np.sum(pi)
    A = remove_A(A, k)
    print ("$ Cluster %i removed" % (k))
#    print A.shape
#    print pi.shape
    return theta, [pi, A]

def manage_clusters(data,gamma, distribution, 
                    model_theta, theta_new, theta_prev):
    
    """ This function will deal with the generated clusters, 
    both from the estimation and the parameters.  
    For every cluster it will check if the estimation degenerated, if it did then
    we use the handler function to set the new ones. If it is None, then they will be removed.
    
    Then we check that the pdf of the distribution can be computed, a.k.a the normalization
    constant can be computed. If it cannot be computed then we call the handler. If the result is None,
    then the cluster will be removed !! """
    
    pi,A = model_theta
    ## We avoid 0s in A or pi...
    pi = pi + 1e-200; pi = pi / np.sum(pi)
    A = A + 1e-200; A = A / np.sum(A, axis = 1)  # TODO Check
    model_theta = [pi,A]
    I = K = len(theta_new)

    clusters_change = 0  # Flag is we modify the clusters so that we dont stop
                        # due to a decrease in likelihood.
    
    if (type(gamma) != type(None)):
        r = []
        N = len(gamma)
        for i in range(I): # For every cluster
             # We compute the gamma normalization of the cluster
            # The contribution of every sample is weighted by gamma[i,n,t];
            # The total responsibility of the cluster for the samples is N_i_gamma
            rk = gamma[0][i,:]
            for n in range(1,N):
                rk = np.concatenate((rk, gamma[n][i,:]), axis = 0)
            rk = rk.reshape(rk.size,1)
            r.append(rk)
        r = np.concatenate(r, axis = 1)
    else:
        r = None
    theta_new, clusters_change = distribution.manage_clusters(data, r, theta_prev, theta_new)
                    
    ################## Last processing that you would like to do with everything ##############
    if  hasattr(distribution, 'use_chageOfClusters'):
        if (type(distribution.use_chageOfClusters) != type(None)):
            theta_new = distribution.use_chageOfClusters(theta_new, theta_prev)
    
    ############## Remove those clusters that are set to None ###################
    for k in range(K):
        k_inv = K - 1 -k
        if(type(theta_new[k_inv]) == type(None)):  # Degenerated cluster during estimation
            # Remove cluster from parameters
            theta_new,model_theta = remove_cluster(theta_new, [pi, A],k_inv)
            # Remove cluster from distribution data structure
            distribution.remove_cluster(k_inv)  
    
    return theta_new,model_theta, clusters_change

        
    return theta_new,model_theta, clusters_change

def match_clusters(mus_1, kappas_1, mus_2, kappas_2):
    # The index of the new clusters do not have to match with the order
    # of our previous cluster so we will assign to each new cluster the index
    # that is most similar to the ones we had 
    
    # Initially we will chose the one with lower distance between centroids
    
    pass

def get_initial_HMM_params_from_EM(EM_params):
    pimix = EM_params[0]
    theta = EM_params[1]
    
    I = pimix.size
    
    pi_init = pimix
    B_init = theta
    A_init = np.repeat(pi_init, I, axis = 0)
    
    return pi_init, B_init, A_init

def get_stationary_pi(pi, A, N = 20):
    for i in range(N):
        A = A.dot(A)
    spi = pi.dot(A)
        
    return spi