
import numpy as np
import HMM_libfunc2 as HMMlf

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

import copy

def st_FB_MAP_dec(t,gamma,T):
    # This function outputs the MAP state at time t, st, given all the
    # observations as   st = arg max (gamma(t))
    # gamma(i,t)  Tells us the gamma at time t for the state i.
    
#    print gamma.shape
#    for i in range(T):     # For every observation of the sequence
#    print gamma.shape
    idx = np.argmax(gamma[:,t]);
    st =  idx;
    return st

def SbS_MAP_dec(Y,gamma,T):
    # This function outputs the sequence S = {s1,..., sT} of the Step by Step MAP 
    # probability of Y = {y1,..., yT}.
    # For every observation yt, we have st = arg max (gamma(t))
    # gamma(i,t)  Tells us the gamma at time t for the state i.
    S = np.zeros((1,T));
#    print S.shape
#    print T, gamma.shape
    for t in range(T):         # For every observation of the sequence
         S[0,t] =  st_FB_MAP_dec(t,gamma,T);
    
    return S

def SbS_decoder(data, A, B, pi):
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%% Step By Step MAP decoder %%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Data values

    N = len(data)       # Number of Realizations of the HMM

    #  HMM parameters
    I = A.shape[0]
    alpha = HMMlf.get_alfa_matrix_log(A,B,pi,data);
    beta = HMMlf.get_beta_matrix_log(A,B,data);
    gamma = HMMlf.get_gamma_matrix_log(alpha,beta );


    S_MAP_SbS = []

#    print S_MAP_SbS.shape
#    print gamma.shape
    
    for n in range(N):   # For every sequence Y = {y1,..., yT}.
        Y_n = data[n];
        T = data[n].shape[0]
        S_MAP_SbS.append (SbS_MAP_dec(Y_n,gamma[n][:,:],T).flatten());

    return S_MAP_SbS


def MLViter_decoder(data, A, B, pi):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%% ML Viterbi decoder %%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TODO: fix
    # If the journey is too long... then all the samples have state 0...
     
    N = len(data)       # Number of Realizations of the HMM

    #  HMM parameters
    I = A.shape[0]
    
    # TODO: Maybe we do not need them ? Maybe we should use the gammas instead ??
#    alpha = HMMlf.get_alfa_matrix_log(I,N,T, A,B,pi,data);
#    beta = HMMlf.get_beta_matrix_log(I,N,T, A,B,data);
#    gamma = HMMlf.get_gamma_matrix_log( I,N,T, alpha,beta );


    # We compute it by means of the log-likelihood of Y given S
    S_ML_Vit = [];
    

    # Journeys(t,i) contains the survival path at time index t, that ends at
    # state number i
    # Costs(t,i) contains the cost of the survival path al time index t, 
    # that ends atstate number i
    # Possible_journeys_c(j) contains all the I possible journeys that go from
    # Journey(t-1,:) to state(t) = i
    # First step for t = 1

    for n in range(N):
        T = data[n].shape[0]
        Past_Journeys = np.zeros((T,I));   # Over dimensioned to abarcar all cases
        Journeys = np.zeros((T,I)); 
        Costs = np.zeros((T,I));  # We compute the costs in logarithmic form
        Possible_journeys_c = np.zeros((1,I));
        
        for i in range(I):
            Journeys[0,i] = i;
            Costs[0,i] = np.log(Wad.Watson_pdf(data[n][0,:], B[0][:,i], B[1][:,i]))
#            Costs[0,i] = gamma[i,n,0]
        Past_Journeys = copy.deepcopy(Journeys)
        for t in range(1,T):          # For every time index
            for i in range(I):                # For every state(t) = i we want to get the survival journey for
                 for j in range(I):        # For every Journey(t-1,j) we could come from
                    Possible_journeys_c[0,j] = Costs[t-1,j] + \
                    np.log(Wad.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i])) 
                    
#                    gamma[i,n,t]
                 Prob  = np.max (Possible_journeys_c[0,:])
                 best_j = np.argmax (Possible_journeys_c[0,:]);
#                 print Possible_journeys_c
                 # Generate the new journey that ends in state i at time t as the
                 # joining of the best previous survival journey and the state i
                 # TODO: Maybe the index of the following is wrong !!
                 Journeys[0:t,i] = copy.deepcopy(Past_Journeys[0:t,best_j]);
                 Journeys[t,i] = i;
                 Costs[t,i] = Costs[t-1,best_j] + Prob;
         
            Past_Journeys = copy.deepcopy(Journeys);
   
        Prob = np.max (Costs[T-1,:])
        best_S =  np.argmax (Costs[T-1,:]);
        S_ML_Vit.append(copy.deepcopy(Journeys[:,best_S]).flatten());    

    return S_ML_Vit


def MAPViter_decoder(data, A, B, pi):

#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    %%%%%%%%%%%%%%% MAP Viterbi decoder %%%%%%%%%%%%%
#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N = len(data)       # Number of Realizations of the HMM

    #  HMM parameters
    I = A.shape[0]
    
    S_MAP_Vit = []
    aux_sigma_j = np.zeros((1,I)); # Aux probabilities to get the best jouerney
    for n in range(N):
        T = data[n].shape[0]

        sigma = np.zeros((T,I));             # Sigma matriz x of values
        Survival_paths = np.zeros((T,I));    # Sigma matriz x of values
        Past_Survival_paths = np.zeros((T,I)); # Sigma matriz x of values
    
        # Calculate sigma 1
        for i in range(I):
            sigma[0,i] = np.log(Wad.Watson_pdf(data[n][0,:], B[0][:,i], B[1][:,i])) + np.log(pi[0,i]);
            Survival_paths[0,i] = i;

        Past_Survival_paths =  copy.deepcopy(Survival_paths);
        # Calculate sigma t recursively with dynamic programming
        for t in range(1,T):          # For every time index
            for i in range(I):                # For every state(t) = i we want to get the survival journey for
                 for j in range(I):        # For every Journey(t-1,j) we could come from
                     aux_sigma_j[0,j] = np.log(A[j,i]) + sigma[t-1,j];
                
                 sigma_tj = np.max (aux_sigma_j[:])
                 best_j = np.argmax (aux_sigma_j[:]);
                 # sti is the estimated state of the survival path
                 sigma[t,i] =  np.log(Wad.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i])) + sigma_tj;
                 Survival_paths[0:t,i] = copy.deepcopy(Past_Survival_paths[0:t,best_j]);
                 Survival_paths[t,i] = i;
            Past_Survival_paths = copy.deepcopy(Survival_paths);

        
        # The best Survival path is the one that maxizes the sigma(T,i)
        # It is the one that ends with the i, that is: Survival_paths(:,best_S)
        best_sigma = np.max (sigma[T-1,:]);
        best_S = np.argmax (sigma[T-1,:]);
        
        S_MAP_Vit.append(copy.deepcopy(Survival_paths[:,best_S]).flatten());

    return S_MAP_Vit
    