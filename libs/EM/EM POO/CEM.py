# Change main directory to the main folder and import folders
import EM_lib as EMl
import EM_libfunc as EMlf
import HMM_libfunc as HMMlf

class CEM ():
    
    def __init__(self, distribution = None, clusters_relation = "independent", 
                 T = 30, Ninit = 20, delta_ll = 0.01, 
                 verbose = 0, time_profiling = "no"):
        
        self.distribution = distribution
        self.clusters_relation = clusters_relation
        self.delta_ll = delta_ll
        self.Ninit = Ninit
        self.T = T
        self.verbose = verbose;
        self.time_profiling = time_profiling

    """  
    Inputs:
        - distribution: Distribution manager with the different distributions of the 
        mixture and their associated clusters.
        - clusters_relation: Dependency between the clusters:
              - For "independent": The clusters are assumed independent
              - For "MarkovChain1": The clusters are assumed to follow discrete 
                 MC of order 1
        - delta_ll: Minimum increase in likelihood for consideing that the algorithm 
        has not converged. If in an iteration the incomplete loglikelihood does not 
        increase by this minimum value, the iterations stop.
        - T: Maximum number of iterations.
        - model_theta_init: Initial parameters of the model. If not given they will 
        be initilized uniformly. This is [pi] or [pi, A] for example.
        - theta_init: Initial parameters of the clusters. If not provided, the 
        initializer provided in the
          distribution objects will initialize them.
        - Ninit: Number of random random re-initilizations.
        - verbose: The higher the value, the more partial outputs will be printed out.
            - 0: No verbose
            - 1: Indication of each initilization of the EM and its final ll.
            - 2: Indication of each iteration of the EM and its final ll.
        - time_profiling: If we want to see the times it takes for operations.
        
    Output:
        - logl: List of incomplete loglikelihoods asociated to each iteration.
          Notice it will have always the added initial loglikelihood asociated 
          to the initialization. So maximum it will be "T+1" elements.
        - theta_list: List of the cluster parameters for each of iterations.
        - model_theta_init: List of the model parameters for each of the iterations
    """

    def fit(self, data, model_theta_init = None, theta_init = None):
        """
        This method runs the EM algorithm on the provided data. 
        
        """
        output = EMl.run_several_EM(data = data, distribution = self.distribution, clusters_relation = self.clusters_relation,
                                    T = self.T, Ninit = self.Ninit, delta_ll = self.delta_ll,
                                    model_theta_init = model_theta_init, theta_init = theta_init,
                                    verbose = self.verbose, time_profiling = self.time_profiling )
        
        return output
    ### Funcitons and shit 

    """ The class also have a set of static methods to use.
    They are just implemented the ones in EMl"""
    
#    @staticmethod
    def get_loglikelihood(self, X, distribution,theta, model_theta_init):
    # Combined funciton to obtain the loglikelihood and r in one step
    # The shape of X is (N,D)
        
        if self.clusters_relation == "independent":
            incomloglike = EMlf.get_loglikelihood(X,distribution,theta, model_theta_init)
        else:
            incomloglike = HMMlf.get_loglikelihood(X,distribution,theta, model_theta_init)
        return incomloglike
    
    def get_responsibilities(self, X, distribution,theta, model_theta_init):
    # Combined funciton to obtain the loglikelihood and r in one step
    # The shape of X is (N,D)
        
        if self.clusters_relation == "independent":
            incomloglike = EMlf.get_responsibilities(X,distribution,theta, model_theta_init)
        else:
            incomloglike = HMMlf.get_responsibilities(X,distribution,theta, model_theta_init)
        return incomloglike
    
    def get_alpha_responsibilities(self, X, distribution,theta, model_theta_init):
    # Combined funciton to obtain the loglikelihood and r in one step
    # The shape of X is (N,D)
        
        if self.clusters_relation == "independent":
            incomloglike = EMlf.get_responsibilities(X,distribution,theta, model_theta_init)
        else:  
            incomloglike = HMMlf.get_alpha_responsibilities(X,distribution,theta, model_theta_init)
        return incomloglike
    
    
#    def get_samples_loglikelihood(self, X, distribution,theta, model_theta_init):
#        # Get the loglikelihood of each of the samples
#        
#        if self.clusters_relation == "independent":
#            incomloglike = EMlf.get_loglikelihood(X,distribution,theta, model_theta_init)
#        else:
#            incomloglike = HMMlf.get_loglikelihood(X,distribution,theta, model_theta_init)
#        return incomloglike