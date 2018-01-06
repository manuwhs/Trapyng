"""
This code will:
        - Load previously generated data in 3D for either EM and HMM.
        - Perform the EM
        - Plot the data and the evolution of the clusters.
        - Plot the evolution of LL with the iterations of EM
"""

# Change main directory to the main folder and import folders

#sys.path.append('../')
#base_path = os.path.abspath('')

# Official libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Own libraries
from graph_lib import gl

import Gaussian_distribution as Gad
import Gaussian_estimators as Gae
import Watson_distribution as Wad
import general_func as gf
import vonMisesFisher_distribution as vMFd
import vonMisesFisher_estimators as vMFe
import basicMathlib as bMA
import utilities_lib as ul
from scipy.stats import multivariate_normal
import HMM_libfunc as HMMlf

def print_final_clusters(myDManager, clusters_relation, theta_list,model_theta):
    ################## Print the Watson and Gaussian Distribution parameters ###################
    
    print ("$$$$$$$$$$$$$$$$$$$ Clusters Parameters $$$$$$$$$$$$$$$$$$$$$$")
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        if (distribution_name == "Gaussian" ):
            print ("------------ Gaussian Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[k][0])
            print ("Sigma")
            print (theta_list[k][1])
        if (distribution_name == "Gaussian2"):
            print ("------------ Diagonal Gaussian Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[k][0])
            print ("Sigma")
            print (theta_list[k][1])
        elif(distribution_name == "Watson"):
            print ("------------ Watson Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[k][0])
            print ("Kappa")
            print (theta_list[k][1])
        elif(distribution_name == "vonMisesFisher"):
            print ("------------ vonMisesFisher Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[k][0])
            print ("Kappa")
            print (theta_list[k][1])
    
    print ("$$$$$$$$$$$$$$$$$$$ Model Parameters $$$$$$$$$$$$$$$$$$$$$$")
    if(clusters_relation == "independent"):
        print "------------- pi -----------------"
        print (model_theta[0])
    elif(clusters_relation ==  "MarkovChain1"):
        print "------------- Initial pi -----------------"
        print (model_theta[0])
        print "------------- Transition matrix A -----------------"
        print (model_theta[1])
        print "------------- Stationary pi -----------------"
        print HMMlf.get_stationary_pi(model_theta[0], model_theta[1])

def plot_2D_clusters(myDManager, clusters_relation, theta, model_theta, ax):
    """
    This function plots the 2D clusters in the axes provided
    """
    if(clusters_relation ==  "MarkovChain1"):
        model_theta[0] = HMMlf.get_stationary_pi(model_theta[0], model_theta[1])
    
     # Only doable if the clusters dont die
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        
        if (distribution_name == "Gaussian" ):
            ## Plot the ecolution of the mu
            #### Plot the Covariance of the clusters !
            mean,w,h, angle_theta = bMA.get_gaussian_ellipse_params( mu = theta[k][0], Sigma = theta[k][1], Chi2val = 2.4477)
            r_ellipse = bMA.get_ellipse_points(mean,w,h,angle_theta)
            gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax, ls = "-.", lw = 3,
                    AxesStyle = "Normal2",
                   legend = ["Kg(%i). pi:%0.2f"%(k,  float(model_theta[0][0,k]))]) 
            
        if (distribution_name == "Gaussian2" ):
            ## Plot the ecolution of the mu
            #### Plot the Covariance of the clusters !
            mean,w,h,angle_theta = bMA.get_gaussian_ellipse_params( mu = theta[k][0], Sigma = theta[k][1], Chi2val = 2.4477)
            r_ellipse = bMA.get_ellipse_points(mean,w,h,angle_theta)
            gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax, ls = "-.", lw = 3,
                    AxesStyle = "Normal2",
                   legend = ["Kgd(%i). pi:%0.2f"%(k,  float(model_theta[0][0,k]))]) 
        
        elif(distribution_name == "Watson"):
            #### Plot the pdf of the distributino !
            ## Distribution parameters for Watson
            kappa = float(theta[k][1])
            mu = theta[k][0]

            Nsa = 1000
            # Draw 2D samples as transformation of the angle
            Xalpha = np.linspace(0, 2*np.pi, Nsa)
            Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
            
            probs = []  # Vector with probabilities
            for i in range(Nsa):
                probs.append(np.exp(Wad.Watson_pdf_log(Xgrid[:,i],[mu,kappa]) ))
            
            probs = np.array(probs)
            # Plot it in polar coordinates
            X1_w = (1 + probs) * np.cos(Xalpha)
            X2_w = (1 + probs) * np.sin(Xalpha)
            
            gl.plot(X1_w,X2_w, 
                 alpha = 1, lw = 3, ls = "-.",legend = ["Kw(%i). pi:%0.2f"%(k,  float(model_theta[0][0,k]))]) 
            
        elif(distribution_name == "vonMisesFisher"):
            #### Plot the pdf of the distributino !
            ## Distribution parameters for Watson
            kappa = float(theta[k][1]); 
            mu = theta[k][0]
            Nsa = 1000
            # Draw 2D samples as transformation of the angle
            Xalpha = np.linspace(0, 2*np.pi, Nsa)
            Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
            
            probs = []  # Vector with probabilities
            for i in range(Nsa):
                probs.append(np.exp(vMFd.vonMisesFisher_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                
            probs = np.array(probs)
            probs = probs.reshape((probs.size,1)).T
            # Plot it in polar coordinates
            X1_w = (1 + probs) * np.cos(Xalpha)
            X2_w = (1 + probs) * np.sin(Xalpha)
                
            gl.plot(X1_w,X2_w, 
                 alpha = 1, lw = 3, ls = "-.", legend = ["Kvmf(%i). pi:%0.2f"%(k,  float(model_theta[0][0,k]))]) 
            
def generate_gaussian_data(folder_images, plot_original_data, N1 = 200, N2 = 300, N3 = 50):

    
    mu1 = np.array([[0],[0]])
    cov1 = np.array([[0.8,-1.1],
                     [-1.1,1.6]])
    
    mu2 = np.array([[0],[0]])
    cov2 = np.array([[0.3,0.45],
                     [0.45,0.8]])
    mu3 = np.array([[0],[0]])
    cov3 = np.array([[0.1,0.0],
                     [0.0,0.1]])
    
    X1 = np.random.multivariate_normal(mu1.flatten(), cov1, N1).T
    X2 = np.random.multivariate_normal(mu2.flatten(), cov2, N2).T
    X3 = np.random.multivariate_normal(mu3.flatten(), cov3, N3).T

#    samples_X1 = np.array(range(X1.shape[1]))[np.where([X1[0,:] > 0])[0]]
#    samples_X1 = np.where(X1[0,:] > 0)[0] # np.array(range(X1.shape[1]))
#    print samples_X1
#    X1 = X1[:,samples_X1]
#    X2 = np.concatenate((X2,X3),axis = 1)
    
    ######## Plotting #####
    if (plot_original_data):
        gl.init_figure();
        ## First cluster
        ax1 = gl.scatter(X1[0,:],X1[1,:], labels = ["Gaussian Generated Data", "x1","x2"], 
                         legend = ["K = 1"], color = "r",alpha = 0.5)
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu1, Sigma = cov1, Chi2val = 2.4477)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2
                 ,AxesStyle = "Normal2", color = "r")
        
        ## Second cluster
        ax1 = gl.scatter(X2[0,:],X2[1,:], legend = ["K = 2"], color = "b", alpha = 0.5)
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu2, Sigma = cov2, Chi2val = 2.4477)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2,AxesStyle = "Normal2", color = "b")
        
        ## Third cluster
        ax1 = gl.scatter(X3[0,:],X3[1,:], legend = ["K = 3"], color = "g", alpha = 0.5)
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu3, Sigma = cov3, Chi2val = 2.4477)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2,AxesStyle = "Normal2", color = "g")
        
        ax1.axis('equal')

        gl.savefig(folder_images +'Original data.png', 
               dpi = 100, sizeInches = [12, 6])
    
    ############ ESTIMATE THEM ################
    theta1 = Gae.get_Gaussian_muSigma_ML(X1.T, parameters = dict([["Sigma","full"]]))
    print ("mu1:")
    print (theta1[0])
    print ("Sigma1")
    print(theta1[1])
    
    ############## Estimate Likelihood ###################
    ll = Gad.Gaussian_pdf_log (X1, [mu1,cov1])
    ll2 = []
    for i in range (ll.size):
        ll2.append( multivariate_normal.logpdf(X1[:,i], mean=mu1.flatten(), cov=cov1))
    ll2 = np.array(ll2).reshape(ll.shape)
    
    print ("ll ours")
    print (ll.T)
    print ("ll scipy")
    print (ll2.T)
    print ("Difference in ll")
    print ((ll - ll2).T)
    
    ###### Multiple clusters case
    ll_K = Gad.Gaussian_K_pdf_log(X1, [[mu1,cov1],[mu2,cov2]])
    
    if(0):
        X1 = gf.remove_module(X1.T).T
        X2 = gf.remove_module(X2.T).T
        X3 = gf.remove_module(X3.T).T
    Xdata = np.concatenate((X1,X2,X3), axis =1).T

    return X1,X2,X3,Xdata, mu1,mu2,mu3, cov1,cov2, cov3
#######################################################################################################################
#### Obtain the evolution of the centroids to plot them properly #####################################################
#######################################################################################################################

def plot_final_distribution(Xs,mus,covs, Ks ,myDManager, logl,theta_list,model_theta_list, folder_images):

    colors = ["r","b","g"]
    K_G,K_W,K_vMF = Ks
    ################## Print the Watson and Gaussian Distribution parameters ###################
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        if (distribution_name == "Gaussian"):
            print ("------------ Gaussian Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Sigma")
            print (theta_list[-1][k][1])
        elif(distribution_name == "Watson"):
            print ("------------ Watson Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Kappa")
            print (theta_list[-1][k][1])
        elif(distribution_name == "vonMisesFisher"):
            print ("------------ vonMisesFisher Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Kappa")
            print (theta_list[-1][k][1])
    print ("pimix")
    print (model_theta_list[-1])
    
    mus_Watson_Gaussian = []
    # k_c is the number of the cluster inside the Manager. k is the index in theta
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        mus_k = []
        for iter_i in range(len(theta_list)): # For each iteration of the algorihtm
            if (distribution_name == "Gaussian"):
                theta_i = theta_list[iter_i][k]
                mus_k.append(theta_i[0])
            elif(distribution_name == "Watson"):
                theta_i = theta_list[iter_i][k]
                mus_k.append(theta_i[0])
            elif(distribution_name == "vonMisesFisher"):
                theta_i = theta_list[iter_i][k]
                mus_k.append(theta_i[0])
                
        mus_k = np.concatenate(mus_k, axis = 1).T
        mus_Watson_Gaussian.append(mus_k)
    

    
    ######## Plot the original data #####
    gl.init_figure();
    ## First cluster
    for xi in range(len( Xs)):
        ## First cluster
        ax1 = gl.scatter(Xs[xi][0,:],Xs[xi][1,:], labels = ['EM Evolution. Kg:'+str(K_G)+ ', Kw:' + str(K_W) + ', K_vMF:' + str(K_vMF), "X1","X2"], 
                          color = colors[xi] ,alpha = 0.2, nf = 0)
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mus[xi], Sigma = covs[xi], Chi2val = 2.4477)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2
                 ,AxesStyle = "Normal2", color = colors[xi], alpha = 0.7)

    indx = -1
    # Only doable if the clusters dont die
    Nit,Ndim = mus_Watson_Gaussian[0].shape
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        
        if (distribution_name == "Gaussian"):
            ## Plot the ecolution of the mu
            #### Plot the Covariance of the clusters !
            mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = theta_list[indx][k][0], Sigma = theta_list[indx][k][1], Chi2val = 2.4477)
            r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
            gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "-.", lw = 3,
                    AxesStyle = "Normal2", 
                    legend = ["Kg(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 


            gl.scatter(mus_Watson_Gaussian[k][:,0], mus_Watson_Gaussian[k][:,1], nf = 0, na = 0, alpha = 0.3, lw = 1,
                          color = "y")
            gl.plot(mus_Watson_Gaussian[k][:,0], mus_Watson_Gaussian[k][:,1], nf = 0, na = 0, alpha = 0.8, lw = 2,
                          color = "y")
        
        elif(distribution_name == "Watson"):
            #### Plot the pdf of the distributino !
            ## Distribution parameters for Watson
            kappa = float(theta_list[indx][k][1])
            mu = theta_list[indx][k][0]

            Nsa = 1000
            # Draw 2D samples as transformation of the angle
            Xalpha = np.linspace(0, 2*np.pi, Nsa)
            Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
            probs = []  # Vector with probabilities
            for i in range(Nsa):
                probs.append(np.exp(Wad.Watson_pdf_log(Xgrid[:,i],[mu,kappa]) ))
            
            probs = np.array(probs)
            # Plot it in polar coordinates
            X1_w = (1 + probs) * np.cos(Xalpha)
            X2_w = (1 + probs) * np.sin(Xalpha)
            
            gl.plot(X1_w,X2_w, legend = ["Kw(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))] ,
               alpha = 1, lw = 3, ls = "-.")
            
        elif(distribution_name == "vonMisesFisher"):
            #### Plot the pdf of the distributino !
            ## Distribution parameters for Watson
            kappa = float(theta_list[indx][k][1])
            mu = theta_list[indx][k][0]

            Nsa = 1000
            # Draw 2D samples as transformation of the angle
            Xalpha = np.linspace(0, 2*np.pi, Nsa)
            Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
            
            probs = []  # Vector with probabilities
            for i in range(Nsa):
                probs.append(np.exp(vMFd.vonMisesFisher_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                
            probs = np.array(probs)
            probs = probs.reshape((probs.size,1)).T
            # Plot it in polar coordinates
            X1_w = (1 + probs) * np.cos(Xalpha)
            X2_w = (1 + probs) * np.sin(Xalpha)
            
#            print X1_w.shape, X2_w.shape
            gl.plot(X1_w,X2_w, 
                 alpha = 1, lw = 3, ls = "-.", legend = ["Kvmf(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
    ax1.axis('equal')
    gl.savefig(folder_images +'Final_State. K_G:'+str(K_G)+ ', K_W:' + str(K_W)  + ', K_vMF:' + str(K_vMF) + '.png', 
           dpi = 100, sizeInches = [12, 6])

def plot_multiple_iterations(Xs,mus,covs, Ks ,myDManager, logl,theta_list,model_theta_list, folder_images):
    ######## Plot the original data #####
    gl.init_figure();
    gl.set_subplots(2,3);
    Ngraph = 6
    
    colors = ["r","b","g"]
    K_G,K_W,K_vMF = Ks
    
    for i in range(Ngraph):
        indx = int(i*((len(theta_list)-1)/float(Ngraph-1)))
        nf = 1
        for xi in range(len( Xs)):
            ## First cluster
            labels = ['EM Evolution. Kg:'+str(K_G)+ ', Kw:' + str(K_W) + ', K_vMF:' + str(K_vMF), "X1","X2"]
            ax1 = gl.scatter(Xs[xi][0,:],Xs[xi][1,:],labels = ["","",""] , 
                              color = colors[xi] ,alpha = 0.2, nf = nf)
            nf =0
            mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mus[xi], Sigma = covs[xi], Chi2val = 2.4477)
            r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
            gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2
                     ,AxesStyle = "Normal2", color = colors[xi], alpha = 0.7)
            

        # Only doable if the clusters dont die
        for k_c in myDManager.clusterk_to_Dname.keys():
            k = myDManager.clusterk_to_thetak[k_c]
            distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
            
            if (distribution_name == "Gaussian"):
                ## Plot the ecolution of the mu
                #### Plot the Covariance of the clusters !
                mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = theta_list[indx][k][0], Sigma = theta_list[indx][k][1], Chi2val = 2.4477)
                r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
                gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "-.", lw = 3,
                        AxesStyle = "Normal2",
                       legend = ["Kg(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
            
            elif(distribution_name == "Watson"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1])
                mu = theta_list[indx][k][0]
    
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(Wad.Watson_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                
                probs = np.array(probs)
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.",legend = ["Kw(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
            elif(distribution_name == "vonMisesFisher"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1]); mu = theta_list[indx][k][0]
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(vMFd.vonMisesFisher_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                    
                probs = np.array(probs)
                probs = probs.reshape((probs.size,1)).T
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
    #            print X1_w.shape, X2_w.shape
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.", legend = ["Kvmf(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
            

        ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    gl.savefig(folder_images +'Final_State2. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + '.png', 
           dpi = 100, sizeInches = [18, 8])
    
def generate_images_iterations(Xs,mus,covs, Ks ,myDManager, logl,theta_list,model_theta_list,folder_images_gif):
#    os.remove(folder_images_gif) # Remove previous images if existing
    import shutil
    ul.create_folder_if_needed(folder_images_gif)
    shutil.rmtree(folder_images_gif)
    ul.create_folder_if_needed(folder_images_gif)
    ######## Plot the original data #####
    
    colors = ["r","b","g"]
    K_G,K_W,K_vMF = Ks
    
    for i in range(len(theta_list)):  # theta_list
        indx = i
        gl.init_figure()
        ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
        
        for xi in range(len( Xs)):
            ## First cluster
            ax1 = gl.scatter(Xs[xi][0,:],Xs[xi][1,:], labels = ['EM Evolution. Kg:'+str(K_G)+ ', Kw:' + str(K_W) + ', K_vMF:' + str(K_vMF), "X1","X2"], 
                              color = colors[xi] ,alpha = 0.2, nf = 0)
            
            mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mus[xi], Sigma = covs[xi], Chi2val = 2.4477)
            r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
            gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2
                     ,AxesStyle = "Normal2", color = colors[xi], alpha = 0.7)
            

        # Only doable if the clusters dont die
        for k_c in myDManager.clusterk_to_Dname.keys():
            k = myDManager.clusterk_to_thetak[k_c]
            distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
            
            if (distribution_name == "Gaussian"):
                ## Plot the ecolution of the mu
                #### Plot the Covariance of the clusters !
                mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = theta_list[indx][k][0], Sigma = theta_list[indx][k][1], Chi2val = 2.4477)
                r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
                gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "-.", lw = 3,
                        AxesStyle = "Normal2",
                       legend = ["Kg(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
            
            elif(distribution_name == "Watson"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1])
                mu = theta_list[indx][k][0]
    
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(Wad.Watson_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                
                probs = np.array(probs)
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.",legend = ["Kw(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
            elif(distribution_name == "vonMisesFisher"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1]); mu = theta_list[indx][k][0]
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(vMFd.vonMisesFisher_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                    
                probs = np.array(probs)
                probs = probs.reshape((probs.size,1)).T
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
    #            print X1_w.shape, X2_w.shape
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.", legend = ["Kvmf(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
                
        gl.set_zoom(xlim = [-6,6], ylim = [-6,6], ax = ax1)     
        ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
        if (indx == 0):
            gl.add_text(positionXY = [0.1,.5], text = r' Initilization Incomplete LogLike: %.2f'%(logl[0]),fontsize = 15)
            pass
        elif (indx >= 1):
           
            gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:(indx+1)], ax = ax2, 
                    legend = ["Iteration %i, Incom LL: %.2f"%(indx, logl[indx])], labels = ["Convergence of LL with generated data","Iterations","LL"], lw = 2)
            gl.scatter(1, logl[1], lw = 2)
            pt = 0.05
            gl.set_zoom(xlim = [0,len(logl)], ylim = [logl[1] - (logl[-1]-logl[1])*pt,logl[-1] + (logl[-1]-logl[1])*pt], ax = ax2)
            
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
        
        gl.savefig(folder_images_gif +'gif_'+ str(indx) + '.png', 
               dpi = 100, sizeInches = [16, 8], close = "yes",bbox_inches = None)
        
        gl.close("all")
        

def generate_images_iterations_ll(Xs,mus,covs, Ks ,myDManager, logl,theta_list,model_theta_list,folder_images_gif):
#    os.remove(folder_images_gif) # Remove previous images if existing
    """
    WARNING: MEANT FOR ONLY 3 Distributions due to the color RGB
    """
    import shutil
    ul.create_folder_if_needed(folder_images_gif)
    shutil.rmtree(folder_images_gif)
    ul.create_folder_if_needed(folder_images_gif)
    ######## Plot the original data #####

    Xdata = np.concatenate(Xs,axis = 1).T
    colors = ["r","b","g"]
    K_G,K_W,K_vMF = Ks
    
    ### FOR EACH ITERATION 
    for i in range(len(theta_list)):  # theta_list
        indx = i
        gl.init_figure()
        ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
        
        ## Get the relative ll of the Gaussian denoising cluster.
        ll = myDManager.pdf_log_K(Xdata,theta_list[indx])
        N,K = ll.shape
#        print ll.shape
        for j in range(N):  # For every sample
        #TODO: Can this not be done without a for ?
            # Normalize the probability of the sample being generated by the clusters
            Marginal_xi_probability = gf.sum_logs(ll[j,:])
            ll[j,:] = ll[j,:]- Marginal_xi_probability
        
            ax1 = gl.scatter(Xdata[j,0],Xdata[j,1], labels = ['EM Evolution. Kg:'+str(K_G)+ ', Kw:' + str(K_W) + ', K_vMF:' + str(K_vMF), "X1","X2"], 
                              color = (np.exp(ll[j,1]), np.exp(ll[j,0]), np.exp(ll[j,2])) ,  ###  np.exp(ll[j,2])
                              alpha = 1, nf = 0)
            
        # Only doable if the clusters dont die
        for k_c in myDManager.clusterk_to_Dname.keys():
            k = myDManager.clusterk_to_thetak[k_c]
            distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
            
            if (distribution_name == "Gaussian"):
                ## Plot the ecolution of the mu
                #### Plot the Covariance of the clusters !
                mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = theta_list[indx][k][0], Sigma = theta_list[indx][k][1], Chi2val = 2.4477)
                r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
                gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "-.", lw = 3,
                        AxesStyle = "Normal2",
                       legend = ["Kg(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
            
            elif(distribution_name == "Watson"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1]);  mu = theta_list[-1][k][0]
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(Wad.Watson_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                
                probs = np.array(probs)
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.", legend = ["Kw(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
            elif(distribution_name == "vonMisesFisher"):
                #### Plot the pdf of the distributino !
                ## Distribution parameters for Watson
                kappa = float(theta_list[indx][k][1]); mu = theta_list[indx][k][0]
                Nsa = 1000
                # Draw 2D samples as transformation of the angle
                Xalpha = np.linspace(0, 2*np.pi, Nsa)
                Xgrid= np.array([np.cos(Xalpha), np.sin(Xalpha)])
                
                probs = []  # Vector with probabilities
                for i in range(Nsa):
                    probs.append(np.exp(vMFd.vonMisesFisher_pdf_log(Xgrid[:,i],[mu,kappa]) ))
                    
                probs = np.array(probs)
                probs = probs.reshape((probs.size,1)).T
                # Plot it in polar coordinates
                X1_w = (1 + probs) * np.cos(Xalpha)
                X2_w = (1 + probs) * np.sin(Xalpha)
                
    #            print X1_w.shape, X2_w.shape
                gl.plot(X1_w,X2_w, 
                     alpha = 1, lw = 3, ls = "-.", legend = ["Kvmf(%i). pi:%0.2f"%(k,  float(model_theta_list[indx][0][0,k]))]) 
                
            
        gl.set_zoom(xlim = [-6,6], ylim = [-6,6], ax = ax1)     
        ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
        if (indx == 0):
            gl.add_text(positionXY = [0.1,.5], text = r' Initilization Incomplete LogLike: %.2f'%(logl[0]),fontsize = 15)
            pass
        elif (indx >= 1):
           
            gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:(indx+1)], ax = ax2, 
                    legend = ["Iteration %i, Incom LL: %.2f"%(indx, logl[indx])], labels = ["Convergence of LL with generated data","Iterations","LL"], lw = 2)
            gl.scatter(1, logl[1], lw = 2)
            pt = 0.05
            gl.set_zoom(xlim = [0,len(logl)], ylim = [logl[1] - (logl[-1]-logl[1])*pt,logl[-1] + (logl[-1]-logl[1])*pt], ax = ax2)
            
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
        
        gl.savefig(folder_images_gif +'gif_'+ str(indx) + '.png', 
               dpi = 100, sizeInches = [16, 8], close = "yes",bbox_inches = None)
        
        gl.close("all")