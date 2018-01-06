
# Official libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
# Own libraries
import import_folders
from graph_lib import gl


import sampler_lib as sl
import EM_lib as EMl
import EM_libfunc as EMlf
import HMM_lib as HMMl
import HMM_libfunc2 as HMMlf
import decoder_lib as decl
import pickle_lib as pkl
import scipy.io
from sklearn import preprocessing

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf
import data_preprocessing as dp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import cross_validation
plt.close("all")

################################################################
######## System parts to make the main code more legilible ! ##############################
###############################################################


def create_data(X_All_labels, label_classes, 
                channel_sel = None, max_trials = -1,
                rem_timePoint_ave = True, rem_features_ave = False, normalize = True ):

    Nclasses = len(X_All_labels)
    Ntrials, Ntime, Ndim = X_All_labels[0].shape
    
    if (type(channel_sel) == None):
        channel_sel = range (Ndim)
#    channel_sel = [0, 11, 21, 28, 36, 50, 60]

    if (max_trials == -1):
        max_trials = 500 # Number of trials per class

    # Now we remove from each 70-dimensional time point the average.
    if (rem_timePoint_ave == True):
        X_All_labels = dp.remove_timePoints_average(X_All_labels)
    
    ## Remove the average of the 70 channels to each sample
    # We obtain the average of th 70 channels like in a normal ML problem
    # No matter the class and then remove them from each la
    if (rem_features_ave == True):
        X_All_labels = dp.remove_features_average(X_All_labels)

    # Then select the desired channels and then make modulus = 1
    X_data_trials, X_data_labels = dp.preprocess_data_set (
                                    X_All_labels, label_classes, 
                                    max_trials = max_trials, channel_sel= channel_sel,
                                    normalize = normalize)

    return X_data_trials, X_data_labels


def get_clusters_labels_EM(Nclasses, X_train, y_train, 
                           Ninit =5, K  =  5, T  = 100, verbose = 1):

    Ks_params = []
    for i in range(Nclasses):
        X_train_class_i = [ X_train[j] for j in np.where(np.array(y_train) == i)[0]]
        logl,theta_list,pimix_list = EMl.run_several_EM(X_train_class_i, K = K, delta = 0.1, T = T,
                                    Ninit = Ninit, verbose = verbose)
        Ks_params.append([pimix_list[-1],theta_list[-1]])
    
    return Ks_params

def get_clusters_labels_HMM(Nclasses, X_train, y_train, Ks_params = None,
                            Ninit =5, I  =  5, R  = 100, verbose = 1):
    
    # Do we initialize it with clusters
    if (type(Ks_params) != type(None)):
        init_with_EM = 1
    else:
        init_with_EM = 0
    
    Is_params = []
    
    for k in range(Nclasses): 
        pi_init, B_init, A_init = None, None, None
        
        if (init_with_EM):
            pi_init, B_init, A_init = HMMlf.get_initial_HMM_params_from_EM(Ks_params[k])
            I = pi_init.size
            
        X_train_class_k = [ X_train[j] for j in np.where(np.array(y_train) == k)[0]]
        
#        print k, "pene"
        logl,B_list,pi_list, A_list = \
            HMMl.run_several_HMM(data = X_train_class_k,I = I,delta = 0.01, R = R
                     ,pi_init = pi_init, A_init = A_init, B_init = B_init, Ninit = Ninit, verbose = verbose)
    
        Is_params.append(copy.deepcopy([pi_list[-1],A_list[-1], B_list[-1]]))
    
    return Is_params
    
def pca_decomp(X_train, X_test, n_components = 1):
    # Instead of working with the original data we. We first do some PCA
    ## TODO: Idea... project different PCA for classes TO
    Xtrain_all =  np.concatenate(X_train, axis = 0)     # X_train[0]
    
    pca = PCA(n_components=n_components)
    pca.fit(Xtrain_all)
    
    X_train = [pca.transform(X_train[i]) for i in range(len(X_train))]
    X_test = [pca.transform(X_test[i]) for i in range(len(X_test))]

    return X_train, X_test
    

def get_2EM_vectors(X_All_labels, label_classes, 
                    max_trials = 50, channel_sel= None,
                    plot_flag = 0):

    Nclasses = len(X_All_labels)
    # Instead of obtaining the average profile at each sime instant, we run
#     a 2-EM clustering to each label, at each time instance.
#     For each time instance,instead of getting the previous mean, we get 2 vectors.
#     The mean vector of the class with high kappa and the one with low kappa
#     We work on the assumtion that we have 2 clusters,
    
    X_data_ave_EM_plus, X_data_ave_EM_minus = dp.get_labels_ave_EM(
                                    X_All_labels, label_classes, 
                                    max_trials = max_trials, channel_sel= channel_sel)

    # Get the 2-EM for all the trials !! 
    # For every trial we just run a 2 EM and get the direction of the
    
    X_trials_EM_plus, X_trials_EM_minus = dp.get_X_trials_EM(
                                    X_All_labels, label_classes, 
                                    max_trials = max_trials, channel_sel= channel_sel)


    plot_flag = 0
    if (plot_flag):
        
        ## For the 2-EM across time
        gl.scatter_3D(0, 0,0, nf = 1, na = 0)
        for i in range(Nclasses):
            gl.scatter_3D(X_data_ave_EM_plus[i][:,0], X_data_ave_EM_plus[i][:,1],X_data_ave_EM_plus[i][:,2], nf = 0, na = 0)
        gl.scatter_3D(0, 0,0, nf = 1, na = 0)
        for i in range(Nclasses):
            gl.scatter_3D(X_data_ave_EM_minus[i][:,0], X_data_ave_EM_minus[i][:,1],X_data_ave_EM_minus[i][:,2], nf = 0, na = 0)
    
        ## For the 2-EM across trials
        gl.scatter_3D(0, 0,0, nf = 1, na = 0)
        for i in range(Nclasses):
            gl.scatter_3D(X_trials_EM_plus[i][:,0], X_trials_EM_plus[i][:,1],X_trials_EM_plus[i][:,2], nf = 0, na = 0)
        gl.scatter_3D(0, 0,0, nf = 1, na = 0)
        for i in range(Nclasses):
            gl.scatter_3D(X_trials_EM_minus[i][:,0], X_trials_EM_minus[i][:,1],X_trials_EM_minus[i][:,2], nf = 0, na = 0)

def plot_data_trials(Nclasses, X_train, y_train,
                     n_trials_to_show = 2 ,   colors = ["r","k"]):

#    Nclasses = len(X_data_labels)
#    print Nclasses
    ## Plotting evolution in time of several trials for both classes
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    
    for i in range(Nclasses):
        X_train_class_i = [ X_train[j] for j in np.where(np.array(y_train) == i)[0]]
        for ntr in range (n_trials_to_show):      
            X_train_class_i_n = [X_train_class_i[ntr]]
            X_train_class_i_n = np.concatenate(X_train_class_i_n, axis = 0)
            gl.scatter_3D(X_train_class_i_n[:,0], X_train_class_i_n[:,1],X_train_class_i_n[:,2], color = colors[i],
                          nf = 0, na = 0, labels = ["Time Evolution of trials different classes", "D1","D2","D3"])

def plot_trials_for_same_instance(X_data_trials, X_data_labels, X_train, y_train,
                                  colors = ["r","k"],  time_show = 100, normalize = True):

    # For a fixed time instant, the plotting of points from several trials of both classes
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    for i in range(len(X_data_trials)):
        class_i = X_data_labels[i]
        if (normalize == True):
            caca = gf.normalize_module(X_data_trials[i][[time_show],:])
        else:
            caca = X_data_trials[i][[time_show],:]
        
        caca = caca.flatten()
        gl.scatter_3D(caca[0],caca[1],caca[2], 
                          nf = 0, na = 0, color = colors[class_i],labels = ["Trials for the same time instant for different classes", "D1","D2","D3"])

def plot_single_trials(Nclasses, X_train, y_train,
              n_trials_to_show = 2 ,   colors = ["r","k"]):

    gl.plot([0],[0])
    for i in range(Nclasses):
        X_train_class_i = [ X_train[j] for j in np.where(np.array(y_train) == i)[0]]
        max_val = 0
        if (i >= 1):
            max_val += np.max(np.abs(X_train_class_i[i-1])) + np.max(np.abs(X_train_class_i[i]))
        for ntr in range (n_trials_to_show):  
            gl.plot([], X_train_class_i[ntr] + max_val, color = colors[i], nf = 0,
                    labels = ["Some trials evolution","time","signal"])

def plot_means(Nclasses, X_train, y_train,
               colors = ["r","k"], normalize = True):
    print Nclasses
    ## Get the time average profile of every label.
    # For every label, we average across time to get the time profile.
    # We kind of should assume that the trials are somewhat time-aligned
#    X_data_ave = dp.get_timeSeries_average_by_label(X_All_labels, channel_sel = channel_sel)
    X_data_ave = dp.get_average_from_train(Nclasses, X_train, y_train, normalize = normalize)
    # Evolution of the means of each class in time in time representation

    gl.plot([0],[0])
    for i in range(1):
        max_val = 0
        if (i >= 1):
            max_val += np.max(np.abs(X_data_ave[i-1])) + np.max(np.abs(X_data_ave[i]))
            
        gl.plot([], X_data_ave[i] + max_val, color = colors[i], nf = 0,
                labels = ["Mean value of the 70 Channels","Time Index","Channels"])


    # Evolution of the means of each class in time in Spherical representation
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    for i in range(Nclasses):
        gl.scatter_3D(X_data_ave[i][:,0], X_data_ave[i][:,1],X_data_ave[i][:,2], color = colors[i],
                      nf = 0, na = 0, labels = ["Mean Time Evolution the different classes", "D1","D2","D3"])


def plot_PCA_example(X_train, y_train):

    Xtrain =  np.concatenate(X_train, axis = 0)     # X_train[0]
#    np.concatenate(X_train, axis = 0)
#    Xtest = X_test[0]
    nSamples, nFeatures = Xtrain.shape
    pca = PCA()
    pca.fit_transform(Xtrain)
    
    cumvarPCA = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot classification score vs number of components
#    nComponents = np.arange(1,nFeatures,8)
#    pcaScores = np.zeros((5,np.alen(nComponents)))
#    
#    for i,n in enumerate(nComponents):   
#        pca = PCA(n_components=n,whiten=False)
#        XtrainT = pca.fit_transform(Xtrain)
#        XtestT = pca.transform(Xtest)
#        pcaScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    
    #%% Plot figures
    #%% Plot explained variance
    plt.figure()
    plt.plot(np.arange(1,np.alen(cumvarPCA)+1),cumvarPCA,lw=3,label='PCA')
    plt.xlim(1,np.alen(cumvarPCA))
    plt.legend(loc='lower right')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()
    
 #%% Plot data proyections for PCA
    alpha_val = 1
    classColors =  ["r","k"] # np.random.rand(Nclasses,3)
    pca = PCA(n_components=2)
    xtPCA = pca.fit_transform(Xtrain)
    uPCA = pca.components_
    labelsTrain = np.array(y_train)
    # View the  original data and the projection vectors
    labels = np.unique(labelsTrain.astype(np.int))
    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=alpha_val,c=classColors[i])
    plt.title('First 2 components of projected data')
    plt.show()

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
    

def classify_with_Likelihood_EM(X_train, X_test, y_train, y_test, Ks_params):
    # Try to classify using the likelihood of every set of clusters !
    Likelihoods_tr = dp.get_likelihoods_EM(X_train, Ks_params)
#    print [y_train, np.argmax(Likelihoods, axis = 1)]
    acc_tr = gf.accuracy(y_train, np.argmax(Likelihoods_tr, axis = 1))
    print "Train Accuracy %f" %(acc_tr)

    Likelihoods_tst = dp.get_likelihoods_EM(X_test, Ks_params)
#    print [y_test, np.argmax(Likelihoods, axis = 1)]
    acc_tst = gf.accuracy(y_test, np.argmax(Likelihoods_tst, axis = 1))
    print "Test Accuracy %f" %(acc_tst)

    return Likelihoods_tr, Likelihoods_tst

def classify_with_Likelihood_HMM(X_train, X_test, y_train, y_test, Is_params):

    Likelihoods_tr = dp.get_likelihoods_HMM(X_train, Is_params)
#    print [y_train, np.argmax(Likelihoods, axis = 1)]
    acc_tr = gf.accuracy(y_train, np.argmax(Likelihoods_tr, axis = 1))
    print "Train Accuracy %f" %(acc_tr)
    
    Likelihoods_tst = dp.get_likelihoods_HMM(X_test, Is_params)
#    print [y_test, np.argmax(Likelihoods, axis = 1)]
    acc_tst = gf.accuracy(y_test, np.argmax(Likelihoods_tst, axis = 1))
    print "Test Accuracy %f" %(acc_tst)
    return Likelihoods_tr, Likelihoods_tst
    
def get_normalized_ll_byCluster_EM(X_train, X_test, y_train, y_test, Ks_params):
    Likelihoods_by_Cluster_train = dp.get_likelihoods_byClusters_EM(X_train, Ks_params)
    Likelihoods_by_Cluster_test = dp.get_likelihoods_byClusters_EM(X_test, Ks_params)
    
    
#    print Likelihoods_by_Cluster_train
    pos_list = np.where(np.array(y_train) == 0)[0]
    neg_list = np.where(np.array(y_train) == 1)[0]
    gl.scatter(Likelihoods_by_Cluster_train[pos_list,0], Likelihoods_by_Cluster_train[pos_list,2])
    gl.scatter(Likelihoods_by_Cluster_train[neg_list,0], Likelihoods_by_Cluster_train[neg_list,2], nf= 0)
    
    #%% Normalize data
    Xtrain = Likelihoods_by_Cluster_train
    Xtest = Likelihoods_by_Cluster_test
    Ytrain = y_train
    Ytest = y_test
    
    Ntrain,Ndim = Xtrain.shape
    Ntest, Ndim = Xtest.shape
    mx = np.mean(Xtrain,axis=0,dtype=np.float64)
    stdx = np.std(Xtrain,axis=0,dtype=np.float64)
    
#    print mx
#    print stdx
    
    Xtrain = np.divide(Xtrain-np.tile(mx,[Ntrain,1]),np.tile(stdx,[Ntrain,1]))
    Xtest = np.divide(Xtest-np.tile(mx,[Ntest,1]),np.tile(stdx,[Ntest,1]))
    
#    print Xtrain
    return Xtrain, Xtest, Ytrain, Ytest
    

def perfrom_CV_EM_classes(Nclasses, X_data_trials, X_data_labels, nfolds = 2,
                          Klusters = [4,5,6,7,8,9], Ninit = 50, T = 100):
    # We compute the EM for both clusters for the classes.
    # We do it for different kluster sizes K. We train with the means of train and we validate with the means of test.
    

    final_ll_train = 0
    final_ll_test = 0
    # Example on how to do crossvalidation
    stkfold = cross_validation.StratifiedKFold(X_data_labels, n_folds = nfolds)
    for train_index, val_index in stkfold:
        X_train_notNormalized = [X_data_trials[itr] for itr in train_index]
        y_train = [X_data_labels[itr] for itr in train_index]
        
        X_test_notNormalized = [X_data_trials[iv] for iv in val_index]
        y_test = [X_data_labels[iv] for iv in val_index]

        X_data_ave_train = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                               normalize = True, partitions = nfolds -1)
        X_data_ave_test = dp.get_average_from_train(Nclasses, X_test_notNormalized, y_test, 
                                               normalize = True, partitions = 1)

        print "-----------------"
        print X_data_ave_train[0].shape, len(X_data_ave_train)
        print X_data_ave_test[0].shape, len(X_data_ave_test)
        
        All_Ks_params = [];
        
        ll_train = []; # This has the likelihood for both class
        ll_test = []; # This has the likelihood for both classes
        
        for K in Klusters:
            Ninit = Ninit; K  =  K; verbose = 0; T  = T
            Ks_params = get_clusters_labels_EM(Nclasses, X_train = X_data_ave_train, y_train = range(Nclasses), 
                                     Ninit = Ninit, K  =  K, T  = T, 
                                     verbose = verbose)
            print "Finished EM"
            All_Ks_params.append(Ks_params)
        
            ## Get the likelihoods for the train and test sets
            ll_Allclasses = []
            for ic in range(Nclasses):
                new_ll = EMlf.get_EM_Incomloglike_log(Ks_params[ic][1],Ks_params[ic][0],X = X_data_ave_train[ic])
                ll_Allclasses.append(new_ll)
            ll_train.append(copy.deepcopy(ll_Allclasses))
            
            ll_Allclasses = []
            for ic in range(Nclasses):
                new_ll = EMlf.get_EM_Incomloglike_log(Ks_params[ic][1],Ks_params[ic][0],X = X_data_ave_test[ic])
                ll_Allclasses.append(new_ll)
            ll_test.append(copy.deepcopy(ll_Allclasses))
        
        # We change ll_train and ll_test to transpose them
        # Now they are just ll_train[Nclasses][Nclsuters]
        ll_train = np.array(ll_train).T
        ll_test = np.array(ll_test).T
        
        final_ll_train += ll_train;
        final_ll_test += ll_test;
    
    final_ll_train =  final_ll_train/((nfolds-1)*nfolds);
    final_ll_test = final_ll_test/nfolds;
        
    return final_ll_train, final_ll_test, All_Ks_params


def perfrom_CV_HMM_classes(Nclasses, X_data_trials, X_data_labels, nfolds = 2,
                          Klusters = [4,5,6,7,8,9], Ninit = 50, R = 100):
    # We compute the EM for both clusters for the classes.
    # We do it for different kluster sizes K. We train with the means of train and we validate with the means of test.
    

    final_ll_train = 0
    final_ll_test = 0
    # Example on how to do crossvalidation
    stkfold = cross_validation.StratifiedKFold(X_data_labels, n_folds = nfolds)
    for train_index, val_index in stkfold:
        X_train_notNormalized = [X_data_trials[itr] for itr in train_index]
        y_train = [X_data_labels[itr] for itr in train_index]
        
        X_test_notNormalized = [X_data_trials[iv] for iv in val_index]
        y_test = [X_data_labels[iv] for iv in val_index]

        X_data_ave_train = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                               normalize = True, partitions = nfolds -1)
        X_data_ave_test = dp.get_average_from_train(Nclasses, X_test_notNormalized, y_test, 
                                               normalize = True, partitions = 1)
    
        All_Ks_params = [];
        
        ll_train = []; # This has the likelihood for both class
        ll_test = []; # This has the likelihood for both classes
        
        for K in Klusters:
            Ninit = Ninit; K  =  K; verbose = 0; R  = R
            Ks_params = get_clusters_labels_HMM(Nclasses, X_train = X_data_ave_train, y_train = range(Nclasses), 
                                     Ninit = Ninit, I  =  K, R  = R, 
                                     verbose = verbose)
            print "Finished HMM"
            All_Ks_params.append(Ks_params)
#            print len(All_Ks_params[0])
#            print len(All_Ks_params)
            ## Get the likelihoods for the train and test sets
            ll_Allclasses = []
            for ic in range(Nclasses):
                pi = Ks_params[ic][0]
                A = Ks_params[ic][1]
                B = Ks_params[ic][2]
                print len(X_data_ave_train[ic])
                print X_data_ave_train[ic].shape
                
                new_train = HMMlf.get_HMM_Incomloglike(A,B,pi , [X_data_ave_train[ic]])
                ll_Allclasses.append(new_train)
            ll_train.append(ll_Allclasses)
            
            ll_Allclasses = []
            for ic in range(Nclasses):
                pi = Ks_params[ic][0]
                A = Ks_params[ic][1]
                B = Ks_params[ic][2]
        
                new_train = HMMlf.get_HMM_Incomloglike(A,B,pi , [X_data_ave_test[ic]])
                ll_Allclasses.append(new_train)
            ll_test.append(ll_Allclasses)
        
        # We change ll_train and ll_test to transpose them
        # Now they are just ll_train[Nclasses][Nclsuters]
        ll_train = np.array(ll_train).T
        ll_test = np.array(ll_test).T
        
        final_ll_train += ll_train;
        final_ll_test += ll_test;
    
    final_ll_train =  final_ll_train/((nfolds-1)*nfolds);
    final_ll_test = final_ll_test/nfolds;
        
    return final_ll_train, final_ll_test, All_Ks_params
    
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.
from sklearn.lda import LDA
from sklearn.qda import QDA


    # GridSearchCV implements a CV over a variety of Parameter values !! 
    # In this case, over C fo the linear case, C and "degree" for the poly case
    # and C and "gamma" for the rbf case. 
    # The parameters we have to give it are:
    # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
    # 2 -> Subset of parameters to validate. C 
    # 3 -> Type of validation: K-fold
    # 4 -> Scoring function. sklearn.metrics.accuracy_score


def get_LogReg(Xtrain, Xtest, Ytrain, Ytest):
        lr = LogisticRegression()
        lr.fit(Xtrain,Ytrain)
        scores = np.empty((4))
        scores[0] = lr.score(Xtrain,Ytrain)
        scores[1] = lr.score(Xtest,Ytest)
        print('Logistic Regression, train: {0:.02f}% '.format(scores[0]*100))
        print('Logistic Regression, test: {0:.02f}% '.format(scores[1]*100))
        return lr
        

def get_LDA(Xtrain, Xtest, Ytrain, Ytest):
        lda = LDA()
        lda.fit(Xtrain,Ytrain)
        scores = np.empty((4))
        scores[0] = lda.score(Xtrain,Ytrain)
        scores[1] = lda.score(Xtest,Ytest)
        print('LDA, train: {0:.02f}% '.format(scores[0]*100))
        print('LDA, test: {0:.02f}% '.format(scores[1]*100))
        
        return lda
        
#%% QDA Classification
'''
La clasificacion con QDA falla porque el metodo emplea matrices de covarianza para
cada clase (al contratrio de LDA que calcula una para todos los datos). Por tanto
si existen pocos datos en una clase la matriz de coverianza va a ser de rango
deficiente y causara problemas en la clasificacion.
'''

def get_QDA(Xtrain, Xtest, Ytrain, Ytest):
    qda = QDA()
    qda.fit(Xtrain,Ytrain)
#    predLabels = qda.predict(Xtest)
#    print("Classification Rate Test QDA: " + str(np.mean(Ytest==predLabels)*100) + " %")
    scores = np.empty((4))
    scores[0] = qda.score(Xtrain,Ytrain)
    scores[1] = qda.score(Xtest,Ytest)
    print('QDA, train: {0:.02f}% '.format(scores[0]*100))
    print('QDA, test: {0:.02f}% '.format(scores[1]*100))
    return qda
    
def get_GNB(Xtrain, Xtest, Ytrain, Ytest):
    gnb = GaussianNB()
    gnb.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = gnb.score(Xtrain,Ytrain)
    scores[1] = gnb.score(Xtest,Ytest)
    print('GNB, train: {0:.02f}% '.format(scores[0]*100))
    print('GNB, test: {0:.02f}% '.format(scores[1]*100))
    return gnb
    
def get_LSVM(Xtrain, Xtest, Ytrain, Ytest):
    C = np.logspace(-3,3,10)

    # Create dictionaries with the Variables for the validation !
    # We create the dictinary for every TYPE of SVM we are gonna use.
    param_grid_linear = dict()
    param_grid_linear.update({'kernel':['linear']})
    param_grid_linear.update({'C':C})
    
    # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    gsvml = GridSearchCV(SVC(class_weight='balanced'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvml.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = gsvml.score(Xtrain,Ytrain)
    scores[1] = gsvml.score(Xtest,Ytest)
    print('LSVM, train: {0:.02f}% '.format(scores[0]*100))
    print('LSVM, test: {0:.02f}% '.format(scores[1]*100))
    return gsvml

def get_SVM_rf(Xtrain, Xtest, Ytrain, Ytest):
        # Parameters for the validation
        C = np.logspace(-3,3,20)
        gamma = np.array([0.125,0.25,0.5,1,2,4])/200
        
        param_grid_rbf = dict()
        param_grid_rbf.update({'kernel':['rbf']})
        param_grid_rbf.update({'C':C})
        param_grid_rbf.update({'gamma':gamma})
        
        stkfold = StratifiedKFold(Ytrain, n_folds = 5)
        
        # The score function is the one we want to minimize or maximize given the label and the predicted.
        acc_scorer = make_scorer(accuracy_score)

        gsvmr = GridSearchCV(SVC(class_weight='balanced'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)

        gsvmr.fit(Xtrain,Ytrain)
        scores = np.empty((4))
        scores[0] = gsvmr.score(Xtrain,Ytrain)
        scores[1] = gsvmr.score(Xtest,Ytest)
        print('SVM_rf, train: {0:.02f}% '.format(scores[0]*100))
        print('SVM_rf, test: {0:.02f}% '.format(scores[1]*100))
        return gsvmr
        
def get_KNN(Xtrain, Xtest, Ytrain, Ytest):
    # Perform authomatic grid search
    params = [{'n_neighbors':np.arange(1,10)}]
    gknn = GridSearchCV(KNeighborsClassifier(),params,scoring='precision',cv=4,refit=True,n_jobs=-1)
    gknn.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = gknn.score(Xtrain,Ytrain)
    scores[1] = gknn.score(Xtest,Ytest)
    print('{0}-NN, train: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[0]*100))
    print('{0}-NN, test: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[1]*100))
    
    return gknn
    
def get_TreeCl(Xtrain, Xtest, Ytrain, Ytest):
   #%% Tree Classifier
    param_grid = dict()
    param_grid.update({'max_features':['auto',"log2", 'sqrt']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring='precision',cv=5,refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)
    scores = np.empty((2))
    scores[0] = gtree.score(Xtrain,Ytrain)
    scores[1] = gtree.score(Xtest,Ytest)
    print('Decision Tree, train: {0:.02f}% '.format(scores[0]*100))
    print('Decision Tree, test: {0:.02f}% '.format(scores[1]*100))
    
    return gtree
    
def get_RF(Xtrain, Xtest, Ytrain, Ytest, gtree):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=1000,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,oob_score=True,n_jobs=-1)
    rf.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = rf.score(Xtrain,Ytrain)
    scores[1] = rf.score(Xtest,Ytest)
    print('Random Forest, train: {0:.02f}% '.format(scores[0]*100))
    print('Random Forest, test: {0:.02f}% '.format(scores[1]*100))

    return rf
    
def get_ERT(Xtrain, Xtest, Ytrain, Ytest, gtree):
    # Extremely Randomized Trees
    ert = ExtraTreesClassifier(n_estimators=1000,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,n_jobs=-1)
    ert.fit(Xtrain,Ytrain)
    scores = np.empty((2))
    scores[0] = ert.score(Xtrain,Ytrain)
    scores[1] = ert.score(Xtest,Ytest)
    print('Extremely Randomized Trees, train: {0:.02f}% '.format(scores[0]*100))
    print('Extremely Randomized Trees, test: {0:.02f}% '.format(scores[1]*100))

    return ert
        
def get_SVM(Xtrain, Xtest, Ytrain, Ytest):
        # Parameters for the validation
        C = np.logspace(-3,3,10)
        p = np.arange(2,5)
        gamma = np.array([0.125,0.25,0.5,1,2,4])/200
        
        # Create dictionaries with the Variables for the validation !
        # We create the dictinary for every TYPE of SVM we are gonna use.
        param_grid_linear = dict()
        param_grid_linear.update({'kernel':['linear']})
        param_grid_linear.update({'C':C})
        
        param_grid_pol = dict()
        param_grid_pol.update({'kernel':['poly']})
        param_grid_pol.update({'C':C})
        param_grid_pol.update({'degree':p})
        
        param_grid_rbf = dict()
        param_grid_rbf.update({'kernel':['rbf']})
        param_grid_rbf.update({'C':C})
        param_grid_rbf.update({'gamma':gamma})
        
        
        param = [{'kernel':'linear','C':C}]
        param_grid = [param_grid_linear,param_grid_pol,param_grid_rbf]
        
        # Validation is useful for validating a parameter, it uses a subset of the 
        # training set as "test" in order to know how good the generalization is.
        # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
        stkfold = StratifiedKFold(Ytrain, n_folds = 5)
        
        # The score function is the one we want to minimize or maximize given the label and the predicted.
        acc_scorer = make_scorer(accuracy_score)
    
        # GridSearchCV implements a CV over a variety of Parameter values !! 
        # In this case, over C fo the linear case, C and "degree" for the poly case
        # and C and "gamma" for the rbf case. 
        # The parameters we have to give it are:
        # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
        # 2 -> Subset of parameters to validate. C 
        # 3 -> Type of validation: K-fold
        # 4 -> Scoring function. sklearn.metrics.accuracy_score
    
        gsvml = GridSearchCV(SVC(class_weight='balanced'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
        gsvmp = GridSearchCV(SVC(class_weight='balanced'),param_grid_pol, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
        gsvmr = GridSearchCV(SVC(class_weight='balanced'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
        
        gsvml.fit(Xtrain,Ytrain)
        gsvmp.fit(Xtrain,Ytrain)
        gsvmr.fit(Xtrain,Ytrain)
        
        trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
        testscores = [gsvml.score(Xtest,Ytest),gsvmp.score(Xtest,Ytest),gsvmr.score(Xtest,Ytest)]
        maxtrain = np.amax(trainscores)
        maxtest = np.amax(testscores)
        print('SVM, train: {0:.02f}% '.format(maxtrain*100))
        print('SVM, test: {0:.02f}% '.format(maxtest*100))
    
        return gsvmr