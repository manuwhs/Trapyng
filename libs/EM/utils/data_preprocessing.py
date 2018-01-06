
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

plt.close("all")

################################################################
######## Load the dataset ! ##############################
###############################################################

def load_one_person_dataset(dataset_folder = "./dataset/", filename = 'face_scrambling_spm8proc_sub07.mat'):
    dataset_folder = "./dataset/"
    mat = scipy.io.loadmat(dataset_folder + filename)
    keys = mat.keys()
#    print keys
    X = mat["X"]   # Nchannels x Time x Ntrials
    trial_indices = mat["trial_indices"][0][0]  # Labels of the trials
    label_classes = ["Famous", "Unfamiliar", "Scrambled"]

    ############## Separate by classes in lists ###############
    X_All_labels = []  # Contains the trials for every label
    for label in label_classes:
        label_trials = trial_indices[label].flatten()
        X_label_trials = X[:,:,np.where(label_trials == 1)[0]].T
        X_All_labels.append(X_label_trials)
    # Now X_All_labels has in every postion the trials for each class in the form
    # of a matrix Ntrials x Ntimes x Ndim
    return X_All_labels, label_classes
    
################################################################
######## Loading in lists and preprocessing! ##############################
###############################################################
def preprocess_data_set (X_All_labels, label_classes, 
                         max_trials = 100, channel_sel = None,
                         normalize = True):
    # Subselect channels and trials and then normalize modulus
                         
    max_trials = max_trials  # Maximum trials, preprocessed per class, to reduce computation
    channel_sel = channel_sel  # Subset of selected channels

    Nclasses = len(label_classes)
    label_numbers = range(Nclasses)
    
    X_data_trials = []  # List of trials :)
    X_data_labels = []  # Labels of the trials :)

    # Every label will have set of clusters, 
    # this is the parameters of the clusters [K][[pimix], [thetas]]
    
    ############# Preprocess by class ? ###################
    for i in range(Nclasses):
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        # Limit the number of trials processed by labels
        Ntrials = np.min([Ntrials,max_trials])
        if (type(channel_sel) == type(None)):
            channel_sel = range(Ndim)
            
        for nt in range(Ntrials):
            ################################################################
            ######## Preprocessing ! ##############################
            ###############################################################
            X_trial = X_All_labels[i][nt,:,:]
            X_trial = X_trial[:,channel_sel]
#            X_trial = X_trial - np.sum(X_trial, axis = 1).reshape(X_trial.shape[0],1)
    #        scaler = preprocessing.StandardScaler().fit(X_trial)
    #        X_trial = scaler.transform(X_trial)    
            if (normalize == True):
                X_trial = gf.normalize_module(X_trial)
            
            X_data_trials.append(X_trial)
            X_data_labels.append(i)
            
    # Now have a normal machine learning problem :)
    # X_data_trials,  X_data_labels
    
    return X_data_trials, X_data_labels

def normalize_trialList(tL):
    # Normalize all the trials in a list
    tL_norm = []
    for i in range (len(tL)):  # Put the data in the sphere.
        tL_norm.append( gf.normalize_module(tL[i]))
    return tL_norm

def get_sublist(li, indexes = [0,1]):
    # Gets a new list with the indexes given
    sublist = []
    for i in indexes:
        sublist.append(li[i])
    return sublist

def get_subTraining_list(li, indexes = [0,1]):
    subtraining_list = [[]] * len(li)
    for i in range(len(li)):
        subtraining_list[i].extend(get_sublist(li[i], indexes))
        subtraining_list[i] = np.array(subtraining_list[i])
    return subtraining_list
    
def get_average_from_train(Nclasses, X_train_notNormalized, y_train,
                           normalize = True, partitions = 1):
    ## We feed it with the average of the X_train, before normalizing
    ## Create mean profile only with training samples !
    X_All_labels_train = []
    for i in range(Nclasses):
        X_train_class_i = [ X_train_notNormalized[j] for j in np.where(np.array(y_train) == i)[0]]
        X_train_class_i = np.array(X_train_class_i)
        X_All_labels_train.append(X_train_class_i)
    
    if (partitions == 1):  # Only 1 partition
        X_data_ave = get_timeSeries_average_by_label(X_All_labels_train)

        if (normalize):    
            X_data_ave = normalize_trialList(X_data_ave) 
    else:                 # Several partitions
        # TODO: change this because different number of classes might have different number of elements
        X_data_ave = []  # Nunca mas hacer N*[[]]
        for class_i in range(Nclasses):
            X_data_ave.append([])
        Nelem = []
        num = []
        
        for class_i in range(Nclasses):
            Nelem.append (len(X_All_labels_train[class_i]))
            num.append( int(Nelem[class_i]/partitions))
            
#        print Nelem, num, X_data_ave
#        print range(partitions - 1)
        for i in range(partitions - 1):
            for class_i in range(Nclasses):
#                print "fr"
                st_indx = i*num[class_i]
                end_indx = (i+1)*num[class_i]
                
                X_All_labels_train_partition = get_subTraining_list([X_All_labels_train[class_i]], range (st_indx, end_indx))
                
#                print len(X_All_labels_train_partition),len(X_All_labels_train_partition[0])  
               
                X_data_ave_p = get_timeSeries_average_by_label(X_All_labels_train_partition)
#                print "fgrgr"
#                print len(X_data_ave[0]), i, class_i
                X_data_ave[class_i].append(X_data_ave_p[0])
#                print len(X_data_ave[0]), i, class_i
                
        # Last partition to create
        for class_i in range(Nclasses):
            st_indx = num[class_i]*(partitions - 1) 
            end_indx = Nelem[class_i]
            
#            print range (st_indx, end_indx)
            X_All_labels_train_partition = get_subTraining_list([X_All_labels_train[class_i]], range (st_indx, end_indx))
            X_data_ave_p = get_timeSeries_average_by_label(X_All_labels_train_partition)
        
            X_data_ave[class_i].append(X_data_ave_p[0])
#            print len(X_data_ave[0])
            X_data_ave[class_i] = np.concatenate(X_data_ave[class_i], axis = 0)
        
        
        if (normalize):    
#            print len(X_data_ave), X_data_ave[0].shape
            X_data_ave = normalize_trialList(X_data_ave)
            
    return X_data_ave
    
# Get at each time instant, the average data
def get_timeSeries_average_by_label (X_All_labels, channel_sel = None):
    Nclasses = len(X_All_labels)
    X_data_ave = []  # List of trials :)
    
    
    # Every label will have set of clusters, 
    # this is the parameters of the clusters [K][[pimix], [thetas]]
    ############# Preprocess by class ? ###################
    for i in range(Nclasses):
#        print X_All_labels[i].shape
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        if (type(channel_sel) == type(None)):
            channel_sel = range(Ndim)
        X_trial_ave_label = np.mean(X_All_labels[i][:,:,channel_sel] , axis = 0)
        X_data_ave.append(X_trial_ave_label)
    return X_data_ave
    
# Removes average of each channel (70) for each sample of each trial
def remove_timePoints_average (X_All_labels):
    Nclasses = len(X_All_labels)
    for i in range(Nclasses):
        caca = X_All_labels[i].shape
#        print caca
        Ntrials, Nsamples, Ndim = caca
        X_trial_ave_label = np.mean(X_All_labels[i][:,:,:] , axis = 2)
        X_All_labels[i] = X_All_labels[i] - X_trial_ave_label.reshape(Ntrials, Nsamples,1)
#        X_trial_ave_label = gf.normalize_module(X_trial_ave_label)
    return X_All_labels

def remove_features_average (X_All_labels):
    # This function gets the 
    # This function gets the the ave
    Nclasses = len(X_All_labels)
    X_data_ave = []
    for i in range(Nclasses):
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        X_trial_ave_label = np.mean(X_All_labels[i][:,:,:] , axis = 0)
        X_time_ave = np.mean(X_trial_ave_label, axis = 0)
        X_data_ave.append(X_time_ave)

    X_ave = 0
    for X_i in X_data_ave:
        X_ave += X_i
    X_ave = X_ave / len(X_data_ave)
#    print Ndim
#    print X_ave.shape
    X_ave = X_ave.reshape(1,Ndim)
    
    for i in range(Nclasses):
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        for nt in range(Ntrials):
            X_trial = X_All_labels[i][nt,:,:]
            X_trial = X_trial - X_ave
            
    return X_All_labels
    
    
# Get at each time instant, run the EM 2D and get the main direction
def get_labels_ave_EM (X_All_labels, label_classes, 
                         max_trials = 100, channel_sel= []):

    max_trials = max_trials  # Maximum trials, preprocessed per class, to reduce computation

    Nclasses = len(label_classes)

    X_data_ave_EM_plus = []  # List of trials :)
    X_data_ave_EM_minus = []

    # Every label will have set of clusters, 
    # this is the parameters of the clusters [K][[pimix], [thetas]]
    
    Ninit = 5
    K = 2
    verbose = 0
    T = 20
    ############# Preprocess by class ? ###################
    for i in range(Nclasses):
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        
        X_ave_EM_class_plus = []
        X_ave_EM_class_minus = []
        for i_sample in range(Nsamples):
            print "%i / %i " %(i_sample, Nsamples)
            X_trial_samples = X_All_labels[i][:,i_sample,channel_sel]
            X_trial_samples = gf.normalize_module(X_trial_samples)
#            print X_trial_samples.shape
            
            logl,theta_list,pimix_list = EMl.run_several_EM(X_trial_samples, K = K, delta = 0.1, T = T,
                                        Ninit = Ninit, verbose = verbose)
            
            good_cluster_indx_plus = np.argmax(theta_list[-1][1])
            good_cluster_indx_minus = np.argmin(theta_list[-1][1])
            
            mu_plus = theta_list[-1][0][:,[good_cluster_indx_plus]]
            mu_minus = theta_list[-1][0][:,[good_cluster_indx_minus]]
#            print mu.shape
            X_ave_EM_class_plus.append(mu_plus.T)
            X_ave_EM_class_minus.append(mu_minus.T)
            
        X_ave_EM_class_plus = np.concatenate(X_ave_EM_class_plus, axis = 0)
        X_ave_EM_class_minus = np.concatenate(X_ave_EM_class_minus, axis = 0)
#        print X_ave_EM_class.shape
        X_data_ave_EM_plus.append(X_ave_EM_class_plus)
        X_data_ave_EM_minus.append(X_ave_EM_class_minus)
        
    return X_data_ave_EM_plus, X_data_ave_EM_minus
    
#def get_clusters_of_labels_EM():
    
# Get fot every trial, a 2-EM to see its components
def get_X_trials_EM (X_All_labels, label_classes, 
                         max_trials = 100, channel_sel= [1,4,10]):

    max_trials = max_trials  # Maximum trials, preprocessed per class, to reduce computation

    Nclasses = len(label_classes)

    X_data_ave_EM_plus = []  # List of trials :)
    X_data_ave_EM_minus = []
    # Every label will have set of clusters, 
    # this is the parameters of the clusters [K][[pimix], [thetas]]
    
    Ninit = 5
    K = 2
    verbose = 0
    T = 20
    ############# Preprocess by class ? ###################
    for i in range(Nclasses):
        Ntrials, Nsamples, Ndim = X_All_labels[i].shape
        Ntrials = np.min([Ntrials,max_trials])
        X_ave_EM_class_plus = []
        X_ave_EM_class_minus = []
        for i_trial in range(Ntrials):
            print "%i / %i " %(i_trial, Ntrials)
            # TODO no entiendo por que pelotas tengo que transformar aqui
            X_trial_samples = X_All_labels[i][i_trial,:,channel_sel].T
#            print X_trial_samples.shape
#            print channel_sel
            X_trial_samples = gf.normalize_module(X_trial_samples)
#            print X_trial_samples.shape
#            print X_trial_samples.shape
            logl,theta_list,pimix_list = EMl.run_several_EM(X_trial_samples, K = K, delta = 0.1, T = T,
                                        Ninit = Ninit, verbose = verbose)
            
            good_cluster_indx_plus = np.argmax(theta_list[-1][1])
            good_cluster_indx_minus = np.argmin(theta_list[-1][1])
            
            mu_plus = theta_list[-1][0][:,[good_cluster_indx_plus]]
            mu_minus = theta_list[-1][0][:,[good_cluster_indx_minus]]
#            print mu.shape
            X_ave_EM_class_plus.append(mu_plus.T)
            X_ave_EM_class_minus.append(mu_minus.T)
            
        X_ave_EM_class_plus = np.concatenate(X_ave_EM_class_plus, axis = 0)
        X_ave_EM_class_minus = np.concatenate(X_ave_EM_class_minus, axis = 0)
#        print X_ave_EM_class.shape
        X_data_ave_EM_plus.append(X_ave_EM_class_plus)
        X_data_ave_EM_minus.append(X_ave_EM_class_minus)
        
    return X_data_ave_EM_plus, X_data_ave_EM_minus
    
def get_likelihoods_EM(Xdata,Ks_params):
    # Xdata is a list of trials
    Nclasses = len(Ks_params)
    Likelihoods = []
    ### Calculate test accuracy
    for trial_i in range(len(Xdata)): 
            likeli = []
            for K_p_i in range(Nclasses):   # For every cluster type
                ll = EMlf.get_EM_Incomloglike_log(Ks_params[K_p_i][1],Ks_params[K_p_i][0],Xdata[trial_i])
                likeli.append(ll)
            Likelihoods.append(likeli)
    
    Likelihoods = np.array(Likelihoods)
    return Likelihoods

def get_likelihoods_byClusters_EM(Xdata,Ks_params):
    # Xdata is a list of trials
    Nclasses = len(Ks_params)
    Likelihoods = []
    ### Calculate test accuracy
    for trial_i in range(len(Xdata)): 
            likeli = []
            for K_p_i in range(Nclasses):   # For every cluster type
#                print Xdata[trial_i].shape
                ll = EMlf.get_EM_Incomloglike_byCluster_log(Ks_params[K_p_i][1],Ks_params[K_p_i][0],Xdata[trial_i])
                likeli.append(ll)
            likeli = np.concatenate(likeli, axis = 1)
            Likelihoods.append(likeli)
    
    Likelihoods = np.concatenate(Likelihoods, axis = 0)
    return Likelihoods

def get_likelihoods_byClusters_EM_onlyOwn(Xdata,Ks_params):
    # Xdata is a list of trials
    Nclasses = len(Ks_params)
    Likelihoods = []
    ### Calculate test accuracy
    for trial_i in range(len(Xdata)): 
            likeli = []
            for K_p_i in range(Nclasses):   # For every cluster type
#                print Xdata[trial_i].shape
                ll = EMlf.get_EM_Incomloglike_byCluster_log(Ks_params[K_p_i][1],Ks_params[K_p_i][0],Xdata[trial_i])
                likeli.append(ll)
            likeli = np.concatenate(likeli, axis = 1)
            Likelihoods.append(likeli)
    
    Likelihoods = np.concatenate(Likelihoods, axis = 0)
    return Likelihoods
    
def plot_Clusters_time(Xdata,Ks_params):
    # this function takes the list of trials and computes the likelihoods
    # of every sample of every trial belonging to a cluster, sums them up and
    # then plots them

    Ntrials = len(Xdata)
    Nclasses = len (Ks_params)
    
    total_llresp = []
    for K_p_i in range(Nclasses): 
        total_llresp.append([0])
    for trial_i in range(Ntrials): 
            for K_p_i in range(Nclasses):   # For every cluster type
#                print Xdata[trial_i].shape
                ll_resp =  EMlf.get_responsabilityMatrix_log(Xdata[trial_i],Ks_params[K_p_i][1],Ks_params[K_p_i][0])
                ll_resp = np.exp(ll_resp)
#    for nsample
                total_llresp[K_p_i] += ll_resp
#                total_llresp[K_p_i] = HMMlf.sum_logs([total_llresp[K_p_i],ll_resp])
    for K_p_i in range(Nclasses): 
        total_llresp[K_p_i] = total_llresp[K_p_i]/Ntrials

    
#    total_llresp[0] = np.log (total_llresp[0])
#    total_llresp[1] = np.log (total_llresp[1])
    # Now we have the sum of log responsabilities !!
    # Product for chains !! 
    
    
#    gl.plot([],total_llresp[0])
#    gl.plot([],total_llresp[1])
    for K_p_i in range(Nclasses): 
        pi = Ks_params[K_p_i][0]
        NK = pi.size
        labels = []
        for k in range(NK):
            labels.append("Cluster " + str(k +1))
        gl.plot_filled([],total_llresp[K_p_i], labels = ["Responsability EM", "Time index","Responsability"], legend = labels)

def plot_Clusters_HMM_time(Xdata,Ks_params):
    # this function takes the list of trials and computes the likelihoods
    # of every sample of every trial belonging to a cluster, sums them up and
    # then plots them

    Ntrials = len(Xdata)
    Nclasses = len (Ks_params)
    
    total_llresp = []
    for K_p_i in range(Nclasses): 
        total_llresp.append([0])
    for trial_i in range(Ntrials): 
            for K_p_i in range(Nclasses):   # For every cluster type
#                print Xdata[trial_i].shape
                A = Ks_params[K_p_i][1]
                B = Ks_params[K_p_i][2]
                pi = Ks_params[K_p_i][0]
                alpha = HMMlf.get_alfa_matrix_log(A,B,pi, [Xdata[trial_i]])
                beta = HMMlf.get_beta_matrix_log(A,B,pi, [Xdata[trial_i]])
                ll_resp =  HMMlf.get_gamma_matrix_log(alpha, beta)[0]
#                print ll_resp[0].shape
                ll_resp = np.exp(ll_resp).T
                
#    for nsample
                total_llresp[K_p_i] += ll_resp
#                total_llresp[K_p_i] = HMMlf.sum_logs([total_llresp[K_p_i],ll_resp])
    for K_p_i in range(Nclasses): 
        total_llresp[K_p_i] = total_llresp[K_p_i]/Ntrials

    
#    total_llresp[0] = np.log (total_llresp[0])
#    total_llresp[1] = np.log (total_llresp[1])
    # Now we have the sum of log responsabilities !!
    # Product for chains !! 
    
    
#    gl.plot([],total_llresp[0])
#    gl.plot([],total_llresp[1])

    
    for K_p_i in range(Nclasses): 
        pi = Ks_params[K_p_i][0]
        NK = pi.size
        labels = []
        for k in range(NK):
            labels.append("Cluster " + str(k +1))
        gl.plot_filled([],total_llresp[K_p_i], labels = ["Responsability HMM", "Time index","Responsability"], legend = labels)

def get_likelihoods_HMM(Xdata,Is_params):
    Nclasses = len(Is_params)
    Likelihoods = []
    for trial_i in range(len(Xdata)): 
            likeli = []
            for K_p_i in range(Nclasses):   # For every cluster type
                ll = HMMlf.get_HMM_Incomloglike(Is_params[K_p_i][1],  # A,B,pi
                                                Is_params[K_p_i][2],
                                                Is_params[K_p_i][0],
                                                [Xdata[trial_i]])
                likeli.append(ll)
            Likelihoods.append(likeli)
    
    Likelihoods = np.array(Likelihoods)
    return Likelihoods
 