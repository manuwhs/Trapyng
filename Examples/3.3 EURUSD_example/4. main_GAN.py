"""
This file implements several classifications systems over the selected features
"""
# 
# Load all the directories needed for the code to be executed both from
# a console and Spyder, and from the main directory and the local one.
# Ideally this code should be executed using Spyder with working directory

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Specific utilities
import t_utils as tut
import baseClassifiersLib as bCL
import basicMathlib as bMl
import indicators_lib as indl
import pickle_lib as pkl 
import utilities_lib as ul

import pyTorch_utils as pytut
from GeneralVBModel import GeneralVBModel
from GAN import GAN_reg1
import Variational_inferences_lib as Vil

import pyTorch_utils as pytut
import data_loaders as dl
import config_files as cf 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import plotting_func as pf
import VAE_plotting_func as Vpf
plt.close("all") # Close all previous Windows

# 
"""
################### EXECUTING OPTIONS ###################
"""

folder_model = "../models/pyTorch/EURUSD/VAE1/"
folder_images = "../pics/EURUSD/GAN/"
storage_folder = ".././storage/EURUSD/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"
video_fotograms_folder_training  = folder_images +"Training/"

ul.create_folder_if_needed(folder_images)
ul.create_folder_if_needed(video_fotograms_folder_training)
ul.create_folder_if_needed(folder_model)
    
    
load_data = 1
preprocessing_data = 1
extract_features = 0

create_video_training = 0
#################### DATA AND DEVICES OPTIONS ############
linear_data = 0  # Linear or sinusoid data
dtype = torch.float
#device = torch.device("cpu")
device = pytut.get_device_name(cuda_index = 0)

load_previous_state = 0
train_model = 1
Step_video = 1
"""
  ################  INPUT DATA OPTIONS  ###############
"""

# Using the library of function built in using the dataFrames in pandas
typeChart = "Bar"  # Line, Bar, CandleStick
tranformIntraday = 1

symbols = ["EURUSD"]
periods = [15]  # 1440 15

######## SELECT DATE LIMITS ###########
## We set one or other as a function of the timeSpan

sdate_str = "01-01-2010"
edate_str = "31-12-2016"

sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

"""
###################### Setting up func ###########################
"""
## Set up seeds for more reproducible results
time_now = dt.datetime.now() 
random_seed = time_now.microsecond
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
    
## Windows and folder management
plt.close("all") # Close all previous Windows

    
#
if (load_data):
    ######## CREATE THE OBJECT AND LOAD THE DATA ##########
    # Tell which company and which period we want
    timeData = CTD.CTimeData(symbols[0],periods[0])
    timeData.set_csv(storage_folder,file_name)  # Load the data into the model

if (preprocessing_data):
    timeData = tut.preprocess_data(timeData, sdate,edate)
    ## Get the valid trading days sorted
    days_keys, day_dict = timeData.get_indexDictByDay()
    Ndays = len(days_keys)

    timeData_daily = tut.get_daily_timedata(timeData, symbols[0])
    H,L,O,C,V = np.array(timeData_daily.TD[["High","Low","Open","Close","Volume"]][:]).T
    
if (1):
    
    """
    ############## TRANSFORM DATA INTO PYTORCH ###################
    """
    
    prop_train = 0.8
    Nsamples = len(days_keys)
    Nsamples_train = int(prop_train*Nsamples)
    
    all_samples_tr = []
    all_samples_tst = []
    for i in range(Nsamples): # (len(days_keys)):
        samples_index = day_dict[days_keys[i]]
        C = np.array([timeData.TD["Close"][samples_index]])
        C = 100*(C - C[0,0])/ C[0,0]
        if (i < Nsamples_train):
            all_samples_tr.append(C)
        else:
            all_samples_tst.append(C)

    X_data_tr = np.concatenate(all_samples_tr, axis = 0)
    X_data_val = np.concatenate(all_samples_tst, axis = 0)
    
    ## Turn data into pyTorch Tensors !!
    Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)
    Xval = torch.tensor(X_data_val,device=device, dtype=dtype)

    
    """
    ######################## Config files ########################
    """
    Ntr, D_in = Xtrain.shape
#    Nval, D_out = Ytrain.shape
    criterion = nn.MSELoss()
    ## LOAD THE BASE DATA CONFIG
    cf_a = cf.cf_a_GAN_EURUSD
    cf_a.eta_KL = eta_KL
    ## Set data and device parameters
    cf_a.D_in = D_in 
#    cf_a.D_out = D_out 
    cf_a.dtype = dtype  # Variable types
    cf_a.device = device
    # Set other training parameters
    cf_a.loss_func = criterion
        
    """
    ######################## CREATE DATA GENERATORS !! ########################
    """
    train = data_utils.TensorDataset(Xtrain)
    train_loader = data_utils.DataLoader(train, batch_size=cf_a.batch_size_train, shuffle=True)

    """
    ######################## Instantiate Architecture ########################
    """
    cf_a.Nsamples_train = Ntr

    myGAN = GAN_reg1(cf_a).to(device = device)
    # Set the model in training mode for the forward pass. self.training = True
    # This is for Dropout and VI
    myGAN.train()  
    
    if (load_previous_state):
        files_path = ul.get_allPaths(folder_model)
        if (len(files_path) >0):
            # Load the latest params !
            files_path = sorted(files_path, key = ul.cmp_to_key(ul.filenames_comp))
            myGAN.load(files_path[-1])
            

    """
     ###################3 TRAINING ##############################
    """
    
    myGAN.eval()
    if (train_model):
        tr_data_loss = []
        val_data_loss = [] 
        
        KL_loss_tr = []
        KL_loss_val = []
        
        final_loss_tr = []
        final_loss_val = []
    
        for i_epoch in range(cf_a.Nepochs):
            print ("Epoch %i/%i"%(i_epoch+1,cf_a.Nepochs))
            
            ## Train one Epoch !!!
            myGAN.train()  
            for local_batch in train_loader:
#                num_iter_row += 1
#                D_loss = myGAN.train_Disc_batch(local_batch[0])
#                
#                if(num_iter_row == N_batches_on_a_row):
#                    for jj in range(N_batches_on_a_row):
#                        G_loss = myGAN.train_Gen_batch(local_batch[0]) 
#                        pass
#                    num_iter_row = 0
                D_loss, G_loss = myGAN.train_batch(local_batch[0])
#                print("Loss D: ",D_loss.item(), "Loss G: ", G_loss.item())
    
            ## Plot smaples !!
            if(i_epoch%5 == 0):
                myGAN.eval()  
                gl.init_figure()
                ax1 = gl.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
                ax2 = gl.subplot2grid((2,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
                
                #### Plot examples or real samples
                alpha = 0.5
                x = []
                
                for i in range(10):
                    Xtrain_sample = Xtrain[i,:].detach().cpu().numpy()
                    gl.plot(x,Xtrain_sample , ax = ax1,legend = []  , color = "b",alpha = alpha,
                                    labels = ["Real","",r"Rate"], AxesStyle = "Normal - No xaxis")
            
                #### Plot examples or fake samples
                for i in range(10):
                    Xtrain_sample = Xtrain[i,:]
                    generated = myGAN.Gen.generate_samples(1).detach().cpu().numpy()
                    gl.plot(x,generated , ax = ax2,legend = []  , color = "b",alpha = alpha,
                                    labels = ["Fake","",r"Rate"], AxesStyle = "Normal - No xaxis")
                    
                ## Final touches
                gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                                  legend = 10, xticks = 10, yticks = 10)
            
                image_name = "reconstrunction_" + str(i_epoch)
                gl.savefig(folder_images + image_name, 
                           dpi = 100, sizeInches = [20, 7], close = True)
                
            
            
        """
        ######################  SAVE MODEL ####################
        """
        myGAN.save(folder_model + "model_parameters_epoch:%i.pk"%i)
        
        """
        #####################  STATIC PLOTS ###################### 
            """
        if(create_video_training):  #
            pf.create_video_from_images(video_fotograms_folder_training,output_file =folder_images +  "/training_weights.avi", fps = 2)
                

