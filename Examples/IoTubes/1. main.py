"""
In this document we will generate a set of fake data and plot it every second
"""


import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../../")
import import_folders
import CUpdateable_Chart as CU
from graph_lib import gl
import numpy as np
import tasks_lib as tl
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import serial as serrial
import FTP_lib 

gl.close("all")
plt.ion()

if(0):
    def upload_files(localfolder = "./FTP_added", remotefolder = "wp-content/uploads/FTP_added"):
        remotefolder = "./"
        mySFTP.put_dir(localfolder, remotefolder,  recursive = True)
        print mySFTP.get_list_dir(subfolder = remotefolder);
        
    ### Directories to play with
    directory = "wp-content/uploads/2018/04/"
    directory = "./"
    
    source_directory = "wp-content/uploads/2018/"
    
    new_directory ="wp-content/uploads/2018/new/"
    remotefile = "One-time-buy-150x150.png"
    localfile = "./" + remotefile
    localdir = "./"
    
    mySFTP = FTP_lib.SFTP_class()
    #print mySFTP.get_list_dir(subfolder = directory);
    #mySFTP.get_dir("./", "./content",recursive = True)
    
    if(0):
        mySFTP.get_file(directory+remotefile, localfile)
        mySFTP.get_dir(directory, localdir + directory)
    
    if(0):
        
        
        #print mySFTP.get_list_dir(subfolder = source_directory);
        mySFTP.put_file(localfile, new_directory + remotefile)
        print mySFTP.get_list_dir(subfolder = new_directory);
        
        ## Copy a dir:
        fas = open("./Other/./One-time-buy-150x150.png")
        mySFTP.put_dir("./Other/", new_directory,  recursive = True)
        print mySFTP.get_list_dir(subfolder = new_directory);
    
    
        mySFTP.get_dir(new_directory, "./Download/")
    
    if(0):
        html_file_path = "/cleaning_trials/"
        mySFTP.get_file(directory+remotefile, localfile)
    
    upload_files()

###### SET THE READING ########
get_data_from_Serial = False
if(get_data_from_Serial):
    
    ## Set the serial
    ser =None
    for i in range(8):
        try:
            ser = serrial.Serial('/dev/ttyUSB%i'%i, 9600)
            break
        except:
            print("Port %i not connected "%i)
    ## First test of the serial
    print("Testing the Serial Port")
    ser.readline()
    for i in range(2):
        print (float(ser.readline().decode("utf-8").split("\n")[0]))
else:
    ###### GENERATE FAKE DATA ############
    ser = None

if(1):
    mySource = CU.Csource(ser); ## Source of data
    
    myUpdatable_Chart = CU.CUpdate_chart(name = "IoTubes trial", source = mySource);
    myUpdatable_Chart.init_figure();

#myUpdatable_Chart.auto_update_test()

#myUpdatable_Chart.start_plotting(0.3)

###### GENERATE THE FIGURE ############

     
###### SET THE Initialization #######3

#rt = tl.RepeatedTimer(0.5, update_data, information) # it auto-starts, no need of rt.start()
#information.rt = rt

## Put the information into the functions so that they can access it

    
    ## If I sleep, it blocks !!
#try:
#    sleep(10) # your long-running job goes here...
#finally:
#    rt.stop() # better in a try/finally block to make sure the program ends!



