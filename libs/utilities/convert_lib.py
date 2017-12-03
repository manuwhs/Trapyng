#########################################################3
############### CONVERT LIBRARY  ##############################
##########################################################
## Library with function to convert data.
## Initially from .hst to .csv

import struct
from time import sleep
import time
import pandas as pd
import utilities_lib as ul
HEADER_SIZE = 148
OLD_FILE_STRUCTURE_SIZE = 44
NEW_FILE_STRUCTURE_SIZE = 60
 
def process_hst(filedir, filetype  = "new"):
    ## This funciton reads a hst file and createas a .csv !!
    if filedir == None:
        print "Enter a valid filedir (-f)"
        quit()
        
    if filetype != "new" and filetype != "old":
        print "Enter a valid filetype (valid options are old and new)"
        quit()

    ########################################################################
    ################ FORMATTING OF THE NAME ############
    #### WHERE TO STORE THE CSV !!
    ########################################################################
    filename = filedir.split("/")[-1]
    original_filename =  filename
    
    filename = filename.split(".hst")[0]
#    print filename
    ## This is of the type : Alcatel-Luc1440.hst
    ## We detect the period
    periods = ul.periods
    
    flag_detected = 0
    # Check in decreasing order of length
    periods = periods[::-1]
#    periods = periods[:]
    for period in periods[::-1]:
        period_str = str(period)
        per_len = len(period_str)
        if (filename[-per_len:] == period_str):
            flag_detected = 1
            print period
            break
    
    # If we did not detect it
    if (flag_detected != 1):
        return -1
        
    filename = filename[:-per_len] + "_" + ul.period_dic[period] + '.csv'


    read = 0
    openTime = []
    openPrice = []
    lowPrice = []
    highPrice = []
    closePrice = []
    volume = []
 
    with open(filedir, 'rb') as f:
        while True:
            
            if read >= HEADER_SIZE:
            
                if filetype == "old":
                    buf = f.read(OLD_FILE_STRUCTURE_SIZE)
                    read += OLD_FILE_STRUCTURE_SIZE        
                         
                if filetype == "new":
                    buf = f.read(NEW_FILE_STRUCTURE_SIZE)
                    read += NEW_FILE_STRUCTURE_SIZE
                    
                if not buf:
                    break
                    
                if filetype == "old":
                    bar = struct.unpack("<iddddd", buf)
                    openTime.append(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar[0])))
                    openPrice.append(bar[1])
                    highPrice.append(bar[3])
                    lowPrice.append(bar[2])
                    closePrice.append(bar[4])
                    volume.append(bar[5])  
                if filetype == "new":
#                    print filedir
                    bar = struct.unpack("<Qddddqiq", buf)
                    try:
                        openTime.append(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar[0])))
                    except ValueError: 
                        print "Error: " + filedir
                        return -1
                    openPrice.append(bar[1])
                    highPrice.append(bar[2])
                    lowPrice.append(bar[3])
                    closePrice.append(bar[4])
                    volume.append(bar[5])  
                                              
            else:           
                buf = f.read(HEADER_SIZE)
                read += HEADER_SIZE
                
    data = {'Time':openTime, 'Open':openPrice,'High':highPrice,
    'Low':lowPrice,'Close':closePrice,'Volume':volume}
    result = pd.DataFrame.from_dict(data)
    result = result.set_index('Time')
#    print result
    
#    print filename
    ## Filepath
    filepath = filedir.split(original_filename)[0]
    filepath = "/".join(filepath.split("/")[:-2]) + "/CSVS/"
    filepath = filepath + ul.period_dic[period] + "/"
    ul.create_folder_if_needed(filepath)
#    print filepath
    result.to_csv(filepath + filename)    # , header = False
        

