import pickle
import gc
import os

# Library for  loading and storing big amounts of data into
# different files because pickle takes a lot of RAM otherwise.
# If the number of partitions = 1, then it just loads like a
# regular pickle file.
# It uses gc also to remove garbage variables. 

def store_pickle (filename, li, partitions = 1, verbose = 1):
    gc.collect()
    splitted = filename.split(".")
    if (len(splitted) == 1):  # If there was no extension
        fname = filename
        fext = ""
    else:
        fname = '.'.join(splitted[:-1])   # Name of the file
        fext = "." + splitted[-1]    # Extension of the file
    
    # li: List of variables to save.
    # It saves the variables of the list in "partitions" files.
    # This function stores the list li into a number of files equal to "partitions" in pickle format
    # If "partitions" = 1 then it is a regular load and store
    num = int(len(li)/partitions);
    
    if (partitions == 1):  # Only 1 partition
        if (verbose == 1):
            print ("Creating file: " + fname + fext)
        with open(fname + fext, 'wb') as f:
            pickle.dump(li, f)    
    else:                 # Several partitions
        for i in range(partitions - 1):
            if (verbose == 1):
                print ("Creating file: " + fname + str(i) + fext)
            with open(fname + str(i)+ fext, 'wb') as f:
                pickle.dump(li[i*num:(i+1)*num], f)    
                # We dump only a subset of the list
        # Last partition to create
        if (verbose == 1):
            print ("Creating file: " + fname + str(partitions -1) + fext)
        
        with open(fname + str(partitions - 1)+ fext, 'wb') as f:
                pickle.dump(li[num*(partitions - 1):], f)    
                # We dump the last subset.
    gc.collect()
    
def load_pickle (filename, partitions = 1, verbose = 0):

    gc.collect()
    total_list = []
    splitted = filename.split(".")
    if (len(splitted) == 1):  # If there was no extension
        fname = filename
        fext = ""
    else:
        fname = '.'.join(splitted[:-1])   # Name of the file
        fext = "." + splitted[-1]    # Extension of the file
    
    if (partitions == 1):  # Only 1 partition
        if (verbose == 1):
            print ("Loading file: " + fname + fext)
        
        if (os.path.exists(fname + fext) == True):   # Check if file exists !!
    
            with open(fname + fext, 'rb') as f:
                total_list = pickle.load(f)    # We read the pickle file  
        else:
            print ("File does not exist: " + fname + fext)
            return []
    else:                         # Several partitions
        for i in range(partitions):
            if (verbose == 1):
                print ("Loading file: " + fname + str(i)+ fext)
            
            if (os.path.exists(fname + str(i)+ fext) == True):   # Check if file exists !!
    
                with open(fname + str(i)+ fext, 'rb') as f:
                    part = pickle.load(f)    # We read the pickle file  
                total_list.extend(part)
            
            else:
                print ("File does not exist: " + fname + str(i)+ fext)
                return []
            
    gc.collect()
    return total_list

#n = 3
#lista = [10, 23, 43, 65, 34, 98, 90, 84, 98]
#store_pickle("lista",lista,n)
#lista2 = load_pickle("lista",n)
