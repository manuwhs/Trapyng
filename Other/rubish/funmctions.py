
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as ColCon

w = 10  # Width of the images
h = 6   # Height of the images

def load_dataset(file_dir = "./dataprices.csv"):
    # Reads the dataprices
    data = pd.read_csv(file_dir, sep = ',') # header = None, names = None  dtype = {'phone':int}
    Nsamples, Ndim = data.shape   # Get the number of bits and attr

    return data
    
def get_Return(price_sequence):
#    price_sequence = np.array(price_sequence)
#    shape = price_sequence.shape
#    
#    for i in range (shape[1]):
    R = (price_sequence[1:] - price_sequence[0:-1])/ price_sequence[0:-1]
    return R
    
def get_SharpR(Returns, axis = 0):
    E_Return = np.mean(Returns,axis)
    std_Return = np.std(Returns,axis)
    SR = E_Return/std_Return
    return SR

def get_SortinoR(Returns, axis = 0):
    E_Return = np.mean(Returns,axis)
    
    Positive_Returns = Returns[np.where(Returns < 0)]
    
    std_Return = np.std(Positive_Returns,axis)
    
    SR = E_Return/std_Return
    return SR
    

    



    