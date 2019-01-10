from __future__ import unicode_literals, print_function, division
# -*- coding: utf-8 -*-
"""
"""

"""
######################## LOAD AND PROCESS THE DATA ########################
"""
## Original Numpy data. The last data point is an outlier !!

import numpy as np
import pickle_lib as pkl

from io import open
import glob
import os

import unicodedata
import string

def get_linear_dataset(outlier = False):
    """
    Function that generated the data for the linear regression
    """
    if (outlier):
        X_data_tr =  np.array([1.2,1.8,3.2,3.9,4.7,6.4, 4.4]).reshape(-1,1)
        Y_data_tr =  np.array([0, -1,-2,-3,-4, -5, -9]).reshape(-1,1)
    else:
        X_data_tr =  np.array([1.2,1.8,3.2,3.9,4.7,6.4]).reshape(-1,1)
        Y_data_tr =  np.array([0, -1,-2,-3,-4, -5]).reshape(-1,1)
        
    X_data_val =  np.array([8,9,10,11]).reshape(-1,1)
    Y_data_val =  np.array([-7.1, -7.9,-9.2,-10.1]).reshape(-1,1)
    
    return [X_data_tr, Y_data_tr, X_data_val,Y_data_val]

def get_sinuoid_dataset(Ntrain = 100, Nval = 50, sigma_noise = 0.1, Ncycles = 1):
    
    X_data_tr =  np.array(np.linspace(0,2*np.pi*Ncycles,Ntrain) ).reshape(-1,1) 
    Y_data_tr =  np.array(np.sin(np.linspace(0,2*np.pi*Ncycles,Ntrain)) + \
                          np.random.randn(Ntrain) * sigma_noise).reshape(-1,1) + \
                          X_data_tr/3
    
    X_data_val =  np.array(np.linspace(0,2*np.pi*Ncycles,Nval) ).reshape(-1,1)
    Y_data_val =  np.array(np.sin(np.linspace(0,2*np.pi*Ncycles,Nval)) + \
    + np.random.randn(Nval)* sigma_noise).reshape(-1,1)+ \
          X_data_val/3                 

    return [X_data_tr, Y_data_tr, X_data_val,Y_data_val]

def normalize_data(data_tr, data_val, k = 1.0):
    mean, std = np.mean(data_tr, axis = 0), np.std(data_tr, axis = 0)
#    data_tr = k*(data_tr - mean)/std 
    data_val = k*(data_val - mean)/std 
    return data_val


"""
CLASSIFICATION DATA !!!!
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import  make_circles
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(vector, classes):
    Nclasses = classes.size
    Nsamples = vector.size
    
    one_hot = np.zeros((Nsamples, Nclasses))
    one_hot[np.arange(Nsamples), vector] = 1

    return one_hot

def get_toy_classification_data(n_samples=100, centers=3, n_features=2, type_data = "blobs"):
    # generate 2d classification dataset
    if (type_data == "blobs"):
        X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features)
    elif(type_data == "moons"):
        X, y = make_moons(n_samples=n_samples, noise=0.1)
    elif(type_data == "circles"):
        X, y =  make_circles(n_samples=n_samples, noise=0.05)
    # scatter plot, dots colored by class value
#    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#    colors = {0:'red', 1:'blue', 2:'green'}
#    fig, ax = pyplot.subplots()
#    grouped = df.groupby('label')
#    for key, group in grouped:
#        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#    pyplot.show()
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, stratify = None)
    
    classes = np.unique(y_train)
    
    if(0):
        enc = OneHotEncoder().fit(classes.reshape(-1,1))
        
        y_train = enc.transform(y_train.reshape(-1, 1))
        print (y_test)
        y_test = enc.transform(y_test.reshape(-1, 1))
        print (y_test)
    
    y_train = one_hot_encode(y_train, classes)
    y_test = one_hot_encode(y_test, classes)
    
    return  X_train, y_train, X_test, y_test, classes


"""
Sequential data
"""

import torch

def generate_sine_data(T = 20, L = 1000,  N = 100):
    # It computes several instances of the same sinusouid function, at different lags.
    
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    torch.save(data, open('traindata.pt', 'wb'))
    
    return data


def load_RNN_data_generated(folder_data =  "../data/artificial/"):

    Ndivisions = 10;

    
    X_list = pkl.load_pickle(folder_data +"X_values.pkl",Ndivisions)
    Y_list = pkl.load_pickle(folder_data +"Y_values.pkl",Ndivisions)
    
    num_steps, X_dim = X_list[0].shape
    num_chains = len(X_list)
    
    
    ## Divide in train val and test
    proportion_tr = 0.8
    proportion_val = 0.1
    proportion_tst = 1 -( proportion_val + proportion_tr)
    
    num_tr = 1000
    num_val = 500
    num_tst = 500
    
    train_X = [X_list[i] for i in range(num_tr)]
    train_Y = [Y_list[i] for i in range(num_tr)]
    
    val_X = [X_list[i] for i in range(num_tr, num_tr + num_val)]
    val_Y = [Y_list[i] for i in range(num_tr, num_tr + num_val)]
    
    tst_X = [X_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]
    tst_Y = [Y_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]

    train_X = np.concatenate(train_X,1).T
    return train_X


"""
Dataset for the names prediction !!
"""

# Find letter index from all_letters, e.g. "a" = 0
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def one_hot_encode_characters_line(line):
    array = np.zeros((len(line), 1, n_letters))
    for li, letter in enumerate(line):
        array[li][0][letterToIndex(letter)] = 1
    return array


def load_names_dataset(filepath = '../data/names/*.txt'):
    
    def findFiles(path): return glob.glob(path)
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )
        
#    print(findFiles(filepath))
    
#    print(unicodeToAscii('Ślusàrski'))
    
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]
    
    for filename in findFiles(filepath):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    

    ######################################################################
    # Now we have ``category_lines``, a dictionary mapping each category
    # (language) to a list of lines (names). We also kept track of
    # ``all_categories`` (just a list of languages) and ``n_categories`` for
    # later reference.
    

#    print("Categories: ", all_categories)
    n_categories = len(all_categories)
    
    ## ONE HOT THE LETTERS OF EACH LINE (NAME)
    ## We end up with np.arrays(Length_line, 1, Dimensionality_one_hot)
    # The one is for the batches. 
    
    example_line = category_lines[all_categories[0]][0]
    example_one_hot_encoded = one_hot_encode_characters_line(example_line)
    
    if (0):
        print ("Line: ",example_line)
        print (example_one_hot_encoded)
        print ("Dimensions: ", example_one_hot_encoded.shape )
    
    data = [];
    labels = []
    for i in range(n_categories):
        data.extend([one_hot_encode_characters_line(category_lines[all_categories[i]][j]) for j in range(len(category_lines[all_categories[i]]))])
        labels.extend([i for j in range(len(category_lines[all_categories[i]]))])
    
        X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.25, stratify = None)
    
    
    return all_categories, X_train,y_train, X_test, y_test 


    ######################################################################
    # Turning Names into Tensors
    # --------------------------
    #
    # Now that we have all the names organized, we need to turn them into
    # Tensors to make any use of them.
    #
    # To represent a single letter, we use a "one-hot vector" of size
    # ``<1 x n_letters>``. A one-hot vector is filled with 0s except for a 1
    # at index of the current letter, e.g. ``"b" = <0 1 0 0 0 ...>``.
    #
    # To make a word we join a bunch of those into a 2D matrix
    # ``<line_length x 1 x n_letters>``.
    #
    # That extra 1 dimension is because PyTorch assumes everything is in
    # batches - we're just using a batch size of 1 here.
    #


