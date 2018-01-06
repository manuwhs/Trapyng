
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

plt.close("all")
from sklearn.model_selection import train_test_split




def get_EM_clusters():
    
    