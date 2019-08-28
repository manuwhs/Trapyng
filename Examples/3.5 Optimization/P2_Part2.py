#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 01:01:15 2018

@author: montoya
"""



import numpy as np

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

import matplotlib.pyplot as plt
plt.close("all")
# Own graphical library
from graph_lib import gl
import scipy


import numpy as np

def get_next_lambda(lambdai):
    lambdai = lambdai - np.array([0,m*g])
    return lambdai

def get_decision(lambda_next ):
    u = l* lambda_next[0]/(np.sqrt(lambda_next[0]*lambda_next[0] + (0.5*m*g + lambda_next[1])* (0.5*m*g + lambda_next[1])))
    v = l* (lambda_next[1] + 0.5*m*g)/(np.sqrt(lambda_next[0]*lambda_next[0] + (0.5*m*g + lambda_next[1])* (0.5*m*g + lambda_next[1])))
    
    if (u < 0):
        u = -u
        v = -v
    print (u,v)
    return np.array([u,v])

def get_final_position (lambda0, x0):
    for i in range(N):
        lambda0 = get_next_lambda(lambda0)
        decision = get_decision(lambda0)
        x0 += decision
    x_n = x0
    return x_n

def get_xy_chain_values(lambda0):
    x_values = [0]
    y_values =[0]
    for i in range(N):
        lambda0 = get_next_lambda(lambda0)
        decision = get_decision(lambda0)
        x_values.append(decision[0]+ x_values[-1])
        y_values.append(decision[1]+ y_values[-1])

    return x_values, y_values

    
def print_chain(lambda_0):
    x_values, y_values = get_xy_chain_values(lambda_0)
    lambda_0 = get_costate_value(lambda_0,0) 
    ax1 = gl.plot(x_values, y_values, lw = 3, labels = [" Suspended chains for different number of elements N (Pontryagins)", "z","y"], 
                  legend = ["N = %i"%(N) + ". $\lambda_0 = [%.3f,%.3f]$"%(lambda_0[0],lambda_0[1]) ],
            AxesStyle = "Normal ")

    
    return ax1
def get_error(lambda_0):
    x0 = np.array([0.0,0.0])
    x_N = np.array([h,0.0])
    x_n =get_final_position(lambda_0, x0)
    
    error = (x_N - x_n)
    return error

def get_costate_value(lambda0,i):
    for j in range(i):
        
        lambda0 = get_next_lambda(lambda0)

    return lambda0


h = 6 # Horizontal end point constraint
L = 10 # Total lenght of the chain
M = 14 # Mass of the chain
g = 9.8

"""
################################   QUESTION 3 ##############################
"""
"""
SOLVING AND PLOTTING
"""
gl.init_figure()

N =2
m = float(M)/N
l = float(L)/N

## Initial_geuss
lambda0 = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , m*g])
 
N_values = [2,6,100]

for N in N_values:

    m = float(M)/N
    l = float(L)/N


#        nu = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , -m*g ])
    lambda0_opt = scipy.optimize.fsolve(get_error, lambda0)
#        nu_values = nu
    ax1 = print_chain(lambda0_opt)

    print (" For N=%i: nu_guess = "%N,lambda0,", nu_final: ", lambda0_opt)
    
    print("Costate vector: ",get_costate_value(lambda0_opt,0) )
    
gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 15, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P2_3.png",  dpi = 100, sizeInches = [12, 7], close = False, bbox_inches = "tight")

    
    
    
    
    
    
    
    
    
    
