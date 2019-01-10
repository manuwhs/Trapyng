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

def get_angle(i, nu):
    theta_i = np.arctan((m*g*(N-0.5-i) + nu[1])/nu[0])
    return theta_i
    
def get_final_position (nu, x0):
    for i in range(N):
        theta_i = get_angle(i,nu)
        x0 += np.array([l*np.cos(theta_i),l*np.sin(theta_i)])
    x_n = x0
    return x_n

def get_xy_chain_values(nu):
    x_values = [0]
    y_values =[0]
    for i in range(N):
        angle_i = get_angle(i,nu)
        x_values.append(l*np.cos(angle_i)+ x_values[-1])
        y_values.append(l*np.sin(angle_i)+ y_values[-1])

    return x_values, y_values

    
def print_chain(nu):
    x_values, y_values = get_xy_chain_values(nu)
    lambda_0 = get_costate_value(nu_values,0) 
    ax1 = gl.plot(x_values, y_values, lw = 3, labels = [" Suspended chains for different number of elements N", "z","y"], 
                  legend = ["N = %i"%(N) + ". $\lambda_0 = [%.3f,%.3f]$"%(lambda_0[0],lambda_0[1]) ],
            AxesStyle = "Normal ")

    
    return ax1
def get_error(nu):
    x0 = np.array([0.0,0.0])
    x_N = np.array([h,0.0])
    x_n =get_final_position(nu, x0)
    
    error = (x_N - x_n)
    return error

def get_costate_value(nu,i):
    lambda_i = np.array([nu[0], nu[1] + (N -i)*m*g])

    return lambda_i


h = 6 # Horizontal end point constraint
L = 10 # Total lenght of the chain
M = 14 # Mass of the chain
g = 9.8

"""
################################   QUESTION 2 ##############################
"""
"""
SOLVING AND PLOTTING
"""
gl.init_figure()

N =2
m = float(M)/N
l = float(L)/N

## Initial_geuss
nu = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , -m*g ])
 
N_values = [2, 6, 100]

for N in N_values:

    m = float(M)/N
    l = float(L)/N

    if (N == 2):
        nu_values = nu
    else:
#        nu = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , -m*g ])
        nu_values = scipy.optimize.fsolve(get_error, nu)
#        nu_values = nu
    nu_values[0] = -np.abs(nu_values[0])
    ax1 = print_chain(nu_values)

    print (" For N=%i: nu_guess = "%N,nu,", nu_final: ", nu_values)
    
    print("Costate vector: ",get_costate_value(nu_values,0) )
    
gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 15, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P2_2.png",  dpi = 100, sizeInches = [12, 7], close = False, bbox_inches = "tight")


"""
################################   QUESTION 5 ##############################
"""

print (" ----------------- QUESTION 5 ----------------")

def get_half_angle(i, nu_z):
    theta_i = np.arctan((m*g*(N/2-0.5-i) )/nu_z)
    return theta_i

def get_half_final_position (nu_z, x0):
    for i in range(int(N/2)):
        theta_i =get_half_angle(i, nu_z)
        x0 += np.array([l*np.cos(theta_i),l*np.sin(theta_i)]).flatten()
    x_n = x0
    return x_n

def get_error_half_problem(nu_z):
    x0 = np.array([0.0,0.0]).flatten()
    x_N = np.array([h,0.0]).flatten()
    x_n =get_half_final_position(nu_z, x0)
    
    error = x_N[0]/2 - x_n[0]
    return error

def get_xy_half_chain_values(nu_z):
    x_values = [0]
    y_values =[0]
    for i in range(int(N/2)):
        angle_i = get_half_angle(i,nu_z)
        x_values.append(l*np.cos(angle_i)+ x_values[-1])
        y_values.append(l*np.sin(angle_i)+ y_values[-1])

    return x_values, y_values

    
def print_half_chain(nu_z):
    x_values, y_values = get_xy_half_chain_values(nu_z)
    lambda_0 = get_costate_value_half(nu_values,0) 
    ax1 = gl.plot(x_values, y_values, lw = 3, labels = [" Half Suspended chains for different number of elements N", "z","y"], 
                  legend = ["N = %i"%(N) + ". $\lambda_0 = [%.3f,%.3f]$"%(lambda_0[0],lambda_0[1]) ],
            AxesStyle = "Normal ")

def get_costate_value_half(nu_z,i):
    lambda_i = np.array([nu_z,  (N/2 -i)*m*g])

    return lambda_i

"""
SOLVING AND PLOTTING
"""
gl.init_figure()

N =2
m = float(M)/N
l = float(L)/N

## Initial_geuss
nu_z = -0.5*m*g/np.sqrt(-1 +4*l*l/(h*h))
 
N_values = [2, 6, 100]

for N in N_values:

    m = float(M)/N
    l = float(L)/N
#        nu = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , -m*g ])
    nu_values = scipy.optimize.fsolve(get_error_half_problem, nu_z)
#        nu_values = nu
    nu_values[0] = -np.abs(nu_values[0])
    ax1 = print_half_chain(nu_values)

    print (" For N=%i: nu_guess = "%N,nu,", nu_final: ", nu_values)
    
    print("Costate vector: ",get_costate_value_half(nu_values,0) )
    
gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 15, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P2_5.png",  dpi = 100, sizeInches = [12, 7], close = False, bbox_inches = "tight")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
