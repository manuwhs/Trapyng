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

from scipy.integrate import ode

import numpy as np


# % ------------------------------------------------
def tdp(s,x,parm):
    #% ------------------------------------------------
    #% System model. Determine the (time) derivative of the state vector
    #% given the time, state (x) and the EPC Lagrange multipliers.

#    print (z)
    nu_z = parm[0]
    nu_y = parm[1]
    theta_s= np.arctan( (rho*g*(L-s) + nu_y)  / nu_z);
    dx = np.array([np.cos(theta_s), np.sin(theta_s)])
#    print (dx)
    return dx


#% ------------------------------------------------
def get_error(parm):
    #% ------------------------------------------------
#    % Determine the end point error (err) given the EPC Lagrange multipliers
#    % in parm (and the constants that specifies the problem).

    x_N = np.array([h,0.0])
    s_values, x_s_values =  get_values(parm)
    x_n = x_s_values[-1]
    error = (x_N - x_n)
    
#    print (error)
    return error

def get_values(parm):
    #% ------------------------------------------------
#    % Determine the end point error (err) given the EPC Lagrange multipliers
#    % in parm (and the constants that specifies the problem).
    x0 = np.array([0.0,0.0])
    s_span = np.array([0, L])
    ds = 0.0005
    ode_object = ode(tdp)
    ode_object.set_initial_value(x0, s_span[0]).set_f_params(parm)

    s_values = [s_span[0]]
    x_s_values = [x0]
    
#    print ( ode_object.t)
    while ode_object.successful() and ode_object.t < s_span[1]:
#        print (ode_object.t+ds)
        ode_object.integrate(ode_object.t+ds)
        s_values.append(ode_object.t), x_s_values.append(ode_object.y)
#        print (ode_object.t)
    x_n = x_s_values[-1]
    return s_values,x_s_values
    
    
def print_chain(nu):
    s_values, x_s_values = get_values(nu)
    x_s_values = np.array(x_s_values)
    lambda_0 = get_costate_value(nu_values,0) 
    ax1 = gl.plot(x_s_values[:,0], x_s_values[:,1], lw = 3, labels = ["Continuous version of the suspended chain", "z","y"], 
                  legend = ["$\lambda_0 = [%.3f,%.3f]$"%(lambda_0[0],lambda_0[1]) ],
            AxesStyle = "Normal ")

    
    return ax1

def get_costate_value(nu,s):
    lambda_s = np.array([nu[0], nu[1] +(L-s)*rho*g])
    return lambda_s

def get_Hamiltonian(nu):
    s_values,x_s_values = get_values(nu)
    y_values = np.array(x_s_values)[:,1]
    theta_s= [np.arctan( (rho*g*(L-s)+ nu[1])  / nu[0]) for s in s_values]
    
    dx = np.array([np.cos(theta_s), np.sin(theta_s)])
    
    lambda_s_values = np.array([get_costate_value(nu,s) for s in s_values])
    
    print (lambda_s_values.shape)
    print (dx.shape)
    second_term = np.array([lambda_s_values[i].dot(dx[:,i]) for i in range(len(s_values))])
    print (second_term.shape)
    print (y_values.shape)
    H_s  = rho*g*y_values + second_term
    return H_s,lambda_s_values

h = 6 # Horizontal end point constraint
L = 10 # Total lenght of the chain
M = 14 # Mass of the chain
g = 9.8
rho = float(M)/L

"""
################################   QUESTION 2 ##############################
"""
"""
SOLVING AND PLOTTING
"""
gl.init_figure()
N = 2
m = float(M)/N
l = float(L)/N

## Initial_geuss
nu = np.array([-0.5*m*g/np.sqrt(-1 +4*l*l/(h*h)) , -m*g ])
 

nu_values = scipy.optimize.fsolve(get_error, nu)

nu_values[0] = -np.abs(nu_values[0])

s_values, x_s_values = get_values(nu)

ax1 = print_chain(nu_values)

print (" nu_guess",nu,", nu_final: ", nu_values)

print("Costate vector: ",get_costate_value(nu_values,0) )

gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
              legend = 15, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P2_6.png",  dpi = 100, sizeInches = [12, 7], close = False, bbox_inches = "tight")

H_s,lambda_s_values = get_Hamiltonian(nu_values)

gl.init_figure()

ax1 = gl.plot(np.array(s_values), np.diff(H_s), labels = ["Variation of H", "s", r"$\frac{\partial H}{\partial s}$"])
gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
              legend = 15, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P2_62.png",  dpi = 100, sizeInches = [12, 7], close = False, bbox_inches = "tight")

gl.init_figure()
gl.plot(s_values, lambda_s_values[:,1])