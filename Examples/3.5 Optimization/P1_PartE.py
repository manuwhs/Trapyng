# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Own graphical library
from graph_lib import gl

import numpy as np

if (0):
    a = 3; b = 6; c = -6
    x1_plus = (-b+ np.sqrt(b*b - 4*a*c))/(2*a)
    x1_minus = (-b -np.sqrt(b*b - 4*a*c))/(2*a)
    
    print (x1_plus, x1_minus)

"""
PART E 
"""
x_1 = -1
x_2 = -1

f_values = []
t_values = []
x1_values = []
x2_values = []
m1_values = []
m2_values = []
f = 2*x_2 + x_1*x_2 - 4*x_1 - 4*x_2*x_2 - x_1*x_1 + 5

f_values.append(f)
x1_values.append(x_1)
x2_values.append(x_2)


for i in range(10):
    m_1 = x_2 -4 - 2*x_1
    m_2 = 2 + x_1 - 8*x_2
    
    t = (2*m_2 + m_1*x_2 + m_2*x_1 - 4*m_1 - 8*x_2*m_2 - 2*x_1*m_1)/(2*(-m_1*m_2 + 4*m_2*m_2 + m_1*m_1))
    
    x_1 = x_1 + t*m_1
    x_2 = x_2 + t*m_2
    
    f = 2*x_2 + x_1*x_2 - 4*x_1 - 4*x_2*x_2 - x_1*x_1 + 5
    print ("---- Results after Iteration %i ------"%(i))
    print ("m1 = %.4f, m2 = %.4f"%(m_1,m_2))
    print ("x1 = %.4f, x2 = %.4f"%(x_1,x_2))
    print ("t = %.4f"%(t))
    print ("f(x): %.4f"%(f))
    

    f_values.append(f)
    t_values.append(t)
    x1_values.append(x_1)
    x2_values.append(x_2)
    m1_values.append(m_1)
    m2_values.append(m_2)

gl.init_figure();
ax1 = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=1)
ax2 = gl.subplot2grid((1,4), (0,1), rowspan=1, colspan=1, sharex = ax1)
ax3 = gl.subplot2grid((1,4), (0,2), rowspan=1, colspan=1, sharex = ax1)
ax4 = gl.subplot2grid((1,4), (0,3), rowspan=1, colspan=1, sharex = ax1)

ax1 = gl.plot([], f_values, ax = ax1, lw = 3, labels = ["Objective function", "iterations","f(X)"], legend = ["f(x)"],
        AxesStyle = "Normal ", color = "k")

ax2 = gl.plot([], t_values, ax = ax2, lw = 3, labels = ["Optimal learning step", "iterations","t"], legend = ["t"],
        AxesStyle = "Normal ", color = "k")

ax3 = gl.plot([], x1_values, ax = ax3, lw = 3, labels = ["Input variables", "iterations","Variables"], legend = ["x1"],
        AxesStyle = "Normal ", color = "k")
ax3 = gl.plot([], x2_values, ax = ax3, lw = 3, legend = ["x2"],
        AxesStyle = "Normal ", color = "b")

ax4 = gl.plot([], m1_values, ax = ax4, lw = 3, labels = ["Gradients", "iterations","Gradients"], legend = ["m1"],
        AxesStyle = "Normal ", color = "k")
ax4 = gl.plot([], m2_values, ax = ax4, lw = 3, legend = ["m2"],
        AxesStyle = "Normal ", color = "b")


#gl.set_zoom (ax = ax7, xlim = [-3, 3], ylim = [-0.1,2])

# Set final properties and save figure
gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 10, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P1_PardD.png",  dpi = 100, sizeInches = [25, 7], close = False, bbox_inches = "tight")