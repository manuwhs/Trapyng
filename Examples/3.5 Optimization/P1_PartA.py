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

K = 9720
c = 10
d = 100000
p = 12
b = 200000
h = 4
Q = np.sqrt((2*(h+p)*b*d*K)/(h*p*(b-d)))
S = Q*((b*p + h*d)/((p+h)*b))
T = K*d/Q + c*d + p*b*(Q-S)*(Q-S)/(2*Q*(b-d)) + h*(b*S - d*Q)*(b*S - d*Q)/(2*Q*b*(b-d))
print (Q)
print (S)
print (Q-S)
print (S - Q*d/b)
print (T)

## Plot the function
Q_values = range (10000,100000,30)
T_values = []
S_values = []
q_max = []
negative = []

for Q in Q_values:
    S = Q*((b*p + h*d)/((p+h)*b))
    T = K*d/Q + c*d + p*b*(Q-S)*(Q-S)/(2*Q*(b-d)) + h*(b*S - d*Q)*(b*S - d*Q)/(2*b*Q*(b-d))
    T_values.append(T)
    q_max.append(S - Q*d/b)
    negative.append(Q-S)
    
gl.init_figure();
ax1 = gl.subplot2grid((1,3), (0,0), rowspan=1, colspan=1)
ax2 = gl.subplot2grid((1,3), (0,1), rowspan=1, colspan=1, sharex = ax1)
ax3 = gl.subplot2grid((1,3), (0,2), rowspan=1, colspan=1, sharex = ax1)


ax1 = gl.plot(Q_values, T_values, ax = ax1, lw = 3, labels = ["Total Anual Cost", "Q","T(Q)"], legend = ["T(Q)"],
        AxesStyle = "Normal ", color = "k")

ax2 = gl.plot(Q_values, q_max, ax = ax2, lw = 3, labels = ["Maximum inventory", "Q","qmax(Q)"], legend = ["qmax(Q)"],
        AxesStyle = "Normal ", color = "k")

ax3 = gl.plot(Q_values, negative, ax = ax3, lw = 3, labels = ["Maximum Shortage", "Q","(Q-S)(Q)"], legend = ["(Q-S)(Q)"],
        AxesStyle = "Normal ", color = "k")



gl.set_fontSizes(ax = [ax1,ax2,ax3], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 10, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P1_PardA.png",  dpi = 100, sizeInches = [25, 7], close = False, bbox_inches = "tight")



## Limit for Q5
K = 9720
c = 9
d = 100000
p = 12
b = 200000
h = 4
Q = 37500
S = Q*((b*p + h*d)/((p+h)*b))
T = K*d/Q + c*d + p*b*(Q-S)*(Q-S)/(2*Q*(b-d)) + h*(b*S - d*Q)*(b*S - d*Q)/(2*b*Q*(b-d))
print (Q)
print (S)
print (Q-S)
print (S - Q*d/b)
print (T)



## Plotting 
K = 9720
c = 10
d = 100000
p = 12
b = 200000
h = 4

## Plot the function
Q_values = range (10000,100000,300)
T_values = []
S_values = []
q_max = []
negative = []

for Q in Q_values:
    if (Q < 37500):
        c = 10
    else:
        c = 9
    S = Q*((b*p + h*d)/((p+h)*b))
    T = K*d/Q + c*d + p*b*(Q-S)*(Q-S)/(2*Q*(b-d)) + h*(b*S - d*Q)*(b*S - d*Q)/(2*b*Q*(b-d))
    T_values.append(T)

    
gl.init_figure();
ax1 = gl.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)


ax1 = gl.plot(Q_values, T_values, ax = ax1, lw = 3, labels = ["Total Anual Cost", "Q","T(Q)"], legend = ["T(Q)"],
        AxesStyle = "Normal ", color = "k")


gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 10, xticks = 12, yticks = 12)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)

gl.savefig("P1_PardA2.png",  dpi = 100, sizeInches = [25, 7], close = False, bbox_inches = "tight")