from graph_lib import gl
import numpy as np

X = np.array(range(0,1000))
Y = np.sin(1.5*(2*np.pi/X.size)* X)

gl.close("all")
gl.plot(X,Y, 
        color = "k", lw = 5, alpha = 0.7,
        labels = ["Sine chart","Time (s)", "Voltage(V)"],
        legend = ["Rolling measurement"],
        xlim = [-100, 1100], ylim = [-1.3,1.5],

        fontsize = 30,  fontsizeL = 30, fontsizeA = 30,
        loc = 1)
        
