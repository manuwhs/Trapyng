from graph_lib import gl
import numpy as np

X = np.array(range(0,100))
Y = np.sin(1.5*(2*np.pi/X.size)* X)

X2 = np.array(range(0,50))
Y2 = (-15 + np.array(range(0,50)))/25.0

gl.close("all")
type_graph = 2

folder_images = "../pics/gl/"
dpi = 100
sizeInches = [2*8, 2*3]

# Subplot Type 1
if (type_graph == 1):
    gl.set_subplots(nr = 1, nc = 2)
    gl.plot(X,Y, nf = 1,
            color = "k", lw = 5, alpha = 0.7,
            labels = ["Sine chart","Time (s)", "Voltage(V)"],
            legend = ["Rolling measurement"])
            
    gl.stem(X2,Y2, nf = 1,
            color = "k", lw = 2, alpha = 0.7,
            labels = ["Discrete window","Sample (k)", "Amplitud"],
            legend = ["Window values"])
            
    gl.savefig(folder_images +'subplot1.png', 
               dpi = dpi, sizeInches = sizeInches)
# Subplot Type 2
if (type_graph == 2):
    ax1 = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=3)
    gl.plot(X,Y, nf = 0,
            color = "k", lw = 5, alpha = 0.7,
            labels = ["Sine chart","Time (s)", "Voltage(V)"],
            legend = ["Rolling measurement"])
#        
    ax2 = gl.subplot2grid((1,4), (0,3), rowspan=1, colspan=1)
    gl.plot(X2,Y2, nf = 0,
            color = "k", lw = 2, alpha = 0.7,
            labels = ["Discrete window","Sample (k)", "Amplitud"],
            legend = ["Window values"])
    gl.subplots_adjust(left=.1, bottom=.1, right=.90, top=.95, 
                   wspace=.40, hspace=5)      
    gl.savefig(folder_images +'subplot2.png', 
               dpi = dpi, sizeInches = sizeInches)
# Subplot Type 3
if (type_graph == 3):
    ax1 = gl.create_axes(position = [0.2, 0.2, 0.6, 0.4])
    gl.plot(X,Y, nf = 0,
            color = "k", lw = 5, alpha = 0.7,
            labels = ["Sine chart","Time (s)", "Voltage(V)"],
            legend = ["Rolling measurement"])
        
    ax2 = gl.create_axes( position = [0.3, 0.5, 0.3, 0.3])
    gl.stem(X2,Y2, nf = 0,
            color = "k", lw = 2, alpha = 0.7,
            labels = ["Discrete window","Sample (k)", "Amplitud"],
            legend = ["Window values"])

    gl.savefig(folder_images +'subplot3.png', 
               dpi = dpi, sizeInches = sizeInches)
               
# This is only for subplots :)
gl.subplots_adjust(left=.1, bottom=.1, right=.90, top=.95, 
                   wspace=.40, hspace=5)

