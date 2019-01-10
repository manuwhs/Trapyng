"""
In this document we will generate a set of fake data and plot it every second
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.chdir("../../")
import import_folders

from graph_lib import gl
import numpy as np
import tasks_lib as tl
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import serial

gl.close("all")
plt.ion()


######### FUNCTION ############
def update_data(information):
    
    time,data = information.time, information.data
    
    ## Read data to update !!
    information.serial.flush()
    data.append(float(information.serial.readline().decode("utf-8").split("\n")[0]))
    time.append(update_data.index)
    
    update_data.index += 1;
    
    window = 100
    
    start = max([update_data.index - window, 0])
    print (start, data[-1])
    
    # option 2, remove all lines and collections
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    
    gl.plot(np.array(time)[start:update_data.index], np.array(data)[start:update_data.index], 
            labels = ["Sensors values", "time (s)", "Temperature"], color = "k", ax = data_axes);
    gl.set_zoom(xlimPad = [0.2,0.2] ,ylimPad = [0.1, 0.1])
#    if (update_data.index == 1000):
#        rt.stop()
#        information.serial.close()
    
def stop_reading_data(event):
    """
    This function will unable the perdiodict update of the chart
    
    """
    ## HACK
    ## We just get the information from itself 
    
    information = stop_reading_data.information
    
    if (type(information.rt) != type(None)):
        information.rt.stop()
    information.rt = None

def restart_reading_data(event):
    information = restart_reading_data.information
    
    if (type(information.rt) == type(None)):
        information.rt = tl.RepeatedTimer(1, information.update_data, information)

def save_to_disk(event):
    time,data = information.time, information.data
    
    df = pd.DataFrame(
    {'Time': time,
     'Data': data,

    });

    df.to_csv('./out.csv', sep=',')
        
###### SET THE READING ########
ser = serial.Serial('/dev/ttyUSB0', 9600)
ser.readline()
for i in range(10):
    print (float(ser.readline().decode("utf-8").split("\n")[0]))
#ser.close()


###### GENERATE FAKE DATA ############
data = np.random.randn(100,1) + 35
time = range(data.size)

###### GENERATE THE FIGURE ############
fig = gl.init_figure();
data_axes = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=3)


data = []
time = []

update_data.index = 0

print ("starting...")
## Define the class with all the info
class information():
    ## Serial port info
    serial = ser   # Serial port we get the info from
    rt = None
    ## Data information
    time = time
    data = data
    
    ## Plotting information
    figure = fig   # Figure with the graph
    data_axes = data_axes

    ## Functions just in case ##
    update_data = update_data

### Widgets ###
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axsave = plt.axes([0.71, 0.20, 0.1, 0.075])

bnext = Button(axnext, 'Stop')
bnext.on_clicked(stop_reading_data)

bprev = Button(axprev, 'Restart')
bprev.on_clicked(restart_reading_data)

bpsave = Button(axsave, 'Save')
bpsave.on_clicked(save_to_disk)

###### SET THE Initialization #######3
rt = tl.RepeatedTimer(0.5, update_data, information) # it auto-starts, no need of rt.start()
information.rt = rt

## Put the information into the functions so that they can access it

stop_reading_data.information= information
restart_reading_data.information = information
save_to_disk.information = information

    
    ## If I sleep, it blocks !!
#try:
#    sleep(10) # your long-running job goes here...
#finally:
#    rt.stop() # better in a try/finally block to make sure the program ends!
