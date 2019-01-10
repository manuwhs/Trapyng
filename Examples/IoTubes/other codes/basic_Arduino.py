import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)
for i in range(10):
    print (ser.readline())
ser.close()


#    ser = serial.Serial()
#    ser.baudrate = 19200
#    ser.port = 'COM1'
#    
#    ser.open()
#    ser.is_open
#
#    ser.close()
#    ser.is_open
