import os
os.chdir("../../")
import import_folders

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import CTimeData as CTD
import copy as copy
import utilities_lib as ul
from graph_lib import gl
import DDBB_lib as DBl

import Cemail

plt.close("all")

######## SELECT SOURCE ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
folder_images = "../pics/gl/"
######## SELECT SYMBOLS AND PERIODS ########
symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["Amazon", "Alcoa_Inc"]
periods = [15]

######## SELECT DATE LIMITS ###########
sdate = dt.datetime.strptime("21-11-2016", "%d-%m-%Y")
edate = dt.datetime.strptime("25-11-2016", "%d-%m-%Y")
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
TD = DBl.load_TD_from_csv(storage_folder, symbols[1],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_TD(TD)
############## Obtain time series ###########################
price = timeData.get_timeSeries(["Close", "Average"]);
############# Plot time Series and save it to disk #########
gl.plot([],price)

datafolder = "../maildata/"

picdir = datafolder + "pene.png"
gl.savefig(picdir)

###########################################################################
############## BASIC PLOTING FUNC #########################################
###########################################################################

user = "esopo.goldchick@gmail.com"
pwd = "Goldenegg"

#user = "manuwhs@gmail.com"
#pwd = "manumon7g.@"

recipient = "manuwhs@gmail.com"
#recipient = "tsarmarianthi@gmail.com"

subject = "[Trapyng] Update %s" % ("penesd")

body = "Look at this super interesting stuff !!"

myMail = Cemail.Cemail(user,pwd,recipient)
myMail.create_msgRoot(subject = subject)
#myMail.set_subject(subject)  # For some reason we can only initilize the Subject
myMail.add_HTML(body)

## Add some HMTL
fd = open(datafolder + "index.html")
caca = fd.read()
fd.close
myMail.add_HTML(caca)

myMail.add_image(filedir = picdir, inline = 0)
myMail.add_image(filedir = picdir, inline = 1)

myMail.add_file(datafolder + "Email_main.py")
myMail.add_file(datafolder + "main.pdf")

myMail.add_HTML("<h1> Fuck you </h1> <br>")
myMail.add_HTML("Hello my friend")
########## YOU MAY HAVE TO ACTIVATE THE USED OF UNTRUSTFUL APPS IN GMAIL #####
myMail.send_email()
