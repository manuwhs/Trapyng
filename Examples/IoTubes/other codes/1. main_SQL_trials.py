"""
In this document we will generate a set of fake data and plot it every second
"""

import os
import SQL_lib
import datetime as dt
import time
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

import mysql.connector
from mysql.connector import errorcode


class connection_data():
    DB_NAME = 'iotu41656322575'
    DB_HOST = '160.153.153.41'
    DB_PORT = '3312'
    DB_USER = 'iotu41656322575'
    DB_PASSWORD = 'Ezro|5m8pt9j'


DB_NAME = 'iotu41656322575'
DB_HOST = '160.153.153.41'
DB_PORT = '3312'
DB_USER = 'iotu41656322575'
DB_PASSWORD = 'Ezro|5m8pt9j'


######### STBALICH CONNECTION 
cnx = mysql.connector.connect(user=DB_USER, password=DB_PASSWORD,
                              host=DB_HOST, port = DB_PORT,
                              database=DB_NAME)
cursor = cnx.cursor()

if (0):
    query = SQL_lib.create_DDBB("machine_1")
    SQL_lib.excute_query(query,cursor, extra_text = "Creating machine DDBB" )

############## CREATE TABLES ###############

cleaning_id1 = "hello"
drop_and_recreate = 0;
if (drop_and_recreate):
    ## Drop table
    query = SQL_lib.delele_table(table_id = cleaning_id1)
    SQL_lib.excute_query(query,cursor, extra_text = "droppin table" )
    
    ## Create table table
    query = SQL_lib.create_cleaning_table(cleaning_id = cleaning_id1)
    SQL_lib.excute_query(query,cursor, extra_text = "create table" )
        
    ####### CREATE DATA ########
    Nsamples = 60
    data_sensors = np.random.randn(Nsamples,4) + np.array(np.sin(2*np.pi*np.array(range(Nsamples))/(Nsamples/4))*3).reshape(Nsamples,1);
    data_timestamps = []
    base_dt = dt.datetime.now()
    for i in range(Nsamples):
        ts = dt.datetime.now() + dt.timedelta(seconds = i)
        data_timestamps.append(ts)
    
    column_names = ["Temp","PH","Pressure","Conductivity"] #"TS",
    
    df = pd.DataFrame(data_sensors, columns = column_names)
    df["TS"] = data_timestamps
    
    ## Add the data:
    query = SQL_lib.add_cleanning_data(cleaning_id1, df)
    SQL_lib.excute_query(query,cursor, extra_text = "Uploading data" )
    cnx.commit()
#    print query
#time.sleep(1)
#SQL_lib.excute_query(query,cursor, extra_text = "Uploading data" )

### Get the data:
query = SQL_lib.get_cleanning_data(cleaning_id1)
SQL_lib.excute_query(query,cursor, extra_text = " Getting data" )
data = cursor.fetchall()
print "Fetched data"
print data


plt.plot(data_timestamps,data_sensors[:,1])
#SQL_lib.add_cleanning_data(cleaning_id1, data)


#for name, ddl in TABLES.iteritems():
#    try:
#        print("Creating table {}: ".format(name), end='')
#        cursor.execute(ddl)
#    except mysql.connector.Error as err:
#        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
#            print("already exists.")
#        else:
#            print(err.msg)
#    else:
#        print("OK")

cursor.close()
cnx.close()

