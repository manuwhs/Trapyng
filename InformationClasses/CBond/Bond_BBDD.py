
import pandas as pd
import numpy as np
import datetime as dt



#################################################
############### Setting functions ###############
#################################################
import urllib
import urllib2
import re

#Created by Addison Euhus, July 9 2013.

#In order to run this program:
# call yearHistory(CUSIP#, year)
# returns a dictionary variable with all of the prices for indicated bond in that year

def URLRequest(url, params, method="GET"):
    if method == "POST":
        return urllib2.Request(url, data=urllib.urlencode(params))
    else:
        return urllib2.Request(url + "?" + urllib.urlencode(params))

def getPrice(cusip, month, day, year):
    data = URLRequest("https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate.htm",{"priceDate.month":str(month),"priceDate.day":str(day),"priceDate.year":str(year),"submit":"Show+Prices"}, method="POST")
    response = urllib2.urlopen(data)
    data_str = response.read()
    if len(data_str) < 20000:
        return ''
    start = data_str.find(str(cusip))
    end = data_str[start:].find("</td>\r\n\t</tr>")
    arr = []
    for m in re.finditer('</td>\r\n\t\t<td>',data_str[start:][:end]):
        arr.append(m.end())
    return data_str[start:][:end][arr[5]:]

calendar = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

def getMonth(cusip,month,year):
    tempdict = {}
    for b in range(month,calendar[month]+1):
        tempdict[str(month)+"/"+str(b)+"/"+str(year)] = getPrice(cusip,month,b,year)
    return tempdict

def cleanCache(data_dict):
    for x in list(data_dict.keys()):
        if data_dict[x] == '':
            del data_dict[x]
    return data_dict  

def yearHistory(cusip, year):
    yearData = {}
    for i in range(1,13):
        yearData.update(getMonth(cusip,i,year))
        yearData = cleanCache(yearData)
    return yearData 
    
caca = yearHistory("912810RG5", 2015)
# returns a dictionary variable with all of the prices for indicated bond in that year
    