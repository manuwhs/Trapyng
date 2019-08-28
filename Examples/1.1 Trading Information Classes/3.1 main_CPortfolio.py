""" BASIC USAGE OF THE CLASS PORTFOLIO """
# Change main directory to the main folder and import folders
### THIS CLASS DOES NOT DO MUCH, A QUICK WAY TO AFFECT ALL THE TIMESERIES
## OF THE SYMBOL IN A CERTAIN MANNER, like loading, or interval.
## It also contains info about the Symbol Itself that might be relevant.
import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt

# Own graphical library
from graph_lib import gl 
# Data Structures Data
import CSymbol as CSy
import CPortfolio as CPfl
# Import functions independent of DataStructure
import utilities_MQL5 as ul5

plt.close("all")
######## SELECT DATASET, SYMBOLS AND PERIODS ########
source = "MQL5" # Hanseatic  FxPro GCI Yahoo
[storage_folder, updates_folder] = ul5.get_foldersData(source = source)
################## Date info ###################
sdate_str = "01-01-2010"; edate_str = "21-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

#### Load the info about the available symbols #####
#Symbol_info = CSy.load_symbols_info(info_folder)
####### SELECT SYMBOLS AND PERIODS #################
symbolIDs =  ["EURUSD","EURGBP","USDCAD"]
periods = [1440]  # 1440  43200
period1 = periods[0]

####### LOAD SYMBOLS AND SET Properties   ###################
# Set the symbols and periods to load
myPortfolio = CPfl.Portfolio("BEST_PF",symbolIDs, periods)   
# Download if needed.
#myPortfolio.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
myPortfolio.set_csv(storage_folder)    # Load the symbols and periods

mySymbol_1 = myPortfolio.get_symbols([symbolIDs[0]])[0] # Obtain one of the symbols
myPortfolio.del_symbols([symbolIDs[2]])

# Create an independent symbol and add it to the portfolio
mySymbol_2 = CSy.CSymbol(symbolIDs[2],periods) 
mySymbol_2.set_csv(storage_folder)
myPortfolio.add_symbols([mySymbol_2])

PF_symbolIDs = myPortfolio.get_symbolIDs() # ['DANSKE.CO', 'GE', 'HPQ']

## SET THINGS FOR ALL OF THEM
myPortfolio.set_interval(sdate,edate)
myPortfolio.set_seriesNames(["Close"])

########################## GETTING SINGALS ##########
# Obtaining some timeSeries
dates_all = myPortfolio.get_dates(period = periods[0])
close_some = myPortfolio.get_timeSeries(symbolIDs = symbolIDs)
return_open_all = myPortfolio.get_timeSeriesReturn(period = periods[0], seriesNames = ["Open"])

# Obtaining some indicators
EMA = myPortfolio.EMA(period = periods[0], n = 13)
ATR = myPortfolio.ATR(symbolIDs = symbolIDs ,n = 10)
MACD = myPortfolio.MACD()


######################## FILLING DATA  ##########################
#print "Filling Data"
#Cartera.fill_data()
#print "Data Filled"
###### WE CAN OPERATE DIRECTLY ON SYMBOLS #################


basic_joint_info = 0 # Obtaining info for the same period
trading_platform_several = 1 # Plot the trading platfom for all of them

###### FUNCTIONS TO OPERATE ON THE SAME PERIOD #############

    
if (trading_platform_several):
    # Plot for the 3 quitites, their Barchart and returns

    symbolIDs_pf = myPortfolio.get_symbolIDs()

#    dataTransform = ["intraday", opentime, closetime]
    folder_images = "../pics/gl/"
    gl.set_subplots(3,2)
     # TODO: Be able to automatize the shareX thing
    axeshare = None

    period = periods[0]
    for i in range(len(symbolIDs_pf)):
        symbolID = symbolIDs_pf[i]
        myTimeData = myPortfolio.get_symbols([symbolID])[0].get_timeData(period)
        returns = myTimeData.get_timeSeriesReturn(["Close"])
        dates = myTimeData.get_dates()
        AxesStyle = " - No xaxis"
        if (i == len(symbolIDs_pf) -1):
            AxesStyle = ""
        if (i == 0):
            title = "Bar Chart. " + str(symbolIDs) + r" . Price ($\$$)"
            title2 = "Return"
        else:
            title = ""
            title2 = ""
        ylabel =   symbolID + " (" + ul.period_dic[myTimeData.period] + ")"
        ax = gl.tradingBarChart(myTimeData,  legend = ["Close price"], color = "k", 
                    nf = 1, sharex = axeshare, labels = [title,"",ylabel], 
                    AxesStyle = "Normal" + AxesStyle)
#                    dataTransform = dataTransform)
        ax = gl.stem(dates, returns,  legend = ["Return"], color = "k", 
                    nf = 1, sharex = axeshare, labels = [title2,"",""], 
                    AxesStyle = "Normal" + AxesStyle + " - No yaxis")
#                    dataTransform = dataTransform)
                    
        axeshare = ax
        
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.10, hspace=0)
    image_name = "differentSymbols.png"
    gl.savefig(folder_images + image_name, 
               dpi = 100, sizeInches = [30, 12])
           
           