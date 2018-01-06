
import import_folders
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import CSymbol as CSy
import copy as copy
import CColiseum as CCol
import CPortfolio as CPfl
import CMoneyManagement as CMyM
import CStrategy as CStgy

plt.close("all")
PLOT_GRAPHS = 0

#################################################
############# CREATE PORTFOLIO ##################
#################################################

periods = [5,15,60,1440]
Cartera = CPfl.Portfolio(["CBOT.YM","Mad.ITX", "EURUSD", "CBOT.YM"], periods) 
Cartera.load_symbols_csv()

########################################################
############# CREATE Money Management ##################
########################################################

Coliseo = CCol.CColiseum(Portfolio = Cartera)
putoAmo = CMyM.CMoneyManagement(Coliseo)

########################################################
############# CREATE Strategy ##################
########################################################

Hitler = CStgy.CStrategy(Cartera)
crosses, dates = Hitler.XingAverages("CBOT.YM", Ls = 40,Ll = 200)  # 
time_instances = [];  # Time instances when we get the stuff

Ndates = crosses.size



print "--------------------------------------------------"
for i in range(Ndates):
    putoAmo.set_date(dates[i])
    if (crosses[i] == 1):
        print "COMPRRRRA" 
        putoAmo.process_new_actions(["CBOT.YM"],[])
        print putoAmo.Coliseum.Warriors
        
    elif (crosses[i] == -1):
        print "Vende" 
        putoAmo.process_new_actions([],["CBOT.YM"])
        print putoAmo.Coliseum.Warriors

print "--------------------------------------------------"
print "PROFIT: "
print putoAmo.Coliseum.Profit