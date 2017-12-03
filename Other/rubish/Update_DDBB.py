import CPortfolio as CPfl
import CPortfolio_DDBB as Pdb

symbols = ["AAPL", "GLD", "FB", "IBM"]
symbols_n = ["ITX", "BKIA", "SAN"]  

Mercados = Pdb.download_symbols(symbols)
Mercados = Pdb.update_symbols(symbols)

#Cartera = CPfl.Portfolio() 
#Cartera.load_symbols(symbols)
