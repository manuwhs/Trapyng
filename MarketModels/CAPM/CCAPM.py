
import pandas as pd
import numpy as np
import datetime as dt
import CAPM_core as CAco
import CAPM_eff as CAef
import CAPM_ab as CAab
import CAPM_IFE as CAIFE
import CAPM_BL as CABL
class CAPM:
    # THE CAPM class takes a portfolio as input and operates.
    ## Can we maxime the return subject to that the total beta is 0 ?
    # Or minimizing beta ? Can we do that ? 

    def __init__(self, portfolio = [], period = -1, allocation = []):
        if (portfolio != []):
            self.initVariablesPorfolio(portfolio)
            self.period = period  # Period of the portfolio where we work
            self.set_allocation(allocation)
            
    set_Rf =  CAco.set_Rf
    set_allocation = CAco.set_allocation
    get_allocation = CAco.get_allocation
    initVariablesPorfolio = CAco.initVariablesPorfolio
    get_dates = CAco.get_dates
    set_seriesNames = CAco.set_seriesNames
    set_interval = CAco.set_interval
    get_SymbolReturn = CAco.get_SymbolReturn
    ############################################################
    ################### Alpha Beta #############################
    set_index = CAab.set_index
    get_symbol_ab = CAab.get_symbol_ab
    get_portfolio_ab = CAab.get_portfolio_ab
    get_all_symbols_ab = CAab.get_all_symbols_ab
    test_symbol_ab = CAab.test_symbol_ab
    get_portfolio_JensenAlpha = CAab.get_portfolio_JensenAlpha
    test_Jensens_Alpha = CAab.test_Jensens_Alpha
    get_indexReturns = CAab.get_indexReturns
    marketTiming = CAab.marketTiming
    plot_corrab = CAab.plot_corrab
    plot_portfoliocorrab = CAab.plot_portfoliocorrab
    get_indexMeanReturn = CAab.get_indexMeanReturn
    get_symbol_JensenAlpha = CAab.get_symbol_JensenAlpha
    ############################################################
    ############ GET BASIC PARAMETERS ########################
    ############################################################
    get_Returns = CAco.get_Returns
    get_MeanReturns = CAco.get_MeanReturns
    get_covMatrix = CAco.get_covMatrix
    get_corMatrix = CAco.get_corMatrix
    
    get_metrics = CAco.get_metrics
    get_SharpR = CAco.get_SharpR
    
    get_PortfolioReturn = CAco.get_PortfolioReturn
    get_PortfolioStd = CAco.get_PortfolioStd
    get_PortfolioMeanReturn = CAco.get_PortfolioMeanReturn
    simulate_Portfolio = CAco.simulate_Portfolio
    
    compute_allocations = CAco.compute_allocations
    yearly_covMatrix = CAco.yearly_covMatrix
    yearly_Return = CAco.yearly_Return
    #########################################################
    ## Plot Some Graphs
    
    plot_retCorr = CAco.plot_retCorr
    get_random_allocations = CAco.get_random_allocations
    scatter_allocations = CAco.scatter_allocations
    plot_allocations = CAco.plot_allocations
    ############################################################
    ############ Portfolio Optimization ########################
    ############################################################

    randomly_optimize_Portfolio = CAef.randomly_optimize_Portfolio
    efficient_frontier = CAef.efficient_frontier

    TangentPortfolio = CAef.TangentPortfolio
    TangenPortfolioFrontier = CAef.TangenPortfolioFrontier
    TangenPortfolioFrontier2 = CAef.TangenPortfolioFrontier2
    Market_line = CAef.Market_line
    
    ##############################################################
    ################# Exercises of IFE  ##################
    #####################################################
    BlackLitterman = CABL.BlackLitterman
    
    ##############################################################
    ################# Exercises of IFE  ##################
    #####################################################
    IFE_a = CAIFE.IFE_a
    IFE_b = CAIFE.IFE_b
    IFE_c = CAIFE.IFE_c
    IFE_d = CAIFE.IFE_d
    IFE_e = CAIFE.IFE_e
    IFE_f = CAIFE.IFE_f
    IFE_f2 = CAIFE.IFE_f2  # Crazy shit
    IFE_g = CAIFE.IFE_g
    IFE_h = CAIFE.IFE_h
    IFE_i = CAIFE.IFE_i
    IFE_2a = CAIFE.IFE_2a
    IFE_2b = CAIFE.IFE_2b
    IFE_2c = CAIFE.IFE_2c