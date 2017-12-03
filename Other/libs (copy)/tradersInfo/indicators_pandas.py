import numpy  
import pandas as pd  
import math as m

import utilities_lib as ul
#### FOR FUTURE ADDING MAYBE!!!
def MA(df, seriesNames = ["Close"], n = 20):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df
    # Library of indp 
#    MA = indp.MA(timeData.get_timeData(),n = 20)["MA_20"].values
#    MA = ul.fnp(MA)
#    print MA.shape, dates.shape
#########################################################
    #########################################################
    ##################################################
    
# The values is already a correct [Nsam, Nsig] numpy matrix
#Moving Average  
def SMA(df, seriesNames = ["Close"], n = 20):  
    SMA = pd.rolling_mean(df[seriesNames], n)
    return SMA.values

#Exponential Moving Average  
def EMA(df, seriesNames = ["Close"], n = 20):  
    EMA = pd.ewma(df[seriesNames], span = n, min_periods = n - 1)
    return EMA.values

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    
    R1 = 2 * PP - df['Low']
    S1 = 2 * PP - df['High']
    
    R2 = PP + df['High'] - df['Low']
    S2 = PP - df['High'] + df['Low']

    R3 = df['High'] + 2 * (PP - df['Low'])  
    S3 = df['Low'] - 2 * (df['High'] - PP)

    PPSR = ul.fnp([PP,R1,S1,R2,S2,R3,S3])
    return PPSR

#Bollinger Bands  
def BBANDS(df, n = 20, seriesNames = ["Close"]):  
    MA = pd.rolling_mean(df[seriesNames], n)
    MSD = pd.rolling_std(df[seriesNames], n) 
    
    BBh = MA + MSD *2
    BBl = MA - MSD *2
    
    ## Different types of BB bands ? TODO
#    b1 = 4 * MSD / MA  
#    b2 = (df[seriesNames] - MA + 2 * MSD) / (4 * MSD)  

    BB = ul.fnp([BBh.values, BBl.values])
    return BB

#Average True Range  
def ATR(df, n = 14):  
    i = 0  
    TR_l = [0]  
    while i < len(df.index) -1:  
        TR = max(df.get_value(df.index[i + 1], 'High'), 
           df.get_value(df.index[i], 'Close')) - min(df.get_value(df.index[i + 1], 'Low'), 
                     df.get_value(df.index[i], 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    
    TR_s = pd.Series(TR_l)  
    ATR = pd.ewma(TR_s, span = n, min_periods = n)
    
    ATR = ul.fnp(ATR.values)
    return ATR
    
#Momentum  
def MOM(df, n = 1, seriesNames = ["Close"]):  
    M = df[seriesNames].diff(n) 
    return M.values

#Rate of Change  
def ROC(df, n = 1, seriesNames = ["Close"]):  
    n = n + 1
    M = df[seriesNames].diff(n - 1).values
    N = df[seriesNames].shift(n - 1).values
#    print M.shape, N.shape
    ROC = M / N
    return ROC

#Stochastic oscillator %K  
def STOK(df):  
    # TODO: This could be inf
    SOk = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    SOk = SOk.fillna(1)
    SOk = 100 * ul.fnp(SOk)
    return SOk

#Stochastic oscillator %D  
def STO(df, n = 14):  
    SOk = STOK(df)
    SOd = pd.ewma(SOk, span = n, min_periods = n - 1)
    SOd = 1 * ul.fnp(SOd)
    return SOd

#Relative Strength Index  
def RSI(df, n = 20):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 < len(df.index):  
        UpMove = df.get_value(df.index[i + 1], 'High') - df.get_value(df.index[i], 'High')  
        DoMove = df.get_value(df.index[i], 'Low') - df.get_value(df.index[i + 1], 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
    RSI = PosDI / (PosDI + NegDI)
    
    RSI = 100 * ul.fnp(RSI)
  
    return RSI
    
#Average Directional Movement Index  
def ADX(df, n = 14, n_ADX = 14):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 < len(df.index):  
        UpMove = df.get_value(df.index[i + 1], 'High') - df.get_value(df.index[i], 'High')  
        DoMove = df.get_value(df.index[i], 'Low') - df.get_value(df.index[i + 1], 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < len(df.index) -1:  
        TR = max(df.get_value(df.index[i + 1], 'High'), 
                 df.get_value(df.index[i], 'Close')) - min(df.get_value(df.index[i + 1], 'Low'), 
                df.get_value(df.index[i], 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1)  
    
    ADX = ul.fnp(ADX)
    return ADX
    
#Accumulation/Distribution  
def ACCDIST(df, n = 14):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    AD = M / N  
    
    AD = ul.fnp(AD.values)
    return AD
    
#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast = 26, n_slow = 12, n_smooth = 9):  
    EMAfast = pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1)
    EMAslow = pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1) 
    
    MACD = EMAfast - EMAslow
    MACDsign = pd.ewma(MACD, span = 9, min_periods = 8)  
    MACDdiff = MACD - MACDsign

    MACD = ul.fnp(MACD)
    MACDsign = ul.fnp(MACDsign)
    MACDdiff = ul.fnp(MACDdiff)
    
    ret = ul.fnp([MACD,MACDsign,MACDdiff])
    return   ret

#Trix  
## Oscillator similar to MACD
def TRIX(df, n = 30):  
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  

    # Get the returns 
    Trix = EX3.pct_change(periods = 1)
    
    Trix = ul.fnp(Trix.values)
    return Trix
    
#Mass Index  
def MassI(df, n = 9):  
    Range = df['High'] - df['Low']  
    EX1 = pd.ewma(Range, span = n, min_periods = n-1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n-1)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df

#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df


#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['Close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df



#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:  
            OBV.append(df.get_value(i + 1, 'Volume'))  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:  
            OBV.append(0)  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:  
            OBV.append(-df.get_value(i + 1, 'Volume'))  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['Close'].diff(int(n * 11 / 10) - 1)  
    N = df['Close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(int(n * 14 / 10) - 1)  
    N = df['Close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['Close'], n), name = 'STD_' + str(n)))  
    return df  