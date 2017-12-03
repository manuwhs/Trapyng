from trendfollowing.extrema import n_day_highs_for_time_series, n_day_lows_for_time_series

class Breakout:
    def __init__(self, date, price, period):
        self.price = price
        self.date = date
        self.period = period    

class NewHigh(Breakout):
    def __init__(self, date, price, period):
        Breakout.__init__(self, date, price, period)

class NewLow(Breakout):
    def __init__(self, date, price, period):
        Breakout.__init__(self, date, price, period)
    
def get_new_highs(ts, period):
    highs_timeseries = n_day_highs_for_time_series(ts, period, 'adj_close')

    new_highs = []
    for highs in highs_timeseries:
        prev_high = highs[str(period) + '_period_high_adj_close']
        if (prev_high and highs['adj_close'] > prev_high):
            new_highs.append(NewHigh(highs['date'], highs['adj_close'], period))

    return new_highs

def get_new_lows(ts, period):
    lows_timeseries = n_day_lows_for_time_series(ts, period, 'adj_close')

    new_lows = []
    for lows in lows_timeseries:
        prev_low = lows[str(period) + '_period_low_adj_close']
        if (prev_low and lows['adj_close'] < prev_low):
            new_lows.append(NewLow(lows['date'], lows['adj_close'], period))

    return new_lows



    

