class CEntrySignal:
    # This class is the one that characterizes the event of getting into the market
    # Triggered by a certain class

    def __init__(self, EntrySignalID, StrategyID, 
                 datetime, symbolID, BUYSELL):
        # Identify the event !
        self.StrategyID = StrategyID  # ID of the strategy that generated the signal
        self.EntrySignalID = EntrySignalID # ID of the the entry signal
        self.datetime = datetime           # Time when the signal was triggered
        
        # Identify the action to take !
        self.symbolID = symbolID
        self.BUYSELL = BUYSELL  # Binary "BUY" or "SELL"
        
        # Additional information
        self.priority = None          # Default priority
        self.recommendedPosition = None   # How much does the strategy recommend to buysell
        self.tradingStyle = None          # In which timeFrame are operating basically. Scalping, daytrading, long....
        
        self.comments = ""              # Comments for the trader
        
    def set_recommendations(self, priority = None, recommendedPosition = None,
                            tradingStyle = None):
        
        if(type(priority) != type(None)):
            self.priority = priority
        if(type(recommendedPosition) != type(None)):
            self.recommendedPosition = recommendedPosition
        if(type(priority) != type(None)):
            self.tradingStyle = tradingStyle
            