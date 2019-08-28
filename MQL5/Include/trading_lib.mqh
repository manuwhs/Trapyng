//+------------------------------------------------------------------+
//|                                                  trading_lib.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

 #include <Trade\AccountInfo.mqh>
 #include<Trade\SymbolInfo.mqh>
 #include<Trade\Trade.mqh>
 #include <Trade\PositionInfo.mqh>
 //********************************************************************************************
 //********************* Information obtaining func ***********************
 

// **************************** TRADING OPERATION *********************
bool open_trade(
   string BUYSELL = "BUY",
   double volume=0.1,         // specify a trade operation volume
   string symbol="GBPUSD",    //specify the symbol, for which the operation is performed
   
   double SL_pct = 0,           // Stop loss and take profit in percentage of the current price?
   double TP_pct = 0,           // Magic number !!  
   int MagicNumber = 100
   ){
    //--- object for performing trade operations
   CTrade  trade;//+------------------------------------------------------------------+
   //--- Get the properties of the symbol
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);             // current price for closing LONG
      
   /// ******************* Stop Loss and Take profit ****************
   double SL;
   double TP;

   if (BUYSELL == "BUY"){
       SL = bid*(1- SL_pct/100);
       TP = bid*(1 + TP_pct/100);
   }
   else{
       SL = bid*(1+ SL_pct/100);
       TP = bid*(1 - TP_pct/100);
   }
   
   //--- Make sure we provide the correct number of digits
   SL=NormalizeDouble(SL,digits);                              // normalizing Stop Loss
   TP=NormalizeDouble(TP,digits);                              // normalizing Take Profit
   
   if (SL_pct == 0) SL = 0.0;
   if (TP_pct == 0) TP = 0.0;
      
   double open_price;
//--- receive the current open price for the position
   if (BUYSELL == "BUY"){
       open_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
   }
   else{
      open_price=SymbolInfoDouble(symbol,SYMBOL_BID);
   }
   
   string comment=StringFormat("%s %s %G lots at %s, SL=%s TP=%s",
                               BUYSELL, symbol,volume,
                                DoubleToString(open_price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
                               
   trade.SetExpertMagicNumber(MagicNumber);
   printf("******* OPENINING TRADE ************");
   printf(comment);
   
   bool trade_result;
   if (BUYSELL == "BUY"){
       trade_result=trade.Buy(volume,symbol,open_price,SL,TP,comment);
   }
   else{
      trade_result=trade.Sell(volume,symbol,open_price,SL,TP,comment);
   }
   
   
   if(trade_result == false)
     {
      //--- failure message
      Print(BUYSELL,"() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print(BUYSELL,"() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
     return true;
}