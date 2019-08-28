//+------------------------------------------------------------------+
//|                                               socketclientEA.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

 #include <Trade\AccountInfo.mqh>
 #include<Trade\SymbolInfo.mqh>
 #include<Trade\Trade.mqh>
 
 #include <Trade\PositionInfo.mqh>
 #include <print_info.mqh>
sinput int lrlenght = 150;
int socket;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
 socket=SocketCreate();
 print_account_info();
 print_Symbol_info(_Symbol);
 set_trading_main_options();
 print_open_positions();
 open_trade("SELL",0.01,"USDCAD",0,0,123);
 
 return(INIT_SUCCEEDED); }
 
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
 SocketClose(socket); }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Only enter if we have a new bar
   string symbol = _Symbol;
   ENUM_TIMEFRAMES period = PERIOD_CURRENT;
   if (isNewBar(symbol = symbol, period = period) == false){
      return;
   }
   
   socket=SocketCreate();
   int port = 9093;
  
   string command = "PLOT_LINE"; // "BUY";
   
   if(socket!=INVALID_HANDLE) {
      if(SocketConnect(socket,"localhost",port,1000)) {
         Print("Connected to "," localhost",":",port);
         
          if (command == "PLOT_LINE"){
            PLOT_LINE_COMMAND();
          }
          else if (command == "BUY"){
           BUY_COMMAND();
          }
      else {
         Print("Connection ","localhost",":",port," error ",GetLastError());
      }
    }
    SocketClose(socket); 
  }
 else {
   Print("Socket creation error ",GetLastError()); 
   }
}
 
