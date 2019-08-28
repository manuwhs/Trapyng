//+------------------------------------------------------------------+
//|                                                   print_info.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+


 #include <Trade\AccountInfo.mqh>
 #include<Trade\SymbolInfo.mqh>
 #include<Trade\Trade.mqh>
 #include <Trade\PositionInfo.mqh>

//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int print_account_info()
  {
//--- object for working with the account
   CAccountInfo account;
//--- receiving the account number, the Expert Advisor is launched at
   long login=account.Login();
   Print("Login=",login);
//--- clarifying account type
   ENUM_ACCOUNT_TRADE_MODE account_type=account.TradeMode();
//--- if the account is real, the Expert Advisor is stopped immediately!
   if(account_type==ACCOUNT_TRADE_MODE_REAL)
     {
      MessageBox("Trading on a real account is forbidden, disabling","The Expert Advisor has been launched on a real account!");
      return(-1);
     }
//--- displaying the account type    
   Print("Account type: ",EnumToString(account_type));
//--- clarifying if we can trade on this account
   if(account.TradeAllowed())
      Print("Trading on this account is allowed");
   else
      Print("Trading on this account is forbidden: you may have entered using the Investor password");
//--- clarifying if we can use an Expert Advisor on this account
   if(account.TradeExpert())
      Print("Automated trading on this account is allowed");
   else
      Print("Automated trading using Expert Advisors and scripts on this account is forbidden");
//--- clarifying if the permissible number of orders has been set
   int orders_limit=account.LimitOrders();
   if(orders_limit!=0)Print("Maximum permissible amount of active pending orders: ",orders_limit);
//--- displaying company and server names
   Print(account.Company(),": server ",account.Server());
//--- displaying balance and current profit on the account in the end
   Print("Balance=",account.Balance(),"  Profit=",account.Profit(),"   Equity=",account.Equity());
   Print(__FUNCTION__,"  completed"); //---
   return(0);
  }
  
int print_Symbol_info(string symbol)
  {
//--- object for receiving symbol settings
   CSymbolInfo symbol_info;
//--- set the name for the appropriate symbol
   symbol_info.Name(symbol);
//--- receive current rates and display
   symbol_info.RefreshRates();
   Print(symbol_info.Name()," (",symbol_info.Description(),")",
         "  Bid=",symbol_info.Bid(),"   Ask=",symbol_info.Ask());
//--- receive minimum freeze levels for trade operations
   Print("StopsLevel=",symbol_info.StopsLevel()," pips, FreezeLevel=",
         symbol_info.FreezeLevel()," pips");
//--- receive the number of decimal places and point size
   Print("Digits=",symbol_info.Digits(),
         ", Point=",DoubleToString(symbol_info.Point(),symbol_info.Digits()));
//--- spread info
   Print("SpreadFloat=",symbol_info.SpreadFloat(),", Spread(current)=",
         symbol_info.Spread()," pips");
//--- request order execution type for limitations
   Print("Limitations for trade operations: ",EnumToString(symbol_info.TradeMode()),
         " (",symbol_info.TradeModeDescription(),")");
//--- clarifying trades execution mode
   Print("Trades execution mode: ",EnumToString(symbol_info.TradeExecution()),
         " (",symbol_info.TradeExecutionDescription(),")");
//--- clarifying contracts price calculation method
   Print("Contract price calculation: ",EnumToString(symbol_info.TradeCalcMode()),
         " (",symbol_info.TradeCalcModeDescription(),")");
//--- sizes of contracts
   Print("Standard contract size: ",symbol_info.ContractSize(),
         " (",symbol_info.CurrencyBase(),")");
//--- minimum and maximum volumes in trade operations
   Print("Volume info: LotsMin=",symbol_info.LotsMin(),"  LotsMax=",symbol_info.LotsMax(),
         "  LotsStep=",symbol_info.LotsStep());
//--- 

//-----Marcket openning time
// It does not actually work
   MqlDateTime open, close;
   datetime openDT, closeDT;
   bool sessionTrade = SymbolInfoSessionTrade("EURUSD", MONDAY, 0, openDT, closeDT);
   TimeToStruct(openDT, open);
   TimeToStruct(closeDT, close);
   
   PrintFormat("Opening Time(%d) ", open.hour);
   PrintFormat("Closing Time(%d) ", close.hour);
   Print(__FUNCTION__,"  completed");
//---
   return(0);
  }
  
 int set_trading_main_options(){
    //--- object for performing trade operations
   CTrade  trade;//+------------------------------------------------------------------+

 //--- set MagicNumber for your orders identification
   int MagicNumber=123456;
   trade.SetExpertMagicNumber(MagicNumber);
//--- set available slippage in points when buying/selling
   int deviation=10;
   trade.SetDeviationInPoints(deviation);
//--- order filling mode, the mode allowed by the server should be used
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
//--- logging mode: it would be better not to declare this method at all, the class will set the best mode on its own
   trade.LogLevel(1); 
//--- what function is to be used for trading: true - OrderSendAsync(), false - OrderSend()
//---

   bool AsynMode = true;
   if(AsynMode){
        printf("Asyncronous trading mode. The order function does NOT get blocked until getting server response");
    }
    else{
       printf("Syncronous trading mode. The order function does get blocked until getting server response");
    }
   trade.SetAsyncMode(true);
//---
   return(0);
  }

void print_open_positions(){
   CPositionInfo  m_position;                   // trade position object
   CTrade         m_trade;                      // trading object

   int n_positions=PositionsTotal();
   printf("****** There are %i Open positions ******",n_positions);
   
   for(int i= 0; i < n_positions;i++) // returns the number of current positions
      if(m_position.SelectByIndex(i))     // selects the position by index for further access to its properties
        // printf("%i) %s", i+1,  m_position.Symbol(), m_position.Volume());
         printf("%i) Comment: %s. Ref Magic: %i", i+1, m_position.Comment(), m_position.Magic());
         
        // if(m_position.Symbol()==m_name && m_position.Magic()==m_magic)
        //    m_trade.PositionClose(m_position.Ticket()); // close a position by the specified symbol
  }
 