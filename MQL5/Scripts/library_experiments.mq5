//+------------------------------------------------------------------+
//|                                          library_experiments.mq5 |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
#include <broker_to_csv.mqh>
void OnStart()
  {
//---
   positions_info_to_csv();
   symbols_info_to_csv();
  }
//+------------------------------------------------------------------+
