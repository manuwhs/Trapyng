//+------------------------------------------------------------------+
//|                                   All_data_to_csv_downloader.mq5 |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

/*

This code downloads the 

*/

#include <broker_to_csv.mqh>

void OnStart(){
   int i,p;  // Auxiliary index variables 
   // List that will be changed in size to store the 
   string Symbols_names[];  //We cannot declare just a pointer so we will reserve minimum memory and then use  ArrayResize()
   string symbols_file_name = "Symbol_info.csv";
   // Bool to only get the symbols in the watch list or all of them actually.
   bool selected = false;
   // start download date:
   int day = 1; int month = 1; int year = 2005;
   // Set the periods we want to download
   ENUM_TIMEFRAMES periods[] = {PERIOD_M5, PERIOD_M15,PERIOD_H1, PERIOD_D1}; //{PERIOD_M1, PERIOD_M5, PERIOD_M15,PERIOD_H1, PERIOD_D1};
   
   
   //***************************************
   //********* SAVE The list of symbols **********
   //***************************************
   
   symbols_info_to_csv();
   
   // #############################################################################
   // Print the HOLCV values for all symbols and periods
   // #############################################################################
   // Get the number of all symbols we have access to. If true, only of those in the watchlist
   int Nsym = SymbolsTotal(selected);  
   // Save the names of the symbols in the array Symbols_names
   ArrayResize(Symbols_names,Nsym,Nsym);
   for (i = 0; i < Nsym; i++) {
      Symbols_names[i] =  SymbolName(i, selected);
   }
   
   
   for (p = 0; p < ArraySize(periods); p++) {
      for (i = 0; i < Nsym; i++) {
        // For each symbol and period download 
         timeSeries_to_csv(Symbols_names[i],periods[p],day,month,year); // day, month, year 
       }
   }
   
   
   // timeSeries_to_csv("EURUSD",PERIOD_H1,day,month,year);
   
   return;// script completed
   
  }
//+------------------------------------------------------------------+
