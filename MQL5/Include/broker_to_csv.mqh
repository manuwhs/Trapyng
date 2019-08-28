//+------------------------------------------------------------------+
//|                                                broker_to_csv.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |


//| timeSeries_to_csv                                                        |

#include <util.mqh>
#include<Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include<Trade\Trade.mqh>
  
int timeSeries_to_csv(string symbol_name, ENUM_TIMEFRAMES periodo, int Day, int Month,int Year, bool force_showing_of_data_to_download = true){

/*
This funciton downloads the OCHLV data for a given symbol and timeperiod and writes it to csv.
The period downloaded goes from the date specified as input and the current time.
By using force_showing_of_data_to_download = true; the algorithm opens the corresponding chart in MT5 and scrolls back a lot (fixed) 
to have higher chances that MT5 makes accessible the wanted data.


Beware that:
- Metatrader will not have the data for 1M for more than 2 weeks usually. (Depends on the broker)
- There are missing ticks many times.
- Sometimes this does not download the data because of broker blocking or broker bugs.
*/

   string FileName="";
   int copied=0;
   int num_downloaded;
   int FileHandle=0;
   
//--- The created file name has name, (Symbol+Period) /M1/EURUSD_M1.csv

   string periodo_txt = fTimeFrameName(periodo);
   
   Print(periodo, " - ",periodo_txt);
   
   FileName = periodo_txt + "/" + symbol_name +"_"+periodo_txt+"" + ".csv";

   MqlRates rates[];   
   MqlDateTime tm;  // Time structure of mql
   ArraySetAsSeries(rates,true);

  // Create the proper time specification
   string   start_time=IntegerToString(Year)+"."+IntegerToString(Month,2,'0')+"."+IntegerToString(Day,2,'0'); 
   // Example but not used.
  // string   end_time=IntegerToString(2015)+"."+IntegerToString(1,2,'0')+"."+IntegerToString(22,2,'0'); 

   // Logging
   printf("Downloading %s_%s ",symbol_name,periodo_txt );
   
   // ****************************************
   // ******** INFORMATION OBTAINING  ********
   // ****************************************
   
   ResetLastError();

   long chart_id;
   chart_id = ChartOpen(symbol_name,periodo);
   if (force_showing_of_data_to_download){
   
      printf("Backwards scrolling");
      
       // Navigate thourgh the window
      //--- disable auto scroll
      ChartSetInteger(chart_id,CHART_AUTOSCROLL,false);
      //--- set a shift from the right chart border
      ChartSetInteger(chart_id,CHART_SHIFT,true);
      //--- draw candlesticks
      ChartSetInteger(chart_id,CHART_MODE,CHART_CANDLES);
      //--- set the display mode for tick volumes
      ChartSetInteger(chart_id,CHART_SHOW_VOLUMES,CHART_VOLUME_TICK);
      
      int max_scroll = 500;
      int jj = 0;
      int dx = 100;
      
      bool moving = ChartNavigate(chart_id,CHART_END,0);
      while(jj <max_scroll){
      //--- scroll 10 bars to the right of the history start
         // ChartNavigate(chart_id,CHART_BEGIN,-300);  // CHART_BEGIN
         moving = ChartNavigate(chart_id,CHART_CURRENT_POS,-dx);
         if (!moving){
            Print("Navigate failed. Error = ",GetLastError());
            break;
         }
         ChartRedraw();
         Sleep(1);
         jj += 1;
    }
    ChartRedraw();
   
    Print("Total available bars ", symbol_name, " is ", iBars(symbol_name,periodo));
    }
  
   // Now we will use CopyRates until we download the shit !!
   copied = -1;
   //while (copied < 0){
      copied = CopyRates(symbol_name,periodo, StringToTime(start_time),TimeCurrent(),rates);   // StringToTime(end_time) TimeCurrent()
      num_downloaded = ArraySize(rates);
      printf("Downloaded %i candlesticks",num_downloaded);
     // Sleep(1000); // For some reason it makes it less prone to server erro
   //}
   

   // Sometimes we can get copied = -1 but we still downloaded the data. Check documentation
   if(num_downloaded>0){
      // If we actually could download data using CopyRates()
      int  digits=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
      // Create file to write the data
      FileHandle=FileOpen(FileName,FILE_WRITE|FILE_ANSI);   
      
      if(FileHandle!=INVALID_HANDLE) {
         FileWrite(FileHandle,// write data to file
             "Date,Open,High,Low,Close,Volume"); // Time

         for(int i = num_downloaded - 1; i >= 0; i--){
            TimeToStruct(rates[i].time,tm);
               FileWrite(FileHandle,// write data to file
                         string(tm.year)+ "-" + string(tm.mon)+ "-" + string(tm.day)+ " " + string(tm.hour)+ ":"  + string(tm.min) + ":00" + // Time
                         ", " + DoubleToString(rates[i].open,digits) + 
                         ", " + DoubleToString(rates[i].high,digits) + 
                         ", " +DoubleToString(rates[i].low,digits) + 
                         ", " +DoubleToString(rates[i].close,digits) + 
                         ", " + IntegerToString(rates[i].tick_volume));
            }
            
            printf("File created for  %s_%s ",symbol_name,periodo_txt); 
              // Close file (free handle), to make it available for other programs
            FileClose(FileHandle);
        }
      else Print("Error in call of CopyRates for the Symols",Symbol()," err=",GetLastError());
      printf("Sucessful file created for  %s_%s ",symbol_name,periodo_txt);
     }
     else printf("No file created for  %s_%s ",symbol_name,periodo_txt);
     
   ChartClose(chart_id);
   return(num_downloaded);
}


// symbols_info_to_csv
//+------------------------------------------------------------------+

bool symbols_info_to_csv(){
   /*
   This function downloads all the info of the symbols and saves it to csv

   Example:
   Symbol,PointSize,MinTickValue,ContractSize,Currency
   GBPUSD,0.00001000,1.00000000,100000.00000000,GBP
   USDCHF,0.00001000,1.02115840,100000.00000000,USD
   
   */
   ResetLastError();
   
   int i;
   string Symbols_names[];  //We cannot declare just a pointer so we will reserve minimum memory and then use  ArrayResize()
   string symbols_file_name = "Symbol_info.csv";
   // Bool to only get the symbols in the watch list or all of them actually.
   bool selected = false;
   
   // Get the number of all symbols we have access to. If true, only of those in the watchlist
   int Nsym = SymbolsTotal(selected);  
   printf("Number of symbols = %i",Nsym); // Print the obtained number of symbols
  
   int error = GetLastError();
   if(error != 0){
      Print("Error: ", error);
   }
   ResetLastError();
   // Save the names of the symbols in the array Symbols_names
   ArrayResize(Symbols_names,Nsym,Nsym);
   for (i = 0; i < Nsym; i++) {
      Symbols_names[i] =  SymbolName(i, selected);
   }
 
   error = GetLastError();
   if(error != 0){
      Print("Error: ", error);
   }
   ResetLastError();
   
   MqlTick tickdata;
   
   printf("Writing information about symbols in: %s", symbols_file_name);
   int FileHandle=FileOpen(symbols_file_name,FILE_WRITE|FILE_ANSI);   
   if(FileHandle!=INVALID_HANDLE) {
      FileWrite(FileHandle,// write data to file
          "Symbol,PointSize,MinTickValue,ContractSize,Currency"); // Time
      // For each symbol we get its information
      for (i = 1; i < Nsym; i++) {
            SymbolInfoTick(Symbols_names[i], tickdata);
            FileWrite(FileHandle,// write data to file
                Symbols_names[i] +
                "," + DoubleToString(SymbolInfoDouble(Symbols_names[i],SYMBOL_POINT)) +
                "," + DoubleToString(SymbolInfoDouble(Symbols_names[i],SYMBOL_TRADE_TICK_VALUE)) +
                "," + DoubleToString(SymbolInfoDouble(Symbols_names[i],SYMBOL_TRADE_CONTRACT_SIZE)) +
                "," + SymbolInfoString(Symbols_names[i],SYMBOL_CURRENCY_BASE)
            ); // Time
       }
   }
  
     // Close file (free handle), to make it available for other programs
   FileClose(FileHandle);
   
   error = GetLastError();
   if(error != 0){
      Print("Error: ", error);
   }
   ResetLastError();
   
   Print("File written");
   
   error = GetLastError();
   if(error != 0){
      Print("Error: ", error);
   }
   return true;
 }


bool positions_info_to_csv(){
   CPositionInfo  m_position;                   // trade position object
   CTrade         m_trade;                      // trading object
   
   string symbols_file_name = "Open_positions.csv";
   int n_positions=PositionsTotal();
   printf("****** There are %i Open positions ******",n_positions);
   printf("Writing information about Open positions in: %s", symbols_file_name);
  
  
   int FileHandle=FileOpen(symbols_file_name,FILE_WRITE|FILE_ANSI);   
   if(FileHandle!=INVALID_HANDLE) {
      FileWrite(FileHandle,// write data to file
          "Symbol,Volume,MagicNumber,Comment"); // Time
      // For each symbol we get its information
      
   for(int i= 0; i < n_positions;i++) // returns the number of current positions
      if(m_position.SelectByIndex(i)){     // selects the position by index for further access to its properties
        // printf("%i) %s", i+1,  m_position.Symbol(), m_position.Volume());
           printf("%i) Comment: %s. Ref Magic: %i", i+1, m_position.Comment(), m_position.Magic());
           
            FileWrite(FileHandle,// write data to file
                m_position.Symbol() + "," +
                DoubleToString(m_position.Volume()) + "," +
                IntegerToString(m_position.Magic())+ "," +
                m_position.Comment()

            ); // Time
        }
    }
 
    // Close file (free handle), to make it available for other programs
   FileClose(FileHandle);
   
   printf("File written");
   return true;
 }
 
 
//+------------------------------------------------------------------+

