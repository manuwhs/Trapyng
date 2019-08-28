//+------------------------------------------------------------------+
//|                                   All_data_to_csv_downloader.mq5 |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <socket_lib.mqh>
#include <util.mqh>
#include <socket_commands.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

/*

In this code we want to:
- Connect to Python Server
- From python ask for downloading of data to csv interactively.
- 

*/

#include <broker_to_csv.mqh>

void OnStart(){
   int socket;
   socket=SocketCreate();
   int port = 9095;
    int timeout = 1000;
   string result = "";
   bool success;
   
   // string split variables

   string sep=" ";                // A separator as a character 
   ushort u_sep;                  // The code of the separator character 
   string splits[];               // An array to get strings 
   u_sep=StringGetCharacter(sep,0); 
   int num_splits;
   string command;
 
   if(socket!=INVALID_HANDLE) {
      if(SocketConnect(socket,"localhost",port,1000)) {
         Print("Connected to "," localhost",":",port);
         // wait for receiving data 
         while (1){
            // Sleep(1000);
            result = socketreceive(socket, timeout);
            if (StringCompare("", result) !=0){
               Print("Command: ", result );
               num_splits =StringSplit(result,u_sep,splits); 
               command = splits[0];
               
               if (StringCompare(result, "DOWNLOAD_SYMBOLS_INFO")==0){
                  success = symbols_info_to_csv();
                  
                  if(success){
                     printf("Successful download");
                     success = socksend(socket, "OK");
                  }
                  else{
                     printf("Not successful download");
                     success = socksend(socket, "ERROR");
                  }
                  
                 //  break;
               }
               else if(StringCompare(command,"CSV_DATA_REQUEST")==0){
                  
                  success = timeSeries_to_csv(splits[1],min2ENUM_TIMEFRAME(StringToInteger(splits[2])),splits[3],splits[4],splits[5]);
                  if(success){
                     printf("Successful download");
                     success = socksend(socket, "OK");
                  }
                  else{
                     printf("Not successful download");
                     success = socksend(socket, "ERROR");
                  }
                  if(success){
                     printf("Response sent");
                  }
                  else{
                     printf("Response NOT sent");
                   }
               }
               
             }
          }
    }
    SocketClose(socket); 
  }
  
   
   

   //; // day, month, year 

   
   // timeSeries_to_csv("EURUSD",PERIOD_H1,day,month,year);
   
   return;// script completed
   
  }
//+------------------------------------------------------------------+
