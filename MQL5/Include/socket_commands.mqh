//+------------------------------------------------------------------+
//|                                              socket_commands.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
#include <socket_lib.mqh>
#include <util.mqh>
string BUY_COMMAND(){
   printf("tryng to buy");
   
   
   return "2";
}

string PLOT_LINE_COMMAND(int socket, int lrlenght){
         string received;
         string tosend;

         bool all_data_sent = false;
         // Download the latest lrlenght values
         Print("Downloading ",lrlenght," Close prices of ",_Symbol,"_",PERIOD_CURRENT);
         // Array where to store the downloaded data.
         double clpr[];
         int copyed = CopyClose(_Symbol,PERIOD_CURRENT,0,lrlenght,clpr);
         // Format the Array into something passable
         
         tosend = "PLOT_LINE " + format_data_to_send_through_socket (clpr);
         all_data_sent = socksend(socket, tosend);
    
         if (all_data_sent){
            // When all the data is sent, then we await listening for a response.
            received = socketreceive(socket, 10);
         }
         else{
            received = "";
         }
      drawlr(received, lrlenght); 
      
   return received;
}