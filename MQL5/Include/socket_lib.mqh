//+------------------------------------------------------------------+
//|                                                   socket_lib.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+


string format_data_to_send_through_socket(double& clpr[]){
    // This function formats the data to be sent towards the Python listener
     string tosend;
     for(int i=0;i<ArraySize(clpr);i++) tosend+=(string)clpr[i]+" "; 
     return tosend;
}


bool socksend(int sock,string request) {
  // Send the info through the socket
    char req[];
    
    request = request + "#";
    int  len=StringToCharArray(request,req)-1;
    if(len<0) return(false);
    
    int sent_info = 0;
    
    if(SocketIsWritable(sock)){
       while (sent_info < len){
         sent_info = SocketSend(sock,req,len);
         printf("Sent %i/%i",sent_info,len);
       }
    }
    else{
      printf("Socket %i is not writable",sock);
    }
    return sent_info == len; 
}
  
string socketreceive(int sock, int timeout) {
   /*
   The read function has a timeout, if it does not receive data, it exits.
   To make sure we are receiving the entire message once we started reading the 
   beggining, the last characted of the message, has to be "#".
   Why? Because we always finish with hash.
   */
    Print("Waiting to receive data from socket ",sock, " timeout: ",timeout);
    string result = "";
    
    // Wait to receive information from the socket. 
    // If the socket is readable then read it
    
   result = socketreceive_timeout(sock,timeout);
   
   if(StringLen(result)>0) {
      while (result[StringLen(result)-1] != '#'){
         Print("Waiting for the end of the command", result);
          result += socketreceive_timeout(sock,timeout);
     }
      
     result = StringSubstr(result,0,StringLen(result)-1); // Remove the #
     Print("received: ",result);
     }
    
   return result; 
 }
 
 string socketreceive_timeout(int sock,int timeout)
  {
   char rsp[];
   string result="";
   uint len;
   uint timeout_check=GetTickCount()+timeout;
   do
     {
      len=SocketIsReadable(sock);
      if(len)
        {
         int rsp_len;
         rsp_len=SocketRead(sock,rsp,len,timeout);
         if(rsp_len>0) 
           {
            result+=CharArrayToString(rsp,0,rsp_len); 
           }
        }
     }
   while((GetTickCount()<timeout_check) && !IsStopped());
   return result;
  }