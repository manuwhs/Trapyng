//+------------------------------------------------------------------+
//|                                                         util.mqh |
//|                                                   Manuel Montoya |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Manuel Montoya"
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+

bool isNewBar(string symbol, ENUM_TIMEFRAMES period)
  {
//--- memorize the time of opening of the last bar in the static variable
   static long last_time=0;
//--- current time
   long lastbar_time=SeriesInfoInteger(symbol,period,SERIES_LASTBAR_DATE);

//--- if it is the first call of the function
   if(last_time==0)
     {
      //--- set the time and exit
      last_time=lastbar_time;
      return(false);
     }

//--- if the time differs
   if(last_time!=lastbar_time)
     {
      //--- memorize the time and return true
      last_time=lastbar_time;
      return(true);
     }
//--- if we passed to this line, then the bar is not new; return false
   return(false);
  }
  
 //| fTimeFrameName                                                   

// This function returns the string version of  the time periods ints defined in the system|

//+------------------------------------------------------------------+

string fTimeFrameName(int arg){
   int v;
   if(arg==0)
     {
      v=_Period;
     }
   else{
      v=arg;
     }
   switch(v){
      case PERIOD_M1:    return("M1");
      case PERIOD_M5:    return("M5");
      case PERIOD_M15:   return("M15");
      case PERIOD_M30:   return("M30");
      case PERIOD_H1:    return("H1");
      case PERIOD_H4:    return("H4");
      case PERIOD_D1:    return("D1");
      case PERIOD_W1:    return("W1");
      case PERIOD_MN1:   return("MN1");
      default:    return("?");
     }

  } // end fTimeFrameName
  
  
ENUM_TIMEFRAMES min2ENUM_TIMEFRAME(int arg){

   int v;
   if(arg==0)
     {
      v=_Period;
     }
   else{
      v=arg;
     }
   switch(v){
      case 1: return(PERIOD_M1);
      case 5: return(PERIOD_M5);
      case 15:  return(PERIOD_M15);
      case 1440: return(PERIOD_D1);
      default:    return(PERIOD_M1);
     }

  } // end fTimeFrameName
  

void drawlr(string points, int lrlenght) {
    string res[]; 
    StringSplit(points, ' ', res);
        
    if(ArraySize(res)==2) {      
        Print(StringToDouble(res[0]));
        Print(StringToDouble(res[1]));
        datetime temp[]; 
        CopyTime(Symbol(),Period(),TimeCurrent(),lrlenght,temp); 
        ObjectCreate(0,"regrline",OBJ_TREND,0,TimeCurrent(),NormalizeDouble(StringToDouble(res[0]),_Digits),temp[0],NormalizeDouble(StringToDouble(res[1]),_Digits));
   } 
 }  