
import socket
from sklearn.linear_model import LinearRegression
import numpy as np

class socketserver:
    def __init__(self, address = '', port = 9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''
    
    def buy_sell_command(self, BUYSELL, symbol, volume, magicNumber):
        message = BUYSELL + " " + symbol + str(volume) + str(magicNumber)
        self.conn.send(bytes(message, "utf-8"))
    
    def request_csv_symbol_info(self):
        
        print ("Requesting csv with symbol information")
        command = "DOWNLOAD_SYMBOLS_INFO"
        message = command 
        self.conn.send(bytes(message, "utf-8"))
        
        response = self.read_all()
        print("Response: ", response)
        if(response == "OK"):
            pass 
        return response
    
    def request_csv_data_signal(self,symbol, period, sdate):
        """
        Request the historical data of a given signal:
        
        Example: EURUSD 15 10-13-2014
        
        The downloaded data will be saved in ./Trade/...." 
        It is meant to be then loaded by python and incorporate it 
        to the history. 
        """

        command = "CSV_DATA_REQUEST"
        message = command +" "+ symbol + " " + str(period)+" " + sdate
        print ("Requesting csv with symbol information: ", message)
        self.conn.send(bytes(message, "utf-8"))
        
        response = self.read_all()
        print("Response: ", response)
        if(response == "OK"):
            pass 
        
        return response
        
    def read_all(self,  min_charts_to = 4096):
        # TODO: Make polict of read all based on time or size of the message.
        # There is no automatic way to know. We could have a special message delimiter.
        
        # Read all the data send upon connection.
#        cummdata = ""
#        while len(cummdata) < num_charts_to_receive:
#            # Blocking function that reads up to a number of data given as input
#            # If we expect more, we should specify it
#            data = self.conn.recv(num_charts_to_receive)
#            print(data)
#            cummdata+=data.decode("utf-8")
#            
#            if not data:
#                break
        cummdata = self.conn.recv(10000)
        return cummdata
    
    def listen(self, backlog = 1):
        """
        Receive connection from MQL5 file
        """
        ## Accept the connection
        self.sock.listen(backlog)  
        self.conn, self.addr = self.sock.accept()
        
        print('connected to', self.addr)
        
    def wait_input_command_handler(self):
        """
        The format of the message is: COMMAND DATA
        Example:
            - BUY 
            - 
            - PLOT_LINE DATA1 DATA2 .... 
        """
        ## Receive all the available data from the socket until it is empty.
        self.cummdata = self.read_all();

        ## Perform an action !!
        if(len(self.cummdata) > 0):
            self.main_input_command_handler()
            
        command = self.cummdata.split(" ")[0]
        print ("The command is: ", command)
        
        if (command == "PLOT_LINE"):
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            
        elif (command == "BUY"):
            self.buy_sell_command(BUY_SELL = "BUY", symbol = "EURUSD", volume = 0.01)
            
        elif (command == "TRANSACTION_ACK"):
            print (command)
            
        elif (command == "FINISHED_DOWNLOAD"):
            print(command)
            
        return self.cummdata
            
    def __del__(self):
        self.sock.close()
        
def calcregr(msg = ''):
    # Remove command
    msg = msg[len("PLOT_LINE "):]
    chartdata = np.fromstring(msg, dtype=float, sep= ' ') 
    Y = np.array(chartdata).reshape(-1,1)
    X = np.array(np.arange(len(chartdata))).reshape(-1,1)
        
    lr = LinearRegression()
    lr.fit(X, Y)
    Y_pred = lr.predict(X)
    type(Y_pred)
    P = Y_pred.astype(str).item(-1) + ' ' + Y_pred.astype(str).item(0)
    print(P)
    return str(P)

