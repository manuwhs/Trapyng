import socket, numpy as np
from sklearn.linear_model import LinearRegression

# Now we can proceed to creating a class responsible for socket manipulation:

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
    
    def request_csv_data_signal(self,symbol, timeframe):
        """
        Request the historical data of a given signal:
        
        Example: sy
        
        The downloaded data will be saved in ./Trade/...." 
        It is meant to be then loaded by python and incorporate it 
        to the history. 
        """
        command = "CSV_DATA_REQUEST"
        message = command + symbol + timeframe
        self.conn.send(bytes(message, "utf-8"))
            
    def recvmsg(self):
        """
        Receive connection from MQL5 file
        """
        ## Accept the connection
        self.sock.listen(1)  
        self.conn, self.addr = self.sock.accept()
        
        print('connected to', self.addr)
        self.cummdata = ''

    
        ## Receive all the available data from the socket until it is empty.
        while True:
            data = self.conn.recv(10000)
            self.cummdata+=data.decode("utf-8")
            if not data:
                break
            
        ## Perform an action !!
        if(len(self.cummdata) > 0):
            print ("The received data is: ")
            print (self.cummdata)
        """
        The format of the message is: COMMAND DATA
        Example:
            - BUY 
            - 
            - PLOT_LINE DATA1 DATA2 .... 
        """
        
        command = self.cummdata.split(" ")[0]
        print ("The command is: ", command)
        
        if (command == "PLOT_LINE"):
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            
        elif (command == "BUY"):
            self.buy_sell_command(BUY_SELL = "BUY", symbol = "EURUSD", volume = 0.01)
        elif (command == "TRANSACTION_ACK"):
            print (command)
            
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


serv = socketserver('127.0.0.1', 9093)

while True:  
    print ("Listening")
    msg = serv.recvmsg()
