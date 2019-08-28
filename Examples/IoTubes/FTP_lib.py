
#import ftplib
import pysftp
import utilities_lib as ul
 # Code for FTP sending the file to the website.
 
class FTP_config_info():
    URL =  'iotubes.dk' # "sftp://iotub5es.dk" #
    PORT = 22
    USER = 'iotu41656322575'
    PASSWORD = 'FD+s@8aiIOC'
    
class SFTP_class():
    def __init__(self, mySFTP_config_info = None):
        
        self.connection_flag = False
        # Flag to know if the connection is open, they way it will for the higher order functions is:
        # If a connection is already stablished, then it does the job and that is it.
        # If it is not, then it opens it and closes it in the end
        
        if (type(mySFTP_config_info) == type(None)):
            self.mySFTP_config = FTP_config_info()
        else:
            self.mySFTP_config = mySFTP_config_info
    
    def connect(self):
        """ Establish connection with the server """
        
        if (self.connection_flag == False):
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None   
            sftp = pysftp.Connection(self.mySFTP_config.URL, username=self.mySFTP_config.USER, 
                                     password=self.mySFTP_config.PASSWORD, cnopts = cnopts)
            
            self.connection_flag = True
            self.sftp = sftp
        
    def close(self):
        if(self.connection_flag == True):
            self.sftp.close()
            self.connection_flag = False
        
    def get_list_dir(self,subfolder = "wp-content"):
        """ 
        Function that retrieves the documents in the subfolder specified
        """
        self.connect()
        dire = None
    #    subfolder = '/home/iotu41656322575/html/wp-content/uploads/2018/05/'
        with self.sftp.cd(subfolder):  
            dire = self.sftp.listdir()
        self.close()
        return dire
    
    def get_file(self, remotepath, localpath = "./", preserve_mtime=False):
        """ 
        Function that downloads the specified file from the server into the local file name provided
        """
        self.connect()
        self.sftp.get(remotepath, localpath, preserve_mtime=preserve_mtime)
        self.close()
        
    def get_dir(self, remotedir, localdir = "./", recursive = False, preserve_mtime=False):
        """ 
        Function that downloads the specified file from the server into the local file name provided.
        If Recursive it will download all subfolders.
        """
        
        self.connect()
        ul.create_folder_if_needed(localdir)
        if(recursive):
            self.sftp.get_r(remotedir, localdir, preserve_mtime=preserve_mtime)
        else:
            self.sftp.get_d(remotedir, localdir, preserve_mtime=preserve_mtime)
        self.close()
    
    def make_dir(self, remotedir, mode=777):
        """
        Make a directory and assign rights, it will create all the substructure needed"""
        self.connect()
        self.sftp.makedirs(remotedir, mode)
        self.close()


    def put_file(self, localpath, remotepath=None, callback=None, confirm=True, preserve_mtime=False):
        """
        Make a directory and assign rights, it will create all the substructure needed.
        Files are overwritten !!
        """
        self.connect()
        self.sftp.put(localpath, remotepath=remotepath, callback=callback, confirm=confirm, preserve_mtime=preserve_mtime)
        self.close()


    def put_dir(self, localdir = "./" ,remotedir = None , recursive = False, confirm=True, preserve_mtime=False):
        """ 
        Copy a directory form source
        Apparently recursive = False, gives a directory reference error
         Files are overwritten !!
        """
        self.connect()
        
        if(self.sftp.lexists(remotedir) == False):
            self.make_dir(remotedir)
            
        if(recursive):
            self.sftp.put_r(localdir, remotedir , confirm=confirm, preserve_mtime=preserve_mtime)
        else:
            self.sftp.put_d(localdir, remotedir, confirm=confirm, preserve_mtime=preserve_mtime)
        
        self.close()
    
        pass
    #        sftp.cd()         # temporarily chdir to public
        #    sftp.put('/my/local/filename')  # upload file to public/ on remote
        #    sftp.get('remote_file')         # get a remote file
    
        
        ## get file
    #    filename = 'wp-links-opml.php'
    #    data_file = sftp.get(filename, preserve_mtime=True)
    #    filep = open(destination_file,'wb')                  # file to receive
    #    session.retrbinary('RETR '+destination_file, file.write ,1024)     # send the file
    #    filep.close()        
         
    
def download_folder():
    pass
    
if(0):
    def send_file(FTP_config_info, source_file = "", destination_file = ""):
        # might be needed for a different port FTP.connect(host[, port[, timeout]]) and login
        session = ftplib.FTP(FTP_config_info.URL,FTP_config_info.USER,FTP_config_info.PASSWORD)
        file = open(source_file,'rb')                  # file to send
        session.storbinary('STOR '+destination_file, file)     # send the file
        file.close()                                    # close file and FTP
        session.quit()
    
    def get_file(FTP_config_info, source_file = "", destination_file = ""):
        session = ftplib.FTP(FTP_config_info.URL,FTP_config_info.USER,FTP_config_info.PASSWORD)
        file = open(destination_file,'wb')                  # file to receive
        session.retrbinary('RETR '+destination_file, file.write ,1024)     # send the file
        file.close()                                    # close file and FTP
        session.quit()
    
    def mofify_file(FTP_config_info, source_file = "", destination_file = ""):
        # Basically delete + reupload.
        FTP.delete(filename)
    
    ## Using in the program would be:
    def add_report_to_web(local_report_file_dir, report_name = ""):
        # This function uploads the report on the web via FTP and uploads and HTML file so that we can download it.
    
        # Folder path where to add the report
        server_reports_folder = ""
        server_report_name = None    # Random name of the report in the server, we are that cool and security intelligent of course
        server_report_name = str(int(np.random.randn())*1000000000000000) + ".docx"
    
        # HTML files modification variables
        html_file_dir = "/caca"
        local_file_name = "./ggweh"
    
        # Upload the report file
        send_file(FTP_config_info, source_file = report_file_dir, destination_file = server_reports_folder + server_report_name )
    
        # Modify the HTML in the website, we download it, modify it and reupload it.
        get_file(FTP_config_info, source_file = html_file_dir, destination_file = local_file_name)
        file = open(local_file_name,'wb') 
    
        included_line = "\n" + "<a href='/"+server_reports_folder+server_reports_name+ "'download='" + report_name + "'>"
        file.writeline(included_line);
        file.close()
        # Finally modify the file in the DDBB
        mofify_file(FTP_config_info, source_file = local_file_name, destination_file = server_reports_folder + server_report_name )
