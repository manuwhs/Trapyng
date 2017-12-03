import email_lib as emlib

### Library to make easier to create and send automatic emails.
## It uses a generic email library also created that used functions
## from the real library. But out functions are more intuitive,
## and independent of the underlying library.

class Cemail():
    ## This is a class to fucking create a decent email
    ## It uses the functions of the lib
    def __init__(self, user = "", pwd = "", recipient = "", ):
        self.user = user
        self.pwd = pwd
        self.recipient = recipient
        self.subject = ""

    ###### CORE FUNCTIONS #####
    # The thing is that if do not specify a new value, we keep the previous.
    def set_user(self, user = ""):
        if (len(user) != 0):
            self.user = user
            self.msgRoot['From'] = user
            
    def set_pwd(self, pwd = ""):
        if (len(pwd) != 0):
            self.pwd = pwd
            
    def set_recipient(self, recipient = ""):
        if (len(recipient) != 0):
            self.recipient = recipient
            self.msgRoot['To'] = recipient
            
    def set_subject(self, subject = ""):
        if (len(subject) != 0):
            self.subject = subject
            self.msgRoot["Subject"] = subject
            
    ### MORE complex function !!!
    def create_msgRoot(self, user = "",recipient = "", subject = ""):
        self.msgRoot = emlib.create_msgRoot(user, recipient, subject)
        self.set_user(user)
        self.set_recipient(recipient)
        self.set_subject(subject)
        
    def add_HTML(self,html_text):
        emlib.add_HMTL(self.msgRoot, html_text)
        
    def add_image(self, filedir, inline = 1):
        emlib.add_image(self.msgRoot, filedir, inline)
    
    def add_file(self, filedir, filename = ""):
        emlib.add_file(self.msgRoot, filedir, filename)
        
    def send_email(self, recipient = ""):
        self.set_recipient(recipient)
        emlib.send_email(self.user, 
                        self.pwd, 
                        self.recipient, 
                        self.msgRoot, secure = 0)