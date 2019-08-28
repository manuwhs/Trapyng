#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:38:01 2019

@author: montoya
"""

"""
Let us see if we can clean the drafts from email.
We want to:
    - Go through all the drafts emails in a gmail account.
    - Read their content and transform it to plain text.
    - Parse their content into latex:
        - Add \item and new lines
        - Add the date of each email.
    - Save the document to txt file.
    - Optional: Remove the drafts
    
Things to be aware of: !!!
    - You cannot get your emails back if you delete them.
    - The name of the Draft folder changes with the language of your gmail.
    - There is a partial functionality to download the files attached but it is not used.
    - No handling of embedded images.
"""

import imaplib
import os
import email
import html2text
import datetime as dt

class email_handler():
    def __init__(self, user, password):
        self.email_user = user
        self.email_pass = password
        self.port = 993
        self.host = "imap.gmail.com"
    
        self.mail = None
        
    def login(self):
        print ("Logging in...")
        self.mail = imaplib.IMAP4_SSL(self.host,self.port)
        self.mail.login(self.email_user, self.email_pass)
        print ("Logged into: ",self.email_user)
        
    def search_uids(self, folder = "[Gmail]/Drafts"):
        """
        Returns the uids of the emails in the search
        """
        print ("Retrieving mails uids: ",self.email_user)
        self.mail.select()
        
        list_email_folders = self.mail.list()[1]
        print ("List of folders:")
        print(list_email_folders)
        self.mail.select(folder) # connect to inbox.
        
        # Search and return uids We dont get the emails themselves.
        result, data = self.mail.uid('search', None, "ALL") 
        self.list_uids =  data[0].split()   # List of uid
        N_mails = len(self.list_uids)     # Number of mails retrieved in the search
#        latest_email_uid = list_uids[-1]  # Last one retrieved
        print ("%i uids retrieved: "%(N_mails),self.email_user)
        return self.list_uids 

    def read_email(self, uid):
        """
        Reads an email by its uid.
        It returns its "email library formatted" email_message
        """
        
        typ, data = self.mail.uid('fetch', uid, '(RFC822)')   ## Fetch the email by uid !!
        raw_email = data[0][1]
        # converts byte literal to string removing b''
        raw_email_string = raw_email.decode('utf-8')
        
        # get the email_message formated from the email library !!
        email_message = email.message_from_string(raw_email_string)
        
        return email_message
    
    def download_attachments(self, uid, email_message):
        """
        Download all the attachments of a given email
        """
        #    downloading attachments
        for part in email_message.walk():
            # this part comes from the snipped I don't understand yet... 
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            fileName = part.get_filename()
            if bool(fileName):
                filePath = os.path.join('/Users/sanketdoshi/python/', fileName)
                if not os.path.isfile(filePath) :
                    fp = open(filePath, 'wb')
                    fp.write(part.get_payload(decode=True))
                    fp.close()
                    subject = str(email_message).split("Subject: ", 1)[1].split("\nTo:", 1)[0]
                    print('Downloaded "{file}" from email titled "{subject}" with UID {uid}.'.format(file=fileName, subject=subject, uid=uid.decode('utf-8')))
        return email_message

    def get_first_text_block(self, email_message = None, plain_text = True):
        """
        Gets an email message as input and returns its text content
        """

        maintype = email_message.get_content_maintype()
        payload = b"" # Empty just in case
#        print (maintype)
        if maintype == 'multipart':
            for part in email_message.get_payload():
                if part.get_content_maintype() == 'text':
                    ## We get the last one !! Which should be the last edition of the draft?
                    payload =  part.get_payload(decode=True)
        elif maintype == 'text':
            payload +=  email_message.get_payload(decode=True)
        
#        print (payload)
        if (plain_text):
            return html2text.html2text(payload.decode('utf-8'))
        else:
            return payload
    
    def print_email_data(self, email_message = None):

        email_to = email_message['To']
        email_from =  email.utils.parseaddr(email_message['From']) # for parsing "Yuji Tomita" <yuji@grovemade.com>
        email_text = self.get_first_text_block(email_message = email_message) # print all headers

        print ("Email to: ", email_to)
        print ("Email from: ", email_from)
        print ("Content")
        print (email_text)
        
    
    def save_all_ids_into_latex_format(self, folder ="[Gmail]/Drafts", document = "./drafts.txt"):
        """
        Saves all of the drafts into disk with latex format
        """
        draft_uids = my_email_handler.search_uids(folder = folder)
        Nemails = len(draft_uids)
        all_text = ""
        
        print ("Total emails: ", Nemails)
        for i in range(Nemails): ##
            
            if (i % 10 == 0):
                print ("Processing ", i, "/", Nemails)
            
            uid = draft_uids[i]
            email_message = my_email_handler.read_email(uid)
            
            date = email_message["Date"].split(":")[0]
            text = my_email_handler.get_first_text_block(email_message)
            
            text = "\item ["+ date +"] \n" + text + "\n"
            
            all_text+= text
        
        date = dt.datetime.now()
        fd = open("drafts_"+ str(date) + ".txt", "w+")
        fd.write(all_text)
        
        fd.close()

    def remove_emails(self,folder ="[Gmail]/Drafts"):
        """
        Remove all the drafts from the email
        """
        draft_uids = my_email_handler.search_uids(folder = folder)
        Nemails = len(draft_uids)
        print ("Total emails delete: ", Nemails)
        for i in range(Nemails): ##
            if (i % 10 == 0):
                print ("Processing ", i, "/", Nemails)
            
            uid = draft_uids[i]
            self.mail.uid("store",uid, '+FLAGS', r'(\Deleted)')
        
        self.logout()
        
    def logout(self):
        if (self.mail is None):
            pass
        else:
            self.mail.expunge()
            self.mail.close()
            self.mail.logout()
    
    
#########################################################################
#################### USAGE ###########################################
#########################################################################
            
user = "manuwhs@gmail.com"
password = "manumon7g.@"
## Instantiate the class
my_email_handler = email_handler(user, password)

# Login and save all the drafts
my_email_handler.login()
folder_download= "[Gmail]/Kladder"
#my_email_handler.save_all_ids_into_latex_format(folder = folder_download, document = "./drafts.txt")
my_email_handler.remove_emails(folder_download)



