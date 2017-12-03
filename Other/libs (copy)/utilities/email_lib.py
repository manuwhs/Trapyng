import smtplib
import os
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage


# For guessing MIME type based on file name extension
import mimetypes

from argparse import ArgumentParser

from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase


## Functions to handle mails

def create_msgRoot(user, recipient, subject):
    # Create the root message and fill in the from, to, and subject headers
    # Let's begin by creating a mixed MIME multipart message which will house 
    # the various components (email body, images displayed inline and downloadable attachments) of our message
    msgRoot = MIMEMultipart('mixed')
    msgRoot['Subject'] = subject #subject
    msgRoot['From'] = user
    msgRoot['To'] = recipient
    msgRoot["Reply-To"] = "Anomaly Support <different-address@anomaly.net.au>"
    msgRoot.preamble = 'This is a multi-part message in MIME format.'
    
#    ## Creat the body
#    msg = MIMEMultipart('alternative')
#    msgRoot.msg = msg
#    msgRoot.attach(msg)
    return msgRoot

def add_HMTL(msgRoot, html_text):
    # Create Alternative MIME part to append to the Root
    msgAlternative = MIMEMultipart('alternative')
    msgText = MIMEText(html_text, "html")
    msgAlternative.attach(msgText)
    msgRoot.attach(msgAlternative)


def add_file(msgRoot,filedir, filename = ""):
    ## Ataches any file to our mail
    ## If no given filename, we use filedir

    path = filedir
    if( len(filename) == 0):
        filename = filedir.split("/")[-1]
        
    # Guess the content type based on the file's extension.  Encoding
    # will be ignored, although we should check for simple things like
    # gzip'd or compressed files.
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        # No guess could be made, or the file is encoded (compressed), so
        # use a generic bag-of-bits type.
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    if maintype == 'text':
        with open(path) as fp:
            # Note: we should handle calculating the charset
            msg = MIMEText(fp.read(), _subtype=subtype)
    elif maintype == 'image':
        with open(path, 'rb') as fp:
            msg = MIMEImage(fp.read(), _subtype=subtype)
    elif maintype == 'audio':
        with open(path, 'rb') as fp:
            msg = MIMEAudio(fp.read(), _subtype=subtype)
    else:
        with open(path, 'rb') as fp:
            msg = MIMEBase(maintype, subtype)
            msg.set_payload(fp.read())
        # Encode the payload using Base64
        encoders.encode_base64(msg)
    # Set the filename parameter
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    msgRoot.attach(msg)
    
    
def add_image(msgRoot, filedir, inline = 1):
    # This function adds an image to the main message of the email
    # msgRoot: It is the main message
    # filedir: It is where the image is.
    # If attached == 1, then we put it as attached, if not, we put it in the
    # real email.


    #### FOR SOME REASON THIS HAS TO GO BEFORE LOADING THE IMAGE ####
    ## Put HTML in the mail to see the image inline
    if (inline == 1):
        text = '<img src="cid:' + filedir +'">'
        add_HMTL(msgRoot, text)
    ########################################################
    
    add_file(msgRoot, filedir)
#    ## Read the image and include it !!
#    fp = open(filedir, 'rb')
#    msgImage = MIMEImage(fp.read())
#    fp.close()
#    
#    # Define the image's ID as referenced above
#    # We use the filedir as identifier
#    msgImage.add_header('Content-ID', '<' + filedir+  '>')
#    msgRoot.attach(msgImage)
#    
def send_email(user, pwd, recipient, msgRoot, secure = 0):

    gmail_user = user
    gmail_pwd = pwd
    
    FROM = gmail_user
    TO = recipient

    try:

        if (secure == 1):
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        else:
            server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        if (secure != 1):
            server.starttls()
        server.login(gmail_user, gmail_pwd)
        
        server.sendmail(FROM, TO, msgRoot.as_string())
        server.close()
        print 'successfully sent the mail'
    except smtplib.SMTPAuthenticationError:
        print "failed to send mail"
        print smtplib.SMTPAuthenticationError
        return smtplib.SMTPAuthenticationError
#def get_image_as_str(img_dir):
#    img_data = open(img_dir, 'rb').read()
#    image = MIMEImage(img_data, name =os.path.basename(img_dir))
#    return image.as_string()
