from email.mime.application import MIMEApplication
import ssl
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

load_dotenv()

def satbulkmail(sender_emailID: str, receiver_emailID: str, mail_subject: str, pdf_blob: bytes = None, pdf_filename: str = None):
    email_receiver = receiver_emailID
    email_sender = sender_emailID
    PSK = sender_emailID.split("@", 1)
    PSK = PSK[0] + "_PSK"
    email_password = os.getenv(PSK.upper())
    subject = mail_subject

    # Plain text body
    plain_body = """
Dear Aspirants,

As your authorized representative for your visa application, we would like to inform you of the latest updates on your visa application status. Below you will find the PDF of your visa status to give a better understanding. 

If in case we receive any change in the current status of your application, we will notify you personally. There will be a weekly email update every Saturday from Root On Immigration Consultants for the status of your application. 

Thank you for your cooperation in advance. However, there might be a delay from IRCC due to the high volume of applications, in that case, please do not panic as it will be the same for others too.

Please feel free to email us if you have any further questions or concerns.

Regards,
Team Root On
Team Leader

RCIC, ICCRC Council Member Firm (R529956)
https://iccrc-crcic.ca/find-a-professional/
Root On Immigration Consultants Inc.
https://linktr.ee/rooton
"""

    # HTML body
    html_body = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
</head>
<body style="font-family: Serif; font-size: 12pt;">
    <p>Dear Aspirants,</p>
    <p>As your authorized representative for your visa application, we would like to inform you of the latest updates on your visa application status. Below you will find the PDF of your visa status to give a better understanding.</p>
    <p>If in case we receive any change in the current status of your application, we will notify you personally. There will be a weekly email update every Saturday from Root On Immigration Consultants for the status of your application.</p>
    <p>Thank you for your cooperation in advance. However, there might be a delay from IRCC due to the high volume of applications, in that case, please do not panic as it will be the same for others too.</p>
    <p>Please feel free to email us if you have any further questions or concerns.</p>
    <p>Regards,<br>
    Team Root On<br>

    <img src="https://i.postimg.cc/wMn5hJ9g/rooton.png" alt="Rooton" height="50" />
    <p>RCIC, ICCRC Council Member Firm (R529956)<br>
    <a href="https://iccrc-crcic.ca/find-a-professional/">https://iccrc-crcic.ca/find-a-professional/</a><br>
    Root On Immigration Consultants Inc.<br>
    <a href="https://linktr.ee/rooton">https://linktr.ee/rooton</a></p>
    <p style="font-size:12px;line-height:1px;margin:16px 0;color:#b7b7b7;text-align:left;margin-bottom:50px"><br />{0}</p>
</body>
</html>
""".format(datetime.now())

    # Create the root message
    msgRoot = MIMEMultipart("related")
    msgRoot["Subject"] = subject
    msgRoot["From"] = email_sender
    msgRoot["To"] = email_receiver

    # Create the alternative part
    msgAlternative = MIMEMultipart("alternative")
    msgRoot.attach(msgAlternative)

    # Attach the plain text and HTML parts
    msgAlternative.attach(MIMEText(plain_body, "plain"))
    msgAlternative.attach(MIMEText(html_body, "html"))

    # Conditionally attach the PDF if provided
    if pdf_blob is not None and pdf_filename is not None:
        pdf_attachment = MIMEApplication(pdf_blob, _subtype="pdf")
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_filename)
        msgRoot.attach(pdf_attachment)

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Send the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, msgRoot.as_string())
