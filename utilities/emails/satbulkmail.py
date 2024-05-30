from email.mime.application import MIMEApplication
import ssl
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

load_dotenv()

def satbulkmail(sender_emailID: str, receiver_emailID: str, mail_subject: str, pdf_blob: bytes = None, pdf_filename: str = None, cc_addresses: list = None, client_name: str = "Aspirant"):
    email_receiver = receiver_emailID
    email_sender = sender_emailID
    PSK = sender_emailID.split("@", 1)
    PSK = PSK[0] + "_PSK"
    email_password = os.getenv(PSK.upper())
    subject = mail_subject

    # Plain text body
    plain_body = """
Dear {0},

As your authorized representative for your visa application, I am writing to update you on your current visa status. Please review the attached PDF for details.

We will notify you directly of any changes to your application status. Additionally, weekly updates will be provided every Saturday by Root On Immigration Consultants.

Thank you for your cooperation. Please be aware that processing times can vary due to high volumes at IRCC, and we encourage you to regularly check their website for the latest updates: IRCC Processing Times.

Should you have any further questions or concerns, please do not hesitate to email us.

Regards,
Team Root On

RCIC, ICCRC Council Member Firm (R529956)
https://iccrc-crcic.ca/find-a-professional/
Root On Immigration Consultants Inc.
https://linktr.ee/rooton
""".format(client_name)

    html_body = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
</head>
<body style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;">
    <div id="__react-email-preview" style="display:none;overflow:hidden;line-height:1px;opacity:0;max-height:0;max-width:0">Latest updates on your visa application status.<div></div></div>
    <table style="width:100%;background-color:#ffffff;margin:0 auto;" align="center" border="0" cellpadding="0" cellspacing="0" role="presentation">
        <tbody>
            <tr>
                <td>
                    <div style="max-width:600px;margin:0 auto">
                        <table style="width:100%;margin-top:32px" align="center" border="0" cellpadding="0" cellspacing="0" role="presentation">
                            <tbody>
                                <tr>
                                    <td><img alt="Rooton" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display:block;outline:none;border:none;text-decoration:none" /></td>
                                </tr>
                            </tbody>
                        </table>
                        <h1 style="color:#1d1c1d;font-size:36px;font-weight:700;margin:30px 0;padding:0;line-height:42px">Latest Visa Application Status</h1>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Dear {1},</p>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">As your authorized representative for your visa application, I am writing to update you on your current visa status. Please review the attached PDF for details.</p>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">We will notify you directly of any changes to your application status. Additionally, weekly updates will be provided every Saturday by Root On Immigration Consultants.</p>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Thank you for your cooperation. Please be aware that processing times can vary due to high volumes at IRCC, and we encourage you to regularly check their website for the latest updates: <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/application/check-processing-times.html">IRCC Processing Times.</a></p>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Should you have any further questions or concerns, please do not hesitate to email us.</p>
                        <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Regards,<br>Team Root On</p>
                        <table style="margin-bottom:32px;width:100%" border="0" cellpadding="0" cellspacing="10" align="left">
                          <tr>
                            <td align="left" valign="top"><img alt="Rooton" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display:block;outline:none;border:none;text-decoration:none" /></td>
                            <td align="right" valign="top">
                                <a target="_blank" style="color:#067df7;text-decoration:none" href="https://www.facebook.com/pg/rooton/">
                                    <img alt="Facebook" src="https://cdn-icons-png.flaticon.com/512/739/739237.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" />
                                </a>
                                <a target="_blank" style="color:#067df7;text-decoration:none" href="https://instagram.com/rootonofficial">
                                    <img alt="Instagram" src="https://cdn-icons-png.flaticon.com/512/87/87390.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" />
                                </a>
                                <a target="_blank" style="color:#067df7;text-decoration:none" href="https://www.linkedin.com/in/ronak-patel-rcic/">
                                    <img alt="LinkedIn" src="https://cdn-icons-png.flaticon.com/512/220/220343.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" />
                                </a>
                            </td>
                          </tr>
                        <p style="font-size:14px;line-height:24px;margin:16px 0;color:#000">RCIC, ICCRC Council Member Firm (R529956)<br>
                        <a href="https://iccrc-crcic.ca/find-a-professional/" style="color:#067df7;text-decoration:none" target="_blank">https://iccrc-crcic.ca/find-a-professional/</a><br>
                        Root On Immigration Consultants Inc.<br>
                        <a href="https://linktr.ee/rooton" style="color:#067df7;text-decoration:none" target="_blank">https://linktr.ee/rooton</a></p>
                        
                        </table>
                        <p style="font-size:12px;line-height:1px;margin:16px 0;color:#b7b7b7;text-align:left;margin-bottom:50px">
                            {0}
                        </p>
                    </div>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
""".format(datetime.now(), client_name)

    # Create the root message
    msgRoot = MIMEMultipart("related")
    msgRoot["Subject"] = subject
    msgRoot["From"] = email_sender
    msgRoot["To"] = email_receiver
    if cc_addresses:
        msgRoot["Cc"] = ', '.join(cc_addresses)

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
        smtp.sendmail(email_sender, [email_receiver] + (cc_addresses if cc_addresses else []), msgRoot.as_string())