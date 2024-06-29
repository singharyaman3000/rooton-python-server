from email.mime.application import MIMEApplication
import ssl
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

load_dotenv()

def paymailtoacc(payment_id: str, payment_amount: float, payment_date: str, client_name: str, client_email: str, client_address: str, service_plan: str, client_gst: str = None, pdf_blob: bytes = None, pdf_filename: str = None, cc_addresses: list = ['aryaman.singh@rooton.ca']):
    email_receiver = os.getenv("EMAIL_RECEIVER")
    email_sender = os.getenv("EMAIL")
    email_password = os.getenv("EMAIL_PSK")

    subject = f"Payment Received of Client: {client_name} 🎉 of Amount: ₹{payment_amount}"

    gst_info = f"GST Number: {client_gst}" if client_gst else "No GST applicable"

    # Plain text body
    plain_body = f"""
Dear Accounts Team,

We have received a payment via Razorpay. Below are the details:

Client Name: {client_name}
Client Email: {client_email}
Billing Address: {client_address}
{gst_info}
Amount: ₹{payment_amount}
Razorpay Payment ID: {payment_id}
Date of Payment: {payment_date}
Service/Plan: {service_plan}

Please update our records accordingly.

Regards
Aryaman Singh
Software Development Engineer
"""

    html_body = f"""
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
</head>
<body style="font-family: 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; margin: 0; padding: 0;">
    <div id="__react-email-preview" style="display: none; overflow: hidden; line-height: 1px; opacity: 0; max-height: 0; max-width: 0">Payment received via Razorpay.</div>
    <table style="width: 100%; background-color: #f9f9f9; margin: 0; padding: 20px 0;" align="center" border="0" cellpadding="0" cellspacing="0" role="presentation">
        <tbody>
            <tr>
                <td>
                    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
                        <table style="width: 100%; margin: 0; padding: 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; background-color: #f3f2f7;" align="center" border="0" cellpadding="0" cellspacing="0" role="presentation">
                            <tbody>
                                <tr>
                                    <td style="text-align: center;">
                                        <img alt="Root On Immigration & Consultants Pvt. Ltd" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display: block; outline: none; border: none; text-decoration: none; margin: 0 auto;" />
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        <table style="width: 100%; padding: 20px;" align="center" border="0" cellpadding="0" cellspacing="0" role="presentation">
                            <tbody>
                                <tr>
                                    <td>
                                        <h1 style="color: #333333; font-size: 24px; font-weight: 700; margin: 0; padding: 0; line-height: 1.4; text-align: center;">Payment Received</h1>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;">Dear Accounts Team,</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;">We have received a payment via Razorpay. Below are the details:</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Client Name:</strong> {client_name}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Client Email:</strong> {client_email}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; word-break: break-all; margin: 20px 0;"><strong>Billing Address:</strong> scgsccyscygysyiscsiucuicbiusbciubsicbsbcisbiucbsuicbsibc, scusgcubcsubcushcusbubsubvbv, shuchsucsuicnsiciscubsdkucbkusdhvusdhvugvigsdivgisvssbv, dsuhdsbvksdbvskdb</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>{gst_info}</strong></p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Amount:</strong> ₹{payment_amount}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Razorpay Payment ID:</strong> {payment_id}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Date of Payment:</strong> {payment_date}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;"><strong>Service/Plan:</strong> {service_plan}</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;">Please update our records accordingly.</p>
                                        <p style="color: #555555; font-size: 16px; line-height: 1.6; margin: 20px 0;">Regards<br>Aryaman Singh<br>Software Development Engineer</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        <table style="width: 100%; background-color: #f9f9f9; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; padding: 20px;" border="0" cellpadding="0" cellspacing="0" align="center">
                            <tr>
                                <td align="left" valign="top">
                                    <img alt="Root On" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display: block; outline: none; border: none; text-decoration: none;" />
                                </td>
                                <td align="right" valign="top">
                                    <a target="_blank" style="color: #067df7; text-decoration: none;" href="https://www.facebook.com/pg/rooton/">
                                        <img alt="Facebook" src="https://cdn-icons-png.flaticon.com/512/739/739237.png" width="32" height="32" style="display: inline; outline: none; border: none; text-decoration: none; margin-left: 10px;" />
                                    </a>
                                    <a target="_blank" style="color: #067df7; text-decoration: none;" href="https://instagram.com/rootonofficial">
                                        <img alt="Instagram" src="https://cdn-icons-png.flaticon.com/512/87/87390.png" width="32" height="32" style="display: inline; outline: none; border: none; text-decoration: none; margin-left: 10px;" />
                                    </a>
                                    <a target="_blank" style="color: #067df7; text-decoration: none;" href="https://www.linkedin.com/in/ronak-patel-rcic/">
                                        <img alt="LinkedIn" src="https://cdn-icons-png.flaticon.com/512/220/220343.png" width="32" height="32" style="display: inline; outline: none; border: none; text-decoration: none; margin-left: 10px;" />
                                    </a>
                                </td>
                            </tr>
                        </table>
                        <p style="font-size: 14px; line-height: 1.4; color: #000000; text-align: center; margin: 20px 0;">
                            RCIC, ICCRC Council Member Firm (R529956)<br>
                            <a href="https://iccrc-crcic.ca/find-a-professional/" style="color: #067df7; text-decoration: none;" target="_blank">https://iccrc-crcic.ca/find-a-professional/</a><br>
                            Root On Immigration Consultants Inc.<br>
                            <a href="https://linktr.ee/rooton" style="color: #067df7; text-decoration: none;" target="_blank">https://linktr.ee/rooton</a>
                        </p>
                        <p style="font-size: 12px; line-height: 1; color: #b7b7b7; text-align: center; margin: 20px 0;">
                            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </p>
                    </div>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
"""

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
