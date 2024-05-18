from email.message import EmailMessage
import ssl
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

load_dotenv()


def resetpasswordmail(receiver_emailID, auth_id, URLINTI):
    email_sender = os.getenv("EMAIL")
    email_password = os.getenv("EMAIL_PSK")
    email_receiver = receiver_emailID
    subject = "Reset Password"

    Link = f"{URLINTI}/reset-password?authId={auth_id}"

    Body = """
        Welcome to Root-on!
        This is your password reset link {0}
    """.format(
        Link
    )

    em = EmailMessage()
    msgRoot = MIMEMultipart("related")
    msgRoot["Subject"] = "Reset Link For Your Account"
    msgRoot["From"] = email_sender
    msgRoot["To"] = email_receiver
    msgRoot.preamble = "This is a multi-part message in MIME format."
    msgAlternative = MIMEMultipart("alternative")
    msgRoot.attach(msgAlternative)

    msgText = MIMEText("This is the alternative plain text message.")
    msgAlternative.attach(msgText)

    msgText = MIMEText(
        """
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en">
  <head>
    <meta http-equiv="Content-Type" content="text/html charset=UTF-8" />
  </head>
  <div id="__react-email-preview" style="display:none;overflow:hidden;line-height:1px;opacity:0;max-height:0;max-width:0">Your Account Verification Code Is ***-***<div> ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿ ‌‍‎‏﻿</div>
  </div>
  <table style="width:100%;background-color:#ffffff;margin:0 auto;font-family:-apple-system, BlinkMacSystemFont, &#x27;Segoe UI&#x27;, &#x27;Roboto&#x27;, &#x27;Oxygen&#x27;, &#x27;Ubuntu&#x27;, &#x27;Cantarell&#x27;, &#x27;Fira Sans&#x27;, &#x27;Droid Sans&#x27;, &#x27;Helvetica Neue&#x27;, sans-serif" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
    <tbody>
      <tr>
        <td>
          <div><!--[if mso | IE]>
            <table role="presentation" width="100%" align="center" style="max-width:600px;margin:0 auto;"><tr><td></td><td style="width:37.5em;background:#ffffff">
          <![endif]--></div>
          <div style="max-width:600px;margin:0 auto">
            <table style="width:100%;margin-top:32px" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
              <tbody>
                <tr>
                  <td><img alt="Rooton" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display:block;outline:none;border:none;text-decoration:none" /></td>
                </tr>
              </tbody>
            </table>
            <h1 style="color:#1d1c1d;font-size:36px;font-weight:700;margin:30px 0;padding:0;line-height:42px">Reset your Password With Ease!</h1>
            <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Your Reset Link is below - Click On it and we&#x27;ll help you get signed in.</p>
            <table style="width:100%;background:rgb(245, 244, 245);border-radius:4px;margin-right:50px;margin-bottom:30px;padding:43px 23px" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
              <tbody>
                <tr>
                  <td>
                    <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Click the link below to reset your password:</p>
                    <a href="{0}" style="background-color:#000;color:#ffffff;display:inline-block;font-family:sans-serif;font-size:18px;line-height:44px;text-align:center;text-decoration:none;width:200px;border-radius:5px;margin-bottom:30px" target="_blank">Reset Link</a>
                  </td>
                </tr>
              </tbody>
            </table>
            <p style="font-size:14px;line-height:24px;margin:16px 0;color:#000">If you didn't request this email, there's nothing to worry about - you can safely ignore it.</p>
            <table style="margin-bottom:32px;width:100%" border="0" cellPadding="0" cellSpacing="10" align="left">
              <tr>
                <td align="left" valign="top"><img alt="Rooton" src="https://i.postimg.cc/wMn5hJ9g/rooton.png" width="auto" height="50" style="display:block;outline:none;border:none;text-decoration:none" /></td>
                <td align="right" valign="top"><a target="_blank" style="color:#067df7;text-decoration:none" href="https://www.facebook.com/pg/rooton/"><img alt="Rooton" src="https://cdn-icons-png.flaticon.com/512/739/739237.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" /></a><a target="_blank" style="color:#067df7;text-decoration:none" href="https://instagram.com/rootonofficial"><img alt="Slack" src="https://cdn-icons-png.flaticon.com/512/87/87390.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" /></a><a target="_blank" style="color:#067df7;text-decoration:none" href="https://www.linkedin.com/in/ronak-patel-rcic/"><img alt="Rooton" src="https://cdn-icons-png.flaticon.com/512/220/220343.png" width="32" height="32" style="display:inline;outline:none;border:none;text-decoration:none;margin-left:32px" /></a></td>
              </tr>
            </table>
            <table style="width:100%;font-size:12px;color:#b7b7b7;line-height:15px;text-align:left;margin-bottom:50px" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
              <tbody>
                <tr>
                  <td><a target="_blank" style="color:#b7b7b7;text-decoration:underline" href="https://rooton.ca/immigration-insights" rel="noopener noreferrer">Our blog</a>   |   <a target="_blank" style="color:#b7b7b7;text-decoration:underline" href="https://rooton.ca/privacy-policy" rel="noopener noreferrer">Policies</a>   |   <a target="_blank" style="color:#b7b7b7;text-decoration:underline" href="https://rooton.ca/disclaimer" rel="noopener noreferrer">Disclaimer</a>   |   <a target="_blank" style="color:#b7b7b7;text-decoration:underline" href="https://rooton.ca/terms-and-conditions" rel="noopener noreferrer" data-auth="NotApplicable" data-linkindex="6">Terms & Conditions</a>
                    <p style="font-size:12px;line-height:15px;margin:16px 0;color:#b7b7b7;text-align:left;margin-bottom:10px">Copyright © 2024 Root On Immigration Consultants, Inc. or its affiliates.<br />706-1800, Blvd, Rene-Levesque Ouest,<br /> Montreal Quebec, H3H 2H2. <br /><p style="margin-block:6px">All Rights Reserved.</p></p>
                    <p style="font-size:12px;line-height:1px;margin:16px 0;color:#b7b7b7;text-align:left;margin-bottom:50px"><br />{1}</p>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div><!--[if mso | IE]>
          </td><td></td></tr></table>
          <![endif]--></div>
        </td>
      </tr>
    </tbody>
  </table>
</html>
""".format(
            Link, datetime.now()
        ),
        "html",
    )
    msgAlternative.attach(msgText)

    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(Body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, msgRoot.as_string())
