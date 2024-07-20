import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def get_docuseal_templates_fn(client_mail: str):
    reqUrl = "https://api.docuseal.co/templates?q=" + client_mail

    headersList = { 'X-Auth-Token': os.getenv("XAuthToken") }

    payload = ""

    response = requests.get(reqUrl, data=payload,  headers=headersList)

    return response.json()