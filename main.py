from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pymongo
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise as sk
import json
import re
import openai
import time
from cachetools import TTLCache, cached
import math as m
import random as r
from email.message import EmailMessage
import ssl
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import secrets
import bcrypt
from bson import ObjectId

load_dotenv()

cache = TTLCache(maxsize=1000000, ttl=86400)

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Pydantic models for request and response
class CourseRequest(BaseModel):
    fos: str
    level: str
    dictionary: dict
    email: str


class CourseResponse(BaseModel):
    data: dict


class EmailRequest(BaseModel):
    email: str
    Firstname: str
    Lastname: str
    # EmailId: str
    Phone: str
    Password: str


class AuthRequest(BaseModel):
    authId: str


class LoginRequest(BaseModel):
    email: str
    Password: str


# Function to fetch all data (similar to your script)
@cached(cache)
def fetch_all_data(database, collection):
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(MONGODB_URI)

    db = client[database]
    collection = db[collection]

    cursor = collection.find({})

    results = []
    try:
       for document in cursor:
           results.append(document)
    finally:
        cursor.close()
        client.close()

    return results



def perform_database_operation(database, collection_name, operation_type, query=None, update_data=None):
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(MONGODB_URI)

    db = client[database]
    collection = db[collection_name]

    if operation_type == "read":
        cursor = collection.find(query)
        documents = list(cursor)
        client.close()
        return documents

    elif operation_type == "update":
        if query is None or update_data is None:
            raise ValueError(
                "Both query and update_data must be provided for update operation."
            )

        # Prepare update operations
        update_operations = {}
        if "unset" in update_data:
            update_operations["$unset"] = update_data["unset"]
            del update_data["unset"]

        if update_data:  # Check if there's anything left to set
            update_operations["$set"] = update_data

        result = collection.update_one(query, update_operations)
        client.close()
        return result.modified_count

    elif operation_type == "create":
        if query is None:
            raise ValueError("query must be provided for create operation.")
        # List of allowed email domains
        allowed_domains = ["rooton.ca"]  # Add more domains as needed

        # Extract the email domain
        email = query.get("email", "")
        domain_matched = False

        for allowed_domain in allowed_domains:
            if email.endswith("@" + allowed_domain):
                query["Role"] = "Counselor"
                domain_matched = True
                break

        if not domain_matched:
            query["Role"] = "Student"
        result = collection.insert_one(query)
        client.close()
        return result.inserted_id  # Return the _id of the inserted document

    elif operation_type == "delete":
        if query is None:
            raise ValueError("query must be provided for delete operation.")
        result = collection.delete_one(query)
        client.close()
        return result.deleted_count  # Return the number of deleted documents

    else:
        raise ValueError(
            "Invalid operation type. Supported types are 'read', 'update', 'create', and 'delete'."
        )


async def preload_cache():
    try:
        # Preload data for each collection
        print("caching started")
        fetch_all_data("test", "userdetails")
        print("caching done 1")
        fetch_all_data("test", "courses")
        print("caching done 2")
        # Add more collections as needed
    except Exception as e:
        print(f"Error during cache preloading: {e}")


app.add_event_handler("startup", preload_cache)


def process_budget(budget_str):
    """Extract the lower and upper fee limits from the budget string."""
    if "to above" in budget_str:
        lower_limit = int(budget_str.split(" ")[0])
        upper_limit = int("999999")
    else:
        lower_limit, upper_limit = map(int, budget_str.split("-"))
    return lower_limit, upper_limit


def GPTfunction(messages, max_tokens_count=350, text=False):
    openai.api_key = os.getenv("OPEN_AI_KEY")
    if text:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=messages,
            temperature=0.7,
            max_tokens=max_tokens_count,
        )
        message = response["choices"][0]["text"]
        # print(message)
        return message
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            # temperature=0.7,
            max_tokens=max_tokens_count,
        )
        message = response["choices"][0]["message"]["content"]
        # print(message)
        return message


def priorty(dataframe, dictionary, selected_fos):
    results = []
    df_copy = dataframe.copy()
    for i in range(len(dictionary), 0, -1):
        df_copy = dataframe.copy()

        sub_dict = dict(list(dictionary.items())[0:i])
        if len(sub_dict) > 0:
            comp_str = ""
            for i, key in enumerate(sub_dict):
                if i == len(sub_dict) - 1:
                    if i == 0:
                        comp_str += "(df_copy['{0}'].str.contains('|'.join({1}), case=False))".format(
                            key, sub_dict[key]
                        )
                    else:
                        if isinstance(sub_dict[key], (int, float)):
                            comp_str += "(df_copy['{0}']<={1})".format(
                                key, sub_dict[key]
                            )
                        else:
                            comp_str += "(df_copy['{0}'].str.lower()=='{1}')".format(
                                key, sub_dict[key]
                            )
                elif i <= len(sub_dict) - 2:
                    if i == 0:
                        comp_str += "(df_copy['{0}'].str.contains('|'.join({1}), case=False)) & ".format(
                            key, sub_dict[key]
                        )
                    else:
                        if isinstance(sub_dict[key], (int, float)):
                            comp_str += "(df_copy['{0}']<={1}) & ".format(
                                key, sub_dict[key]
                            )
                        else:
                            comp_str += "(df_copy['{0}'].str.lower()=='{1}') & ".format(
                                key, sub_dict[key]
                            )
                else:
                    if isinstance(sub_dict[key], (int, float)):
                        comp_str += "(df_copy['{0}']<={1})".format(key, sub_dict[key])
                    else:
                        comp_str += "(df_copy['{0}'].str.lower()=='{1}')".format(
                            key, sub_dict[key]
                        )
            # print(comp_str)
            df = df_copy[eval(comp_str)]

            counts = df["Title"].str.count("|".join(selected_fos), flags=re.IGNORECASE)

            df1 = (
                df.assign(score=counts)
                .sort_values("score", ascending=False)
                .drop("score", axis=1)
            )
            results.append(df1)

    final_result = pd.concat(results, ignore_index=True)
    final_finalResult = pd.concat([final_result, dataframe], ignore_index=True)

    final_finalResult.drop_duplicates(subset="_id", keep="first", inplace=True)
    lo_bhai = final_finalResult.filter(
        [
            "FieldOfStudy",
            "Province",
            "Institute",
            "Length",
            "IeltsOverall",
            "DuolingoOverall",
            "PteOverall",
            "Intake",
            "City",
            "Campus",
            "Title",
            "Level",
            "Fee",
        ],
        axis=1,
    )

    return final_finalResult


def calibre_checker(df, Imarks):
    noteligible = pd.DataFrame(columns=df.columns)

    # Iterate through each row of the original DataFrame
    for index, row in df.iterrows():
        # Check if the condition is satisfied
        if row["IeltsOverall"] > Imarks:
            # If not satisfied, drop the row and append it to the dropped DataFrame
            noteligible = pd.concat([noteligible, row.to_frame().T], ignore_index=True)
            df.drop(index, inplace=True)
    return df, noteligible


def cleandict(my_dict):
    for key in list(my_dict.keys()):
        # Check if the value of the key is an empty string, zero or None
        if my_dict[key] in ["", 0, None]:
            # If the value meets the specified criteria, remove the key
            del my_dict[key]
    return my_dict


def core_model(fos, level, college_df):
    if level != "false" or fos != "false":
        college_df["IeltsOverall"] = college_df["IeltsOverall"].astype(float).round(1)

        college_df["Length"] = (
            college_df["Length"]
            .astype("str")
            .str.extractall("(\d+)")
            .unstack()
            .fillna("")
            .sum(axis=1)
            .astype(int)
        )
        college_df["FeeText"] = college_df["Fee"]
        college_df["Fee"] = college_df.Fee.str.extract("(\d+)")

        college_df["Fee"] = college_df["Fee"].astype(int)
        college_df[
            ["FieldOfStudy", "Province", "InstituteName", "Title", "Level"]
        ] = college_df[
            ["FieldOfStudy", "Province", "InstituteName", "Title", "Level"]
        ].astype(
            str
        )

        college_df[["Length", "Fee"]] = college_df[["Length", "Fee"]].astype(int)

        comb_frame = (
            college_df["FieldOfStudy"].astype(str)
            + " "
            + college_df["Title"].astype(str)
            + " "
            + college_df["Level"].astype(str)
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(comb_frame)
        if level == None:
            Query_Input = fos
        else:
            Query_Input = fos + " " + level
        Input_transform = vectorizer.transform([Query_Input])

        cosine_similarities = sk.pairwise_kernels(Input_transform, X).flatten()

        related_docs_indices = cosine_similarities.argsort()[:-300:-1]

        df = pd.DataFrame(columns=college_df.columns)
        point = 0

        for index in related_docs_indices:
            program = college_df.iloc[index]
            temp = []
            for v in program.values:
                temp.append(v)
            df.loc[point] = temp
            point = point + 1
        return df
    else:
        return {"Error": "Please Select Atleast Single Option"}


@app.post("/api/recommend_courses", response_model=CourseResponse)
async def recommend_courses(request: CourseRequest):
    # print("recieved Request", request)
    start = time.time()
    my_dict = request.dictionary
    received_dictionary = {}
    for key in my_dict:
        new_key = key
        if key == "FieldOfStudy":
            new_key = "Title"
        elif key == "Budget":
            new_key = "Fee"
        elif key == "Duration":
            new_key = "Length"
        elif key == "Intake":
            new_key = "Seasons"
        received_dictionary[new_key] = my_dict[key]
    received_dictionary["Length"] = int(received_dictionary.get("Length", 0) or 0)
    if received_dictionary["Fee"] == "":
        received_dictionary["Fee"] = 0
    if received_dictionary["Fee"] != 0:
        fee_lower, fee_max = process_budget(received_dictionary["Fee"])
        received_dictionary["FeeLower"] = fee_lower
        received_dictionary["FeeMax"] = fee_max
        received_dictionary["Fee"] = int(received_dictionary["FeeMax"])
        received_dictionary.pop("FeeMax")
        received_dictionary.pop("FeeLower")

    received_dictionary.update(my_dict)
    received_dictionary["Length"] = int(received_dictionary["Length"])

    received_dictionary.pop("FieldOfStudy")
    received_dictionary.pop("Budget")
    received_dictionary.pop("Duration")
    received_dictionary.pop("Intake")
    value = fetch_all_data("test", "courses")
    college_df = pd.DataFrame(value)
    userdetails = fetch_all_data("test", "userdetails")
    selected_useremail = request.email
    selected_fos = received_dictionary["Title"]
    selected_level = received_dictionary["Level"].lower()

    if selected_fos == "":
        response = {"Error": "Please Select A Field Of Study"}
        # print(response)
        exit()
    openai.api_key = os.getenv("OPEN_AI_KEY")
    # messages = [{
    #     "role": "system",
    #     "content": "Generate a list of 18 courses related to " + selected_fos + " that are available at Canadian universities or colleges, including a mix of undergraduate, postgraduate, and certification programs if applicable."
    #     },
    #     {
    #     "role": "user",
    #     "content": "I'm interested in"+ selected_fos
    #     },]

    messages = (
        "Generate a list of 18 courses related to "
        + selected_fos
        + " that are available at Canadian universities or colleges, including a mix of undergraduate, postgraduate, and certification programs if applicable."
    )

    selected_fos = GPTfunction(messages, text=True) + " " + selected_fos

    title = selected_fos
    x = title.replace(",", " ")
    input_words = x.split()
    input_words = set(input_words)

    input_words = set(re.findall(r"[a-zA-Z]+", " ".join(input_words)))

    input_words = set([w.lower() for w in input_words])

    joining_words = {"and", "or", "for", "in", "the", "of", "on", "to", "a", "an"}
    input_words = input_words - joining_words

    selected_fos = input_words
    received_dictionary["Title"] = selected_fos

    received_dictionary["Level"] = received_dictionary["Level"].lower()
    received_dictionary["Province"] = received_dictionary["Province"].lower()
    received_dictionary["Seasons"] = received_dictionary["Seasons"].lower()
    dictionary = received_dictionary
    dictionary = cleandict(dictionary)

    selected_fos = " ".join(selected_fos)

    fill = core_model(selected_fos, selected_level, college_df)

    fill["_id"] = fill["_id"].astype(str)
    for i in range(len(fill["Intake"])):
        for j in range(len(fill["Intake"][i])):
            if fill["Intake"][i][j] is not None and "_id" in fill["Intake"][i][j]:
                del fill["Intake"][i][j]["_id"]

    intakes = []

    seasons = []
    statuses = []
    deadlines = []

    for d in fill["Intake"]:
        intake = d

        d_seasons = []
        d_statuses = []
        d_deadlines = []

        for i in intake:
            d_seasons.append(i["season"])
            d_statuses.append(i["status"])
            d_deadlines.append(i["deadline"])

        seasons.append(", ".join(d_seasons))
        statuses.append(", ".join(d_statuses))
        deadlines.append(", ".join(d_deadlines))

    intake_df = pd.DataFrame(
        {"Seasons": seasons, "Status": statuses, "Deadline": deadlines}
    )
    fill = pd.concat([fill.drop("Intake", axis=1), intake_df], axis=1)
    if len(dictionary) == 1:
        recommended_course_names = fill
    else:
        recommended_course_names = priorty(fill, dictionary, selected_fos)

    recommended_course_names.drop(
        [
            "__v",
            "Language",
            "IeltsReading",
            "IeltsWriting",
            "IeltsSpeaking",
            "IeltsListening",
            "PteReading",
            "PteWriting",
            "PteSpeaking",
            "PteListening",
            "Country",
        ],
        axis=1,
        inplace=True,
    )

    for u in userdetails:
        if u["email"] == selected_useremail:
            IELTS_O = u["Scores"]["IELTS"]["Overall"]
            break
    eligible, noteligible = calibre_checker(recommended_course_names, IELTS_O)

    eligible["Length"] = eligible["Length"].apply(
        lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months"
        if x >= 12
        else f"{x} Months"
    )
    noteligible["Length"] = noteligible["Length"].apply(
        lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months"
        if x >= 12
        else f"{x} Months"
    )
    eligible = eligible.reindex(
        columns=[
            "CreatedOn",
            "FieldOfStudy",
            "InstituteName",
            "Title",
            "Level",
            "Length",
            "ApplicationFee",
            "FeeText",
            "Seasons",
            "Status",
            "Deadline",
            "Percentage",
            "Backlog",
            "Gap",
            "Campus",
            "IeltsOverall",
            "PteOverall",
            "TOEFLOverall",
            "DuolingoOverall",
            "GRE",
            "GMAT",
            "City",
            "Province",
            "Visa Chances",
            "PR",
        ]
    )
    noteligible = noteligible.reindex(
        columns=[
            "CreatedOn",
            "FieldOfStudy",
            "InstituteName",
            "Title",
            "Level",
            "Length",
            "ApplicationFee",
            "FeeText",
            "Seasons",
            "Status",
            "Deadline",
            "Percentage",
            "Backlog",
            "Gap",
            "Campus",
            "IeltsOverall",
            "PteOverall",
            "TOEFLOverall",
            "DuolingoOverall",
            "GRE",
            "GMAT",
            "City",
            "Province",
            "Notes",
        ]
    )

    eligible.fillna("N/A", inplace=True)
    eligible = eligible.head(31)

    recommendData = eligible.to_dict("records")
    end = time.time()
    response_time = end - start
    print(response_time)
    # print(json_data)

    return CourseResponse(
        data={
            "message": "Course recommendations generated successfully",
            "recommended_courses": recommendData,
            "response_time": response_time,
        }
    )


def email_verification(receiver_emailID):
    email_sender = os.getenv("EMAIL")
    email_password = os.getenv("EMAIL_PSK")
    email_receiver = receiver_emailID
    subject = "Email Confirmation"
    string = "0123456789"
    OTP = ""
    varlen = len(string)
    for i in range(6):
        OTP += string[m.floor(r.random() * varlen)]

    Body = """
        Welcome to Root-on!
        THis is verfication code {0}
    """.format(
        OTP
    )

    em = EmailMessage()
    msgRoot = MIMEMultipart("related")
    msgRoot["Subject"] = "Confirm Your Email Address"
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
            <h1 style="color:#1d1c1d;font-size:36px;font-weight:700;margin:30px 0;padding:0;line-height:42px">Confirm your email address</h1>
            <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Your confirmation code is below - enter it in your open browser window and we'll help you get signed in.</p>
            <table style="width:100%;background:rgb(245, 244, 245);border-radius:4px;margin-right:50px;margin-bottom:30px;padding:43px 23px" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
              <tbody>
                <tr>
                  <td>
                    <p style="font-size:30px;line-height:24px;margin:16px 0;text-align:center;vertical-align:middle">{0}</p>
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
            OTP, datetime.datetime.now()
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

    return OTP


def is_email_present(email: str) -> bool:
    """Check if the email is already in the database."""
    database_name = "test"
    collection_name = "users"
    query = {"email": email}
    results = perform_database_operation(database_name, collection_name, "read", query)
    return len(results) > 0


def email_verification1(receiver_emailID, auth_id):
    email_sender = os.getenv("EMAIL")
    email_password = os.getenv("EMAIL_PSK")
    URLINTI = os.getenv("URLINTI")
    email_receiver = receiver_emailID
    subject = "Email Verification"

    OTP = f"{URLINTI}/ver-email?authId={auth_id}"

    Body = """
        Welcome to Root-on!
        This is your verfication link {0}
    """.format(
        OTP
    )

    em = EmailMessage()
    msgRoot = MIMEMultipart("related")
    msgRoot["Subject"] = "Confirm Your Email Address"
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
            <h1 style="color:#1d1c1d;font-size:36px;font-weight:700;margin:30px 0;padding:0;line-height:42px">Confirm your email address</h1>
            <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Your confirmation code is below - enter it in your open browser window and we'll help you get signed in.</p>
            <table style="width:100%;background:rgb(245, 244, 245);border-radius:4px;margin-right:50px;margin-bottom:30px;padding:43px 23px" align="center" border="0" cellPadding="0" cellSpacing="0" role="presentation">
              <tbody>
                <tr>
                  <td>
                    <p style="font-size:20px;line-height:28px;margin:16px 0;margin-bottom:30px">Click the link below to verify your Email:</p>
                    <a href="{0}" style="background-color:#000;color:#ffffff;display:inline-block;font-family:sans-serif;font-size:18px;line-height:44px;text-align:center;text-decoration:none;width:200px;border-radius:5px;margin-bottom:30px" target="_blank">Verification Link</a>
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
            OTP, datetime.datetime.now()
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


@app.post("/api/send-otp")
def send_otp(request: EmailRequest):
    try:
        if is_email_present(request.email):
            return {"Status": "Present", "Message": "Email already exists"}
        else:
            auth_id = secrets.token_hex(40)
            email_verification1(request.email, auth_id)
            hashed_password = bcrypt.hashpw(
                request.Password.encode("utf-8"), bcrypt.gensalt(10)
            )
            perform_database_operation(
                "test",
                "users",
                "create",
                {
                    "email": request.email,
                    "Firstname": request.Firstname,
                    "Lastname": request.Lastname,
                    "Phone": request.Phone,
                    "Password": hashed_password.decode("utf-8"),
                    "verified": False,
                    "authId": auth_id,
                },
            )
            return {
                "Status": "Success",
                "Message": "Verifiaction Mail Sent Into Your Mailbox",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error sending email")


@app.post("/api/verification")
def verification(request: AuthRequest):
    try:
        result = perform_database_operation(
            "test", "users", "read", {"authId": request.authId}
        )
        if result:
            perform_database_operation(
                "test",
                "users",
                "update",
                {"authId": request.authId},
                {"unset": {"authId": ""}, "verified": True},
            )
            return {"Status": "Success", "Message": "Verification Successful"}
        else:
            return {
                "Status": "Failure",
                "Message": "Verification Failed. User not found or already verified.",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Verfication Failed")


@app.post("/api/login")
def login(request: LoginRequest):
    try:
        users = perform_database_operation(
            "test", "users", "read", {"email": request.email}
        )

        # Check if any user is found
        if users and len(users) > 0:
            user_data = users[0]  # Get the first record

            # Check if the user is verified
            if user_data.get("verified"):
                # Verify the password
                if bcrypt.checkpw(
                    request.Password.encode("utf-8"),
                    user_data["Password"].encode("utf-8"),
                ):
                    return {"Status": "Success", "Message": "Login successful."}
                else:
                    raise HTTPException(status_code=401, detail="Invalid password")
            else:
                raise HTTPException(status_code=403, detail="User not verified")
        else:
            raise HTTPException(status_code=404, detail="User not found")

    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        raise HTTPException(status_code=500, detail=f"Login Failed: Unexpected error")


@app.post("/api/visa-pr-prob", response_model=CourseResponse)
def visa_pr_prob(request: CourseRequest):
    try:
        # Call your visa_pr_prob function here
        messages = [
            {
                "role": "system",
                "content": "Generate a list of 18 courses related to "
                " that are available at Canadian universities or colleges, including a mix of undergraduate, postgraduate, and certification programs if applicable.",
            },
            {"role": "user", "content": "I'm interested in" + request},
        ]

        return visa_pr_prob(request.fos, request.level, request.email)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing request")


if __name__ == "__main__":
    import uvicorn

    print("Starting webserver...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("DEBUG", False),
        log_level=os.getenv('LOG_LEVEL', "info"),
        proxy_headers=True
    )
