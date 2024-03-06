from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import pandas as pd
import logging
import pymongo
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise as sk
import json
import re
import openai
import authlib
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
from datetime import datetime, timedelta
import secrets
import bcrypt
from bson import ObjectId
from jose import JWTError, jwt
from fastapi import Security
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from base64 import b64decode


load_dotenv()

cache = TTLCache(maxsize=1000000, ttl=86400)

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
FRONTEND_URL = os.getenv("FRONTEND_URL")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(SessionMiddleware, secret_key=os.getenv("Session_SECRET_KEY"))

# Pydantic models for request and response
class CourseRequest(BaseModel):
    fos: str
    level: str
    dictionary: dict
    email: str
    LanguageProficiency:str
    Score:float | str

class VisaPRRequest(BaseModel):
    email: str
    dictionary: dict
    ask: str

class PromptItem(BaseModel):
    role: str
    content: str

class SOPSOWPRequest(BaseModel):
    prompt: List[PromptItem]
    maxtoken: int 
    model: str

class EncryptedRequest(BaseModel):
    encryptedData: str

class CourseResponse(BaseModel):
    data: dict


class EmailRequest(BaseModel):
    email: str
    Firstname: str
    Lastname: str
    # EmailId: str
    Phone: str
    Password: str
    Second: bool


class AuthRequest(BaseModel):
    authId: str


class LoginRequest(BaseModel):
    email: str
    Password: str

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
    
class ProfileInfoRequest(BaseModel):
    email: str
    profileInfo: dict

class ProfileInfoResponse(BaseModel):
    responsedata: dict

# Function to fetch all data (similar to your script)
@cached(cache)
def fetch_all_data(database, collection):
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(MONGODB_URI)

    db = client[database]
    collection = db[collection]

    cursor = collection.find({})
    counter = 0
    results = []
    try:
        for document in cursor:
            counter += 1
            # print(counter)
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
        # Silently remove email from update_data if present to prevent email changes
        update_data.pop("email", None)
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
        # Ensure email uniqueness at the collection level
        collection.create_index([("email", pymongo.ASCENDING)], unique=True)
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
            query["Role"] = "User"
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

# Secret key to encode the JWT token (should be kept secret)
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # the expiration time for the token
ACCESS_TOKEN_EXPIRE_DAYS = 365*30  # the expiration time for the token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    # Adding this line to use the OIDC discovery document
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@app.get("/api/login/google")
async def login_google(request: Request):
    # referer = request.headers.get("referer")
    # # Check if referer is None
    # if referer is None:
    #     # Handle the case where referer is missing
    #     # You can raise an HTTPException or handle it in another appropriate way
    #     raise HTTPException(status_code=403, detail="Access denied")
    
    # # Normalizing the referer by stripping the trailing slash if it exists
    # normalized_referer = referer.rstrip('/')
    # # Check if the referer is from the allowed origins
    # if normalized_referer not in allowed_origins:
    #     raise HTTPException(status_code=403, detail="Access denied")
    # The state is saved in the session in this call by default
    redirect_uri = request.url_for('authorize_google')
    # print(redirect_uri)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/api/authorize/google')
async def authorize_google(request: Request):
    try:
        # Authlib checks the state parameter against the session automatically
        token = await oauth.google.authorize_access_token(request)
        # If we get here, the state check has passed
        user_data = token.get('userinfo')
        if user_data:
        # Handle user login or registration here
            users = perform_database_operation(
                "test", "users", "read", {"email": user_data["email"]}
            )

            # Check if any user is found
            if users and len(users) > 0:
                token_data = {"Email": user_data["email"], "FirstName": user_data.get("given_name", ""),
                    "LastName": user_data.get("family_name", ""), "type": "access"}
                refresh_token_data = {"Email": user_data["email"],"FirstName": user_data.get("Firstname", ""),
                "LastName": user_data.get("Lastname", ""), "type": "refresh"}
                # Set the token to expire in 30 years
                access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
                response_token_expires = timedelta(minutes=1)
                # Generate the JWT token
                access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
                refresh_token = create_refresh_token(data=refresh_token_data)
                return_token = create_access_token(data={"access_token": access_token,"refresh_token": refresh_token, "token_type": "bearer"}, expires_delta=response_token_expires)
                # Create a query string with the token data
                return RedirectResponse(url=FRONTEND_URL + "/googleauth?token=" + return_token)
            else:
                perform_database_operation(
                    "test",
                    "users",
                    "create",
                    {
                        "email": user_data["email"],
                        "Firstname": user_data.get("given_name", ""),
                        "Lastname": user_data.get("family_name", ""),
                        "Phone": "",
                        "Password": "",
                        "verified": True,
                        # "authId": auth_id,
                    },
                )
                perform_database_operation(
                    "test",
                    "userdetails",
                    "create",
                    {
                        "email": user_data["email"],
                        "Firstname": user_data.get("given_name", ""),
                        "Lastname": user_data.get("family_name", ""),
                        "Phone": "",
                        "profileFilled": False,
                        # "authId": auth_id,
                    },
                )
                token_data = {"Email": user_data["email"], "FirstName": user_data.get("given_name", ""),
                    "LastName": user_data.get("family_name", ""), "type": "access"}
                refresh_token_data = {"Email": user_data["email"],"FirstName": user_data.get("Firstname", ""),
                "LastName": user_data.get("Lastname", ""), "type": "refresh"}
                # Set the token to expire in 30 years
                access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
                # Generate the JWT token
                access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
                refresh_token = create_refresh_token(data=refresh_token_data)
                response_token_expires = timedelta(minutes=1)
                return_token = create_access_token(data={"access_token": access_token,"refresh_token": refresh_token, "token_type": "bearer"}, expires_delta=response_token_expires)
                # Create a query string with the token data
                return RedirectResponse(url=FRONTEND_URL + "/googleauth?token=" + return_token)
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except authlib.integrations.base_client.errors.MismatchingStateError:
        # The state parameter does not match the session state
        raise HTTPException(status_code=400, detail="State mismatch error. Possible CSRF attack.")
    except Exception as e:
        # Log the error and return a generic error response to the user
        logging.exception("An error occurred during the OAuth callback.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.", err=str(e))

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("Email")
        token_type: Optional[str] = payload.get("type")
        if email is None or token_type != "access":
            raise credentials_exception
        # You could add more user verification here if needed
        return email
    except JWTError:
        raise credentials_exception

def process_budget(budget_str):
    """Extract the lower and upper fee limits from the budget string."""
    if "to above" in budget_str:
        lower_limit = int(budget_str.split(" ")[0])
        upper_limit = int("999999")
    else:
        lower_limit, upper_limit = map(int, budget_str.split("-"))
    return lower_limit, upper_limit


def GPTfunction(messages, max_tokens_count=350, text=False, usedmodel="gpt-4"):
    openai.api_key = os.getenv("OPEN_AI_KEY")
    if text:
        response = openai.Completion.create(
            model=usedmodel,
            prompt=messages,
            temperature=0.7,
            max_tokens=max_tokens_count,
        )
        message = response["choices"][0]["text"]
        # print(message)
        return message
    else:
        response = openai.ChatCompletion.create(
            model=usedmodel,
            messages=messages,
            # temperature=0.7,
            max_tokens=max_tokens_count,
        )
        message = response["choices"][0]["message"]["content"]
        # print(message)
        return message


def priority1(dataframe, dictionary, selected_fos):
    results = []
    df_copy = dataframe.copy()
    # print(dictionary)
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
            # print(df)

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


def priority(dataframe, dictionary, selected_fos):
    try:
        results = []
        for i in range(len(dictionary), 0, -1):
            df_copy = dataframe.copy()
            print(f"Processing dictionary with {i} items.")  # Shows the iteration step

            sub_dict = dict(list(dictionary.items())[0:i])
            for key, value in sub_dict.items():
                print(f"Filtering for {key} with value: {value}")  # Shows the current filtering condition

                if isinstance(value, str) and ',' in value:
                    values = [v.strip() for v in value.split(',')]
                    print(f"Applying filter on {key} for any of {values}")  # Shows the filter condition
                    df_copy = df_copy[df_copy[key].isin(values)]
                elif isinstance(value, set):
                    pattern = '|'.join([re.escape(item) for item in value])
                    print(f"Applying regex filter on {key} for pattern: {pattern}")  # Shows the regex being used
                    df_copy = df_copy[df_copy[key].str.contains(pattern, case=False)]
                elif isinstance(value, (int, float)):
                    print(f"Applying numerical filter on {key} for value <= {value}")  # Shows the numerical condition
                    df_copy = df_copy[df_copy[key] <= value]
                else:
                    print(f"Applying exact string match filter on {key} for: {value}")  # Shows the string comparison
                    df_copy = df_copy[df_copy[key].str.lower() == value.lower()]

            if not df_copy.empty:
                counts = df_copy["Title"].str.count("|".join([re.escape(fos) for fos in selected_fos]), flags=re.IGNORECASE)
                df_copy = df_copy.assign(score=counts).sort_values('score', ascending=False).drop('score', axis=1)
                results.append(df_copy)

        final_result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        final_result = pd.concat([final_result, dataframe], ignore_index=True).drop_duplicates(subset="_id", keep="first")

        # No need to filter columns here as we want to return the entire final_result for debugging
        return final_result
    except Exception as e:
        print(f"Error processing request in priority: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request in priority: {e}")
# def calibre_checker(df, Imarks):  
#     noteligible = pd.DataFrame(columns=df.columns)

#     # Iterate through each row of the original DataFrame
#     for index, row in df.iterrows():
#         # Check if the condition is satisfied
#         if row["IeltsOverall"] > Imarks:
#             # If not satisfied, drop the row and append it to the dropped DataFrame
#             noteligible = pd.concat([noteligible, row.to_frame().T], ignore_index=True)
#             df.drop(index, inplace=True)
#     return df, noteligible

def calibre_checker(df: pd.DataFrame, language_proficiency, my_marks):
    try:
        noteligible = pd.DataFrame(columns=df.columns)
        # print("Duplicates in original df:", df.duplicated().sum())
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)


        # Mapping of language proficiency tests to their respective column names in the DataFrame
        proficiency_column_mapping = {
            "IELTS": "IeltsOverall",
            "Duolingo": "DuolingoOverall",
            "PTE": "PteOverall",
            "GMAT": "GMAT",
            "GRE": "GRE",
            "TOEFL": "TOEFLOverall"
        }
        # print("Intial --> ",len(df))
        # Determine the column to use for comparison based on the language proficiency provided
        score_column = proficiency_column_mapping.get(language_proficiency, "IeltsOverall")  # Default to IELTS if not found
        # Create a mask to filter eligible and not eligible entries
        # Note: Changed the logic to correctly identify not eligible entries (scores below my_marks)
        not_eligible_mask = df[score_column].isnull() | (df[score_column] > my_marks)
        
        # Filter based on the mask
        noteligible = df[not_eligible_mask]
        eligible = df[~not_eligible_mask]
        # print(df[~df.index.isin(noteligible.index)])
        
        # print("Eligibile --> ",len(eligible))
        # print("Not Eligible --> ",len(noteligible))

        # print("Duplicates in eligible:", eligible.duplicated().sum())
        # print("Duplicates in noteligible:", noteligible.duplicated().sum())


        return eligible, noteligible
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request : {e}")

def cleandict(my_dict):
    for key in list(my_dict.keys()):
        # Check if the value of the key is an empty string, zero or None
        if my_dict[key] in ["", 0, None,"Any Province"]:
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
        comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(comb_frame)
        if level == None:
            Query_Input = fos
        else:
            Query_Input = fos + " " + level
        Input_transform = vectorizer.transform([Query_Input])

        pairwise_kernels = sk.pairwise_kernels(Input_transform, X).flatten()

        # # Modify this part to include weights
        pairwise_kernels = pairwise_kernels

        related_docs_indices = pairwise_kernels.argsort()[:-59:-1]

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
async def recommend_courses(request: CourseRequest, email: str = Depends(get_current_user)):
    try:
        print("recieved Request", request)
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
        creation_df = pd.DataFrame(value)

        # college_df = creation_df[creation_df['Level'] == received_dictionary["Level"]]
        # college_df = creation_df[(creation_df['Level'] == received_dictionary["Level"]) & (creation_df['Province'] != 'Ontario')]
        levels = received_dictionary["Level"].split(', ')
        if received_dictionary["Province"] == "Any Province" or received_dictionary["Province"] == "":
            provinces = ["Alberta","British Columbia","Manitoba","New Brunswick","Newfoundland and Labrador",
                         "Northwest Territories","Nova Scotia","Ontario","Prince Edward Island",
                         "Quebec","Saskatchewan","Yukon Territory"]
            college_df = creation_df[(creation_df['Level'].isin(levels))]
        else:
            provinces = received_dictionary["Province"].split(', ')
            college_df = creation_df[(creation_df['Level'].isin(levels)) & (creation_df['Province'].isin(provinces))]


        # userdetails = perform_database_operation(
        #     "test", "userdetails", "read", {"email": email}
        # )
        selected_fos = received_dictionary["Title"]
        selected_level = received_dictionary["Level"].lower()

        if selected_fos == "":
            response = {"Error": "Please Select A Field Of Study"}
            raise HTTPException(status_code=500, detail=response)
        
        openai.api_key = os.getenv("OPEN_AI_KEY")
        # messages = [{
        #     "role": "system",
        #     "content": "Generate a list of 18 courses related to " + selected_fos + " that are available at Canadian universities or colleges, including a mix of undergraduate, postgraduate, and certification programs if applicable."
        #     },
        #     {
        #     "role": "user",
        #     "content": "I'm interested in"+ selected_fos
        #     },]


        # messages = "List 18 courses related to " + selected_fos + " that can be studied in canada"
        # # messages = (
        # #     "Generate a list of 18 courses related to "
        # #     + selected_fos
        # #     + " that are available at Canadian universities or colleges, including a mix of undergraduate, postgraduate, and certification programs if applicable."
        # # )

        # selected_fos = GPTfunction(messages, text=True, max_tokens_count=3000, usedmodel="gpt-3.5-turbo-instruct") + " " + selected_fos
        messages = "Give me relevant keywords around " + selected_fos +", also try to give me words which are spelled different in the world related to "+ selected_fos +" (this is just one of the example, Jewellery, Jewelery) also try to break and concate word and make a cluster of relevent keywords in canada"
        selected_fos = GPTfunction(messages, text=True, max_tokens_count=3000, usedmodel="gpt-3.5-turbo-instruct") + " " + selected_fos
        # print("selected_fos", selected_fos)

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
        print("selected_fos", selected_fos)

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
            recommended_course_names = priority1(fill, dictionary, selected_fos)
            # recommended_course_names = fill

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

        # if LanguageProficiency:
        #     if userdetails[0]["Scores"]["IELTS"]["Overall"]:
        #         IELTS_O = userdetails[0]["Scores"]["IELTS"]["Overall"]
        #     else:
                
        #         IELTS_O = 6.5 #Conditioning is remaining...
        # else:
        #     IELTS_O = 6.5
        if (request.LanguageProficiency!=""):
            eligible1, noteligible1 = calibre_checker(recommended_course_names, request.LanguageProficiency,request.Score)

            # print("Duplicates in eligible: 1", eligible1.duplicated().sum())
            # print("Duplicates in noteligible: 3", noteligible1.duplicated().sum())


            eligible1["Length"] = eligible1["Length"].apply(
                lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months"
                if x >= 12
                else f"{x} Months"
            )
            noteligible1["Length"] = noteligible1["Length"].apply(
                lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months"
                if x >= 12
                else f"{x} Months"
            )
            eligible1 = eligible1.reindex(
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
                    "InstituteCategory"
                ]
            )
            noteligible1 = noteligible1.reindex(
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

            eligible1.fillna("N/A", inplace=True)
            eligible1 = eligible1.head(31)

            noteligible1.fillna("N/A", inplace=True)
            # eligible1.to_csv("recommendations.csv", index=False, header=True)
            noteligible1 = noteligible1.head(31)

            recommendData = eligible1.to_dict("records")
            end = time.time()
            response_time = end - start
            print(response_time)
            # print(json_data)

            return CourseResponse(
                data={
                    "message": "Course recommendations generated successfully",
                    "eligible": recommendData,
                    "noteligible": noteligible1.to_dict("records"),
                    "response_time": response_time,
                }
            )
        else:
            eligible1 = recommended_course_names
            eligible1["Length"] = eligible1["Length"].apply(
                lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months"
                if x >= 12
                else f"{x} Months"
            )
            eligible1 = eligible1.reindex(
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
                    "InstituteCategory"
                ]
            )

            eligible1.fillna("N/A", inplace=True)
            eligible1 = eligible1.head(31)

            recommendData = eligible1.to_dict("records")
            end = time.time()
            response_time = end - start
            print(response_time)
            # print(json_data)

            return CourseResponse(
                data={
                    "message": "Course recommendations generated successfully",
                    "eligible": recommendData,
                    "noteligible": [],
                    "response_time": response_time,
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request : {e}")

def is_email_present(email: str) -> bool:
    """Check if the email is already in the database."""
    database_name = "test"
    collection_name = "users"
    query = {"email": email}
    results = perform_database_operation(database_name, collection_name, "read", query)
    return len(results) > 0

def email_verification(receiver_emailID, auth_id):
    email_sender = os.getenv("EMAIL")
    email_password = os.getenv("EMAIL_PSK")
    URLINTI = FRONTEND_URL
    email_receiver = receiver_emailID
    subject = "Email Verification"

    OTP = f"{URLINTI}/verification-email?authId={auth_id}"

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
            OTP, datetime.now()
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
        if request.Second:
            auth_id = secrets.token_hex(40)
            users = perform_database_operation("test", "users", "read", {"email": request.email})
            # Check if any user is found
            if users and len(users) > 0:
                user_data = users[0]  # Get the first record
                # Check if the user is verified
                if user_data.get("verified"):
                    return {"Status": "Verified", "Message": f"The email address '{request.email}' is already verified."}
                else:
                    email_verification(request.email, auth_id)
                    perform_database_operation("test","users","update",{"email": request.email},{"authId": auth_id})
                    return {
                        "Status": "Success",
                        "Message": "Email Verification Mail Resent Into Your Mailbox",
                    }
            else:
                return {"Status": "Error", "Message": "User not found"}   
        elif is_email_present(request.email):
            return {"Status": "Present", "Message": "Email already exists"}
        else:
            auth_id = secrets.token_hex(40)
            email_verification(request.email, auth_id)
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
                "Message": "Email Verification Mail Sent Into Your Mailbox",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error sending email", err=str(e))

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
            perform_database_operation(
                    "test",
                    "userdetails",
                    "create",
                    {
                        "email": result[0]['email'],
                        "Firstname": result[0]['Firstname'],
                        "Lastname": result[0]['Lastname'],
                        "Phone": result[0]['Phone'],
                        "profileFilled": False,
                        # "authId": auth_id,
                    },
                )
            return {"Status": "Success", "Message": "Verification Successful"}
        else:
            return {
                "Status": "Failure",
                "Message": "Verification Failed. User not found or already verified.",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verfication Failed: {e}")

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Function to create a refresh token
def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)  # refresh tokens usually have a long expiry time
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/api/token/refresh")
def refresh_token(refresh_token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise credentials_exception
        email: str = payload.get("Email")
        if email is None:
            raise credentials_exception
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_DAYS)
        new_access_token = create_access_token(data={"Email": email,"FirstName": payload.get("FirstName", ""),"LastName": payload.get("LastName", ""), "type": "access"},expires_delta=access_token_expires)
        return {"access_token": new_access_token, "token_type": "bearer"}
    except JWTError:
        raise credentials_exception

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
                if bcrypt.checkpw(request.Password.encode("utf-8"), user_data["Password"].encode("utf-8")):
                    
                    # Create token data payload
                    token_data = {"Email": user_data["email"], "FirstName": user_data.get("Firstname", ""),
                        "LastName": user_data.get("Lastname", ""), "type": "access"}
                    refresh_token_data = {"Email": user_data["email"],"FirstName": user_data.get("Firstname", ""),
                        "LastName": user_data.get("Lastname", ""), "type": "refresh"}
                    
                    # Set the token to expire in 60 minutes
                    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS) # 30 years into the future

                    # Generate the JWT token
                    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
                    refresh_token = create_refresh_token(data=refresh_token_data)
                    return {"access_token": access_token,"refresh_token": refresh_token, "token_type": "bearer"}
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
        raise HTTPException(status_code=500, detail=f"Login Failed: {e}")

@app.get("/api/profile-info")
def profile_info(email: str = Depends(get_current_user)):
    try:
        users = perform_database_operation(
            "test", "userdetails", "read", {"email": email}
        )
        if users and len(users) > 0:
            # Serialize the data using the custom JSON encoder and return it as a JSON response
            json_compatible_item_data = json.loads(json.dumps(users[0], cls=CustomJSONEncoder))
            return JSONResponse(content=json_compatible_item_data)
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        raise HTTPException(status_code=500, detail=f"Profile Info Failed: {e}")
    
@app.put("/api/update-profile-info")
def profile_info(request: ProfileInfoRequest, email: str = Depends(get_current_user)):
    try:
        users = perform_database_operation(
            "test", "userdetails", "update", {"email": email}, request.profileInfo
        )
        if users==1:
            return{"Status": "Success", "Message":"Profile Info Updated" }
        else:
            raise HTTPException(status_code=404, detail="User not found OR There Nothing To Update")
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        raise HTTPException(status_code=500, detail=f"Profile Info Failed: {e}")


@app.post("/api/visa-pr-prob")
def visa_pr_prob(request: VisaPRRequest, email: str = Depends(get_current_user)):
    try:
        users = perform_database_operation(
            "test", "userdetails", "read", {"email": request.email}
        )
        if users and len(users) > 0:
            user_data = users[0]  # Get the first record
            # Check if the user is verified
            stringUser = str(user_data)
            stringCourse = str(request.dictionary)
            if user_data:
                if request.ask == "Visa":
                    messages = [{
                    "role": "system",
                    "content": "Based on the provided user profile and course details, assess the chances of obtaining a Canadian Visa for the user. Consider factors such as the user's academic background, test scores, work experience, and chosen program of study. The course and user profile data are dynamic and should be evaluated in the context of current immigration policies and program requirements. ALWAYS REMEMBER you have to answer only in single word Low, Medium, Also provide reason in two lines only"
                    },
                    {
                    "role": "user",
                    "content": "Given the course details: "+stringCourse+" and my profile: "+stringUser+" , what are my chances of getting a Canadian Visa?"
                            }]
                    result = GPTfunction(messages, text=False)
                    request.dictionary["Visa Chances"] = result
                    return {"Status": "Success", "Message": request.dictionary}
                    
                elif request.ask == "PR":
                    messages = [{
                    "role": "system",
                    "content": "Based on the provided user profile and course details, assess the chances of obtaining a Canadian PR for the user. Consider factors such as the user's academic background, test scores, work experience, and chosen program of study. The course and user profile data are dynamic and should be evaluated in the context of current immigration policies and program requirements. ALWAYS REMEMBER you have to answer only in single word High, Medium, or Low, Not even a single word extra."
                    },
                    {
                    "role": "user",
                    "content": "Given the course details: "+stringCourse+" and my profile: "+stringUser+" , what are my chances of getting a Canadian PR?"
                            }]
                    result = GPTfunction(messages, text=False)
                    request.dictionary["PR Chances"] = result
                    return {"Status": "Success", "Message": request.dictionary}
                else:
                    raise HTTPException(status_code=400, detail="Invalid request")
            else:
                raise HTTPException(status_code=403, detail="Complete your user profile first")
            
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request : {e}")

def decrypt_with_aes(ciphertext, secret_key):
    key = b64decode(secret_key)
    iv, ct = b64decode(ciphertext[:24]), b64decode(ciphertext[24:])
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size).decode('utf-8')
    return pt

@app.post("/api/sop-sowp-builder")
def sop_sowp_builder(request: EncryptedRequest, email: str = Depends(get_current_user)):
    try:
        # secret_key = os.getenv("NEXT_ENCRYPTION_KEY")
        secret_key = 'pvuM6lsaQHCIEZHD3bO5GFAak0NlWcWc4EvkQ9y+ysg='
        # print(secret_key==secret_key1)
        decrypted = decrypt_with_aes(request.encryptedData, secret_key)
        # print(f'Decrypted: {decrypted}')
        # print(type(decrypted))
        decrypted_data = json.loads(decrypted)
        messages = [{"role": item["role"], "content": item["content"]} for item in decrypted_data["prompt"]]
        result = GPTfunction(messages, usedmodel=decrypted_data["model"], max_tokens_count=decrypted_data["maxtoken"])
        return {"Status": "Success", "Letter": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request : {e}")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/api/dropdowns")
def get_dropdowns(email: str = Depends(get_current_user)):
    try:
        # Perform database operations
        value = fetch_all_data("test","courses")
        college_df = pd.DataFrame(value)
        unique_level = sorted(set(college_df["Level"].values))
        unique_fos = sorted(set(college_df["FieldOfStudy"].values))
        unique_province = sorted(set(college_df["Province"].values))
        # unique_level.insert(0, "")
        # unique_fos.insert(0, "")
        unique_province.insert(0, "Any Province")
        return {"Status": "Success", "Level": unique_level, "FieldOfStudy": unique_fos, "Province": unique_province}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request : {e}")


@app.get('/api/userRole')
def getUserRole(email: str = Depends(get_current_user)):
    try:
        users = perform_database_operation(
            "test", "userdetails", "read", {"email": email}
        )
        if users and len(users) > 0:
            # Serialize the data using the custom JSON encoder and return it as a JSON response
            return ({"Role":users[0]['Role']})
        else:
            raise HTTPException(status_code=404, detail="Invalid Token")
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        raise HTTPException(status_code=500, detail=f"Role Info Fetch Failed: {e}")


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