from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import pandas as pd
import logging
import requests
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
from dotenv import load_dotenv
from datetime import datetime, timedelta
from utilities import resetpasswordmail
from utilities import email_verification
import secrets
import bcrypt
from jose import JWTError, jwt
from fastapi import Security
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from authlib.integrations.starlette_client import OAuth, OAuthError
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from base64 import b64decode
from starlette.datastructures import URL
import traceback
import requests
from models import *
from utilities.emails.satbulkmail import satbulkmail
from utilities.apithird.docusealapi import get_docuseal_templates_fn
from utilities.helperfunc.dbfunc import perform_database_operation
from utilities.helperfunc.slugfinder import get_slug_value
import base64


load_dotenv()

cache = TTLCache(maxsize=1000000, ttl=86400)

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
FRONTEND_URL = os.getenv("FRONTEND_URL")
additional_origin = os.getenv("ADDITIONAL_ORIGIN", "")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(SessionMiddleware, secret_key=os.getenv("Session_SECRET_KEY"))

# Pandas ki SettingWithCopyWarning ko globally suppress karna
pd.options.mode.chained_assignment = None  # default='warn'

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
            results.append(document)
    finally:
        cursor.close()
        client.close()

    return results

async def preload_cache():
    try:
        # Preload data for each collection
        print("caching started")
        fetch_all_data("test", "courses")
        print("caching done 1")
        # Add more collections as needed
    except Exception as e:
        print(f"Error during cache preloading: {e}")


app.add_event_handler("startup", preload_cache)

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponseModel(
            detail="We're having trouble processing your request right now. Please try again later.",
            err=str(exc)
        ).dict(),
    )


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
    referer = request.headers.get("referer")
    # Check if referer is None
    if referer is None:
        # Handle the case where referer is missing
        # You can raise an HTTPException or handle it in another appropriate way
        raise HTTPException(status_code=403, detail="Sorry, you don't have permission to access this. Please check your access rights or contact support for help")
    
    # Normalizing the referer by stripping the trailing slash if it exists
    normalized_referer = referer.rstrip('/')
    # Check if the referer is from the allowed origins
    if normalized_referer not in allowed_origins:
        raise HTTPException(status_code=403, detail="Sorry, you don't have permission to access this. Please check your access rights or contact support for help")
    
    # Store the FRONTEND_URL in the session
    request.session['frontend_url'] = normalized_referer
    # global FRONTEND_URL
    # FRONTEND_URL = normalized_referer
    # # Store the FRONTEND_URL in the session
    # request.session['frontend_url'] = FRONTEND_URL

    # Obtain the redirect URI as a URL object
    redirect_uri: URL = request.url_for('authorize_google')
    
    # Enforce HTTPS if we are not in a development environment (e.g., localhost)
    if 'localhost' not in redirect_uri.netloc and os.getenv('ENVIRONMENT') == 'production':
        redirect_uri = redirect_uri.replace(scheme='https')

    # Pass the redirect_uri to authorize_redirect without converting it to string
    # The state is saved in the session in this call by default
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/api/authorize/google')
async def authorize_google(request: Request):
    try:
        frontend_url = request.session.get('frontend_url')
        if not frontend_url:
            frontend_url = FRONTEND_URL
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
                return RedirectResponse(url=frontend_url + "/googleauth?token=" + return_token)
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
                    signup=True
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
                    },signup=True
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
                return RedirectResponse(url=frontend_url + "/googleauth?token=" + return_token)
        else:
            raise HTTPException(status_code=401, detail="Your login details didn't match our records. Please check and try again.")

    except authlib.integrations.base_client.errors.MismatchingStateError:
        # The state parameter does not match the session state
        raise HTTPException(status_code=400, detail="State mismatch error. Possible CSRF attack.")
    except Exception as e:
        # Log the error and return a generic error response to the user
        logging.exception("An error occurred during the OAuth callback.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

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


def GPTfunction(messages, max_tokens_count=350, text=False, usedmodel="gpt-4o"):
    try:
        openai.api_key = os.getenv("OPEN_AI_KEY")
        if text:
            response = openai.Completion.create(
                model=usedmodel,
                prompt=messages,
                temperature=0.7,
                max_tokens=max_tokens_count,
            )
            message = response["choices"][0]["text"]
            return message
        else:
            response = openai.ChatCompletion.create(
                model=usedmodel,
                messages=messages,
                # temperature=0.7,
                max_tokens=max_tokens_count,
            )
            message = response["choices"][0]["message"]["content"]
            return message
    except Exception as e:
        print(e)
        raise HTTPException(status_code=429, detail="LLM Function Error")
        


def priority(dataframe, dictionary, selected_fos):
    results = []
    dictionary = {k: v for k, v in dictionary.items() if k not in ['Level', 'Province']}
    df_copy = dataframe.copy()
    string_columns = list(dictionary.keys())
    string_columns = [key for key in string_columns if key not in ['Fee', 'Length']]
    # non_string_columns = [col for col in string_columns if df_copy[col].dtype != object]
    # print("Non-string columns:", non_string_columns)
    for column in string_columns:
        df_copy[column] = df_copy[column].astype(str)
    for i in range(len(dictionary), 0, -1):
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

    return final_finalResult


def calibre_checker(df: pd.DataFrame, language_proficiency, my_marks):
    try:
        
        noteligible = pd.DataFrame(columns=df.columns)
        df = df.drop_duplicates(subset="_id", keep="first")
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
        # Determine the column to use for comparison based on the language proficiency provided
        score_column = proficiency_column_mapping.get(language_proficiency, "IeltsOverall")  # Default to IELTS if not found
        # Create a mask to filter eligible and not eligible entries
        # Note: Changed the logic to correctly identify not eligible entries (scores below my_marks)
        not_eligible_mask = df[score_column].isnull() | (df[score_column] > my_marks)
        
        # Filter based on the mask
        noteligible = df[not_eligible_mask]
        eligible = df[~not_eligible_mask]

        return eligible, noteligible
    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Error processing request in Calibre Checker: {e}\nTraceback: {traceback_str}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

def cleandict(my_dict):
    for key in list(my_dict.keys()):
        # Check if the value of the key is an empty string, zero or None
        if my_dict[key] in ["", 0, None,"any province"]:
            # If the value meets the specified criteria, remove the key
            del my_dict[key]
    return my_dict

def data_preciser(received_dictionary, creation_df):
    try:
        levels = received_dictionary["Level"].split(', ')
        language = ["English"]
        if received_dictionary["Province"] == "Any Province" or received_dictionary["Province"] == "":
                provinces = ["Alberta","British Columbia","Manitoba","New Brunswick","Newfoundland and Labrador",
                             "Northwest Territories","Nova Scotia","Ontario","Prince Edward Island",
                             "Quebec","Saskatchewan","Yukon Territory"]
                new_college_df = creation_df[(creation_df['Level'].isin(levels)) & creation_df['Language'].isin(language)]
        else:
            provinces = received_dictionary["Province"].split(', ')
            new_college_df = creation_df[(creation_df['Level'].isin(levels)) & (creation_df['Province'].isin(provinces)) & creation_df['Language'].isin(language)]
        new_college_df["Level1"]=received_dictionary["Level"]
        new_college_df["Province1"]=received_dictionary["Province"]
        return pd.DataFrame(new_college_df)
    except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f"Error processing request in Data Preciser: {e}\nTraceback: {traceback_str}")
            print(f"Error processing request in Data Preciser: {e}")
            raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

def core_model(fos, level, college_df, gptifcondition=False):
    if level != "false" or fos != "false":
        if gptifcondition==False:
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
                new_key = "Seasons1"
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

        if request.partneredtoggle:
            creation_df = pd.DataFrame(value)
            creation_df = creation_df[creation_df["InstituteCategory"].isin(['Direct', 'In Direct'])]
        else:
            creation_df = pd.DataFrame(value)

        if received_dictionary["Seasons1"]=="":
            college_df = data_preciser(received_dictionary,creation_df)
        else:
            intake_df = intake_preciser(received_dictionary["Seasons1"], received_dictionary,creation_df).copy()
            data_df = data_preciser(received_dictionary, creation_df).copy()

            intake_df['_id'] = intake_df['_id'].astype(str).str.strip()
            data_df['_id'] = data_df['_id'].astype(str).str.strip()
            college_df = pd.concat([intake_df, data_df], ignore_index=True).drop_duplicates(subset="_id", keep="first")
        selected_fos = received_dictionary["Title"]
        selected_level = received_dictionary["Level"].lower()

        if selected_fos == "":
            response = {"Error": "Looks like you missed selecting a field of study. Please make a selection to move forward."}
            raise HTTPException(status_code=500, detail=response)
        
        openai.api_key = os.getenv("OPEN_AI_KEY")

        core_selected_fos=selected_fos
        if not request.toggle:
            messages = "Give me relevant keywords around " + selected_fos +", also try to give me words which are spelled differently in the world related to "+ selected_fos +" (this is just one of the example --> Jewellery, Jewelery) also try to break and concate word and make a cluster of relevent keywords in canada"
            selected_fos = selected_fos + " " + GPTfunction(messages, text=True, max_tokens_count=3000, usedmodel="gpt-3.5-turbo-instruct")

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

        dictionary = received_dictionary
        dictionary = rearrange_dictionary(dictionary)

        selected_fos = " ".join(selected_fos)

        core_fill = core_model(core_selected_fos, selected_level, college_df) #Warning in the console comes from this function
        gpt_fill = core_model(selected_fos, selected_level, college_df, gptifcondition=True) #Warning in the console comes from this function

        # Concatenate dataframes one below the other (axis=0 by default)
        df_concatenated = pd.concat([core_fill, gpt_fill]).drop_duplicates(subset="_id", keep="first")
        # # Reset the index for the concatenated dataframe
        fill = df_concatenated.reset_index(drop=True)


        fill["_id"] = fill["_id"].astype(str)
        for i in range(len(fill["Intake"])):
            for j in range(len(fill["Intake"][i])):
                if fill["Intake"][i][j] is not None and "_id" in fill["Intake"][i][j]:
                    del fill["Intake"][i][j]["_id"]

        intakes = []

        seasons = []
        statuses = []
        deadlines = []
        # print("Processing started above here")
        for d_idx, d in enumerate(fill["Intake"]):
            if d is None:
                print(f"Skipping None intake at index {d_idx}")
                continue
            
            d_seasons = []
            d_statuses = []
            d_deadlines = []

            for i in d:
                if not isinstance(i, dict) or 'season' not in i or 'status' not in i or 'deadline' not in i:
                    print(f"Malformed intake entry detected: {i}")
                    continue
                
                try:
                    d_seasons.append(i["season"])
                    d_statuses.append(i["status"])
                    d_deadlines.append(i["deadline"])
                except Exception as e:
                    print(f"Error processing intake entry {i}: {e}")
                    continue  # Skip this entry and proceed to the next
                
            seasons.append(", ".join(d_seasons))
            statuses.append(", ".join(d_statuses))
            deadlines.append(", ".join(d_deadlines))

        intake_df = pd.DataFrame(
            {"Seasons": seasons, "Status": statuses, "Deadline": deadlines}
        )
        fill = pd.concat([fill.drop("Intake", axis=1), intake_df], axis=1)
        if len(dictionary) == 2 or fill.empty:
            recommended_course_names = fill
        else:
            recommended_course_names = priority(fill, dictionary, core_selected_fos)
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


        if (request.LanguageProficiency!=""):
            eligible1, noteligible1 = calibre_checker(recommended_course_names, request.LanguageProficiency,request.Score)


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
            # eligible1 = eligible1.head(31)

            noteligible1.fillna("N/A", inplace=True)
            # noteligible1 = noteligible1.head(31)

            recommendData = eligible1.to_dict("records")
            end = time.time()
            response_time = end - start

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
            # eligible1 = eligible1.head(31)

            recommendData = eligible1.to_dict("records")
            end = time.time()
            response_time = end - start

            return CourseResponse(
                data={
                    "message": "Course recommendations generated successfully",
                    "eligible": recommendData,
                    "noteligible": [],
                    "response_time": response_time,
                }
            )
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Error processing request in CRS: {e}\nTraceback: {traceback_str}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

def is_email_present(email: str) -> bool:
    """Check if the email is already in the database."""
    database_name = "test"
    collection_name = "users"
    query = {"email": email}
    results = perform_database_operation(database_name, collection_name, "read", query)
    return len(results) > 0

@app.post("/api/send-otp")
def send_otp(request: EmailRequest, http_request: Request):
    try:
        referer = http_request.headers.get("referer")
        # Normalizing the referer by stripping the trailing slash if it exists
        normalized_referer = referer.rstrip('/')
        # Check if the referer is from the allowed origins
        if referer is None or normalized_referer not in allowed_origins:
            # Handle the case where referer is missing
            # You can raise an HTTPException or handle it in another appropriate way
            raise HTTPException(status_code=403, detail="Sorry, you don't have permission to access this. Please check your access rights or contact support for help")
        else:
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
                        email_verification(request.email, auth_id, normalized_referer)
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
                email_verification(request.email, auth_id, normalized_referer)
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
                    },signup=True
                )
                return {
                    "Status": "Success",
                    "Message": "Email Verification Mail Sent Into Your Mailbox",
                }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

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
                    },signup=True
                )
            return {"Status": "Success", "Message": "Verification Successful"}
        else:
            return {
                "Status": "Failure",
                "Message": "Verification failed: User account not found or already verified. Please check your details.",
            }
    except Exception as e:
        print(f"Verfication Failed: {e}")
        raise HTTPException(status_code=500, detail="Verification didn't go through. Please double-check your information and try again.")

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
                    raise HTTPException(status_code=401, detail="Your password is incorrect. Please try again.")
            else:
                raise HTTPException(status_code=403, detail="Your account hasn't been verified yet. Please check your email for the verification link.")
        else:
            raise HTTPException(status_code=404, detail="We couldn't find your account. Please double-check your information and try again.")

    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        print(f"Login Failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed. Please check your credentials and try again")

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
            raise HTTPException(status_code=404, detail="We couldn't find your account. Please double-check your information and try again.")
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        print(f"Profile Info Failed: {e}")
        raise HTTPException(status_code=500, detail="Couldn't load profile info. Please refresh and try again.")
    
@app.put("/api/update-profile-info")
def profile_info(request: ProfileInfoRequest, email: str = Depends(get_current_user)):
    try:
        users = perform_database_operation(
            "test", "userdetails", "update", {"email": email}, request.profileInfo
        )
        if users==1:
            return{"Status": "Success", "Message":"Profile Info Updated" }
        else:
            usersread = perform_database_operation("test", "userdetails", "read", {"email": email})
            if usersread and len(usersread) > 0:
                return{"Status": "Success", "Message":"Profile Info Updated" }
            else:
                raise HTTPException(status_code=404, detail="We couldn't find your account. Please double-check your information and try again.")
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        # For any other kind of exception, it's an internal server error
        print(f"Profile Info Failed: {e}")
        raise HTTPException(status_code=500, detail="Couldn't load profile info. Please refresh and try again.")

@app.post("/api/fogot-password")
def forgot_password(request: ForgetRequest, http_request: Request):
    try:
        referer = http_request.headers.get("referer")
        # Normalizing the referer by stripping the trailing slash if it exists
        normalized_referer = referer.rstrip('/')
        # Check if the referer is from the allowed origins
        if referer is None or normalized_referer not in allowed_origins:
            # Handle the case where referer is missing
            # You can raise an HTTPException or handle it in another appropriate way
            raise HTTPException(status_code=403, detail="Sorry, you don't have permission to access this. Please check your access rights or contact support for help")
        else:
            if is_email_present(request.email):
                auth_id = secrets.token_hex(40)
                resetpasswordmail(request.email, auth_id, normalized_referer)
                perform_database_operation("test","users","update",{"email": request.email},{"authId": auth_id})
                return {
                    "Status": "Success",
                    "Message": "Reset Password Mail Sent Into Your Mailbox",
                }
            else:
                return {
                    "Status": "Error", 
                    "Message": "Account not found. Please verify the information provided and try again, or create a new account if you don't have one."
                }   
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

@app.post("/api/reset-password")
def reset_password(request: ResetPasswordRequest):
    try:
        result = perform_database_operation(
            "test", "users", "read", {"authId": request.authId}
        )
        if result:
            hashed_password = bcrypt.hashpw(
                request.newpassword.encode("utf-8"), bcrypt.gensalt(10)
            )
            perform_database_operation(
                "test",
                "users",
                "update",
                {"authId": request.authId},
                {"unset": {"authId": ""}, "Password": hashed_password.decode("utf-8")},
            )
            return {"Status": "Success", "Message": "Password Updated Successfully"}
        else:
            return {
                "Status": "Failure",
                "Message": "Password Updation Failed. Invalid Request",
            }
    except Exception as e:
        print(f"Password Updation Failed: {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

def update_course_details(course_str):
    # Initialize keywords
    keywords = ['High', 'Medium', 'Low']
    
    # Initialize the default values for chance and description
    chance = "Keyword not found"
    description = ''
    
    # Iterate over the string to find the first occurrence of any keyword
    for keyword in keywords:
        keyword_pos = course_str.find(keyword)
        if keyword_pos != -1:
            # Extract the keyword as chance
            chance = keyword
            
            # Cut the string from right after the keyword
            start_pos = keyword_pos + len(keyword)
            
            # Check for specific symbols immediately after the keyword and skip them
            if course_str[start_pos:start_pos+2] in {'. ', ', '}:
                start_pos += 2
            elif course_str[start_pos] in {'\n', '.', ','}:
                start_pos += 1
            
            # The rest of the string becomes the description
            description = course_str[start_pos:].strip()
            break
    
    return chance, description


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
            stringCourse = str(request.course)
            if user_data:
                if user_data.get("profileFilled"):
                    if request.ask == "Visa":
                        messages = [{
                        "role": "system",
                        "content": "Based on the provided applicant profile and course details, ACT & Reply as a Visa Officer & assess the chances of obtaining a Canadian Visa for the applicant. Consider factors such as the applicant's academic background, test scores, work experience, and chosen program of study. The course and user profile data are dynamic and should be evaluated in the context of current immigration policies and program requirements. ALWAYS REMEMBER you have to answer only in single word Low, Medium, or High, Also provide reason in two lines only"
                        },
                        {
                        "role": "user",
                        "content": "Given the course details: "+stringCourse+" and my profile: "+stringUser+" , what are my chances of getting a Canadian Visa?"
                                }]
                        result = GPTfunction(messages, text=False)
                        request.course["Visa Chances"], request.course["Description"] = update_course_details(result)
                        # request.course["Visa Chances"] = result

                        return {"Status": "Success", "Message": request.course}

                    elif request.ask == "PR":
                        messages = [{
                        "role": "system",
                        "content": "Based on the provided user profile and course details, assess the chances of obtaining a Canadian PR for the applicant. Consider factors such as the user's academic background, test scores, work experience, and chosen program of study. The course and user profile data are dynamic and should be evaluated in the context of current immigration policies and program requirements. ALWAYS REMEMBER you have to answer only in single word High, Medium, or Low, Also provide reason in two lines only"
                        },
                        {
                        "role": "user",
                        "content": "Given the course details: "+stringCourse+" and my profile: "+stringUser+" , what are my chances of getting a Canadian PR?"
                                }]
                        result = GPTfunction(messages, text=False)
                        request.course["PR Chances"], request.course["Description"] = update_course_details(result)
                        # request.course["PR Chances"] = result
                        return {"Status": "Success", "Message": request.course}
                    else:
                        raise HTTPException(status_code=400, detail="Invalid request")
                else:
                    raise HTTPException(status_code=422, detail="To assess your "+request.ask+" Chances, please provide the details of Work Experience, Educational Experience, and Proficiency Tests by ")
            else:
                raise HTTPException(status_code=403, detail="Complete your user profile first")
            
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        print(f"Error processing request : {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

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
        print(f"Error processing request : {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

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
        print(f"Error processing request : {e}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")


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

def intake_preciser(IntakeRequest, dict_for_precise,df):
    # Assuming 'documents' is a list of dictionaries directly fetched from the database
    # Fetch the entire collection data once, leveraging caching
    # documents = fetch_all_data("test", "courses")
    
    # # Convert the entire dataset to a DataFrame in one go
    # df = pd.DataFrame(documents)
    
    # Ensure the '_id' field is converted to string format, this will apply the operation across the entire DataFrame
    df['_id'] = df['_id'].astype(str)

    # Create a copy of the original 'Intake' column before exploding
    df['OriginalIntake'] = df['Intake']
    
    # Expand the 'Intake' dictionaries into separate DataFrame rows, preserving the association with the parent document
    # This creates a new row for each item in each document's 'Intake' list, effectively normalizing the nested structure
    intake_df = df.explode('Intake').reset_index(drop=True)

    # Ensure each 'Intake' entry is a dictionary before further processing
    intake_df = intake_df[intake_df['Intake'].apply(lambda x: isinstance(x, dict) if x else False)]

    # Filter out rows where 'Intake' is None or does not contain the expected keys ('season', 'status', 'deadline')
    # This step also filters based on the 'season' matching in the 'IntakeRequest'
    filtered_df = intake_df[intake_df['Intake'].apply(lambda x: x.get('season') in IntakeRequest if x else False)]
    # If necessary, further processing can be done here to structure the data as needed
    filtered_df['Seasons1'] = ', '.join(IntakeRequest)


    # Restore the original 'Intake' data
    filtered_df['Intake'] = filtered_df['_id'].map(df.set_index('_id')['OriginalIntake'])

    preciser_final_df = filtered_df.drop_duplicates('_id')

    final_df_intake = data_preciser(dict_for_precise, preciser_final_df)

    final_df = pd.concat([final_df_intake, preciser_final_df], ignore_index=True).drop_duplicates(subset="_id", keep="first")
    # filtered_df['Intake'] = filtered_df['Intake'].apply(lambda x: x if isinstance(x, list) else [x]) 

    return final_df

def rearrange_dictionary(received_dictionary):
    # Convert relevant fields to lowercase
    received_dictionary["Level"] = received_dictionary["Level"].lower()
    received_dictionary["Province"] = received_dictionary["Province"].lower()
    received_dictionary["Seasons1"] = ', '.join(received_dictionary["Seasons1"]).lower()

    # Initialize a new dictionary that will store the ordered entries
    ordered_dictionary = {}

    # Iterate through the original dictionary items
    for key, value in received_dictionary.items():
        # Add the current item to the ordered dictionary
        ordered_dictionary[key] = value

        # If the current key is 'Level', add 'Level1' right after
        if key == 'Level':
            ordered_dictionary['Level1'] = value  # Level1 has the same content as Level

        # Similarly, add 'Province1' right after 'Province'
        if key == 'Province':
            ordered_dictionary['Province1'] = value  # Province1 has the same content as Province

    # Now, ensure that any cleaning function is applied after setting the order
    # Assuming cleandict is a function you have defined to clean/modify the dictionary further
    ordered_dictionary = cleandict(ordered_dictionary)

    return ordered_dictionary

@app.post('/api/automail')
def automail(request: AutoMailRequest):
    try:
        attachments = [{"filename": item.filename, "content": base64.b64decode(item.content)} for item in request.attachments]
        if attachments:
            attachment = attachments[0]
            satbulkmail(request.sender, request.to, request.subject, attachment["content"], attachment["filename"], request.cc, request.name)
            return {"status": "success with attachment"}
        else:
            satbulkmail(request.sender, request.to, request.subject, cc_addresses=request.cc, client_name = request.name)
            return {"status": "success without attachment"}
    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Error processing request in Automail: {e}\nTraceback: {traceback_str}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")

@app.post('/api/userDoc')
def retainerfunction(request: DocuSealRequest):
    try:
        finder_string = request.email +'-'+ request.serveDoc
        print(finder_string)
        doc = get_docuseal_templates_fn(finder_string)

        count = doc['pagination']['count']
        if count == 1:
            slug = doc['data'][0]['slug']
        elif count > 1:
            slug = None
            slug = doc['data'][0]['slug']
        else:
            default_slug = get_slug_value(request.serveDoc)
            if default_slug == None:
                response = {"Error": "The default signing document is missing for your specific service. Please inform the developer at aryaman.singh@rooton.ca."}
                raise HTTPException(status_code=404, detail=response)
            else:
                slug = default_slug
        
        return {"Status":"Found", "Slug": slug}
    except HTTPException as http_exc:
        # If it's an HTTPException, we just re-raise it
        # This is assuming HTTPException is meant to be used for HTTP status-related errors
        raise http_exc
    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Error processing request in Docuseal: {e}\nTraceback: {traceback_str}")
        raise HTTPException(status_code=500, detail="We're having trouble processing your request right now. Please try again later.")
    
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
