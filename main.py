from imports import *

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cache = TTLCache(maxsize=1000000, ttl=86400)

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
FRONTEND_URL = os.getenv("FRONTEND_URL")
additional_origin = os.getenv("ADDITIONAL_ORIGIN", "")
# Stripe and Razorpay setup
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


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

docs_cache = TTLCache(maxsize=1000000, ttl=86400)


async def preload_cache():
    try:
        # Preload data for each collection
        logger.info("Caching started")
        fetch_all_data("test", "courses")
        logger.info("Caching done for courses")
        logger.info("Cache preloading done")
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
        if 'error' in request.query_params:
            error = request.query_params['error']
            if error == 'access_denied':
                logging.warning("User denied access during OAuth process.")
                return RedirectResponse(url=frontend_url + "/login")
            else:
                logging.error(f"OAuth error: {error}")
                return RedirectResponse(url=frontend_url + "/login")
        # Authlib checks the state parameter against the session automatically
        token = await oauth.google.authorize_access_token(request)
        # If we get here, the state check has passed
        if not token:
            # User has canceled the authorization
            return RedirectResponse(url=frontend_url + "/login")
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

@app.post("/api/recommend_courses", response_model=CourseResponse)
async def recommend_courses(request: CourseRequest, email: str = Depends(get_current_user)):
    """
    Handles the API endpoint for recommending courses based on user input.
    
    Parameters:
    request (CourseRequest): The user's course request data.
    email (str): The email of the current user.
    
    Returns:
    CourseResponse: A response containing the recommended courses and additional metadata.
    """
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
def retainer_function(request: DocuSealRequest):
    try:
        slug = None
        if len(request.email) != 0:
            finder_string = request.email +'-'+ request.serveDoc

            doc = get_docuseal_templates_fn(finder_string)

            count = doc['pagination']['count']
            if count > 0:
                slug = doc['data'][0]['slug']
        
        if slug is None:
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
    
@app.put("/api/docusealCheck")
async def docuSeal(request: CheckDocRequest):
    try:
        user = await perform_database_operation_docuseal(
            "test", "DocuSealDB", "read", {"email": request.email}
        )
        if request.op == 'create':
            if user and len(user) > 0:
                if request.serveDoc not in user[0]['Shorthand']:
                    user[0]['Shorthand'].append(request.serveDoc)
                    update_result = await perform_database_operation_docuseal(
                        "test", "DocuSealDB", "update", {"email": request.email}, {"Shorthand": user[0]['Shorthand']}
                    )
                    if update_result == 1:
                        return {"Status": "Added", "Message": "DocuSeal DB Updated"}
                    else:
                        raise HTTPException(status_code=404, detail="We couldn't find your account. Please double-check your information and try again.")
                else:
                    return {"Status": "Already Signed", "Message": "You have already signed this document."}
            else:
                create_result = await perform_database_operation_docuseal(
                    "test", "DocuSealDB", "create", {"email": request.email, "Shorthand": [request.serveDoc]}
                )
                if create_result:
                    return {"Status": "Added", "Message": "DocuSeal DB Updated"}
                else:
                    raise HTTPException(status_code=404, detail="We couldn't find your account. Please double-check your information and try again.")
        else:
            if user and len(user) > 0 and request.serveDoc in user[0]['Shorthand']:
                return {"Status": "Found", "isAlreadySigned": True}
            else:
                return {"Status": "Not Found", "isAlreadySigned": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify_payment")
async def verify_payment(payload: PaymentVerificationRequest):
    try:
        # Generate the expected signature
        expected_signature = generated_signature(payload.orderCreationId, payload.razorpayPaymentId)
        
        # Compare the signatures
        if expected_signature != payload.razorpaySignature:
            return JSONResponse(
                content={"message": "payment verification failed", "isOk": False},
                status_code=400
            )

        return JSONResponse(
            content={"message": "payment verified successfully", "isOk": True},
            status_code=200
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Verify Payment API Failed: {e}")
        raise HTTPException(status_code=500, detail="Error In Verify Payment API")

@app.post("/api/stripe")
async def handle_stripe_payment(payment: StripePayment):
    try:
        session = stripe.checkout.Session.retrieve(payment.session_id)

        if session.payment_status == 'paid':
            invoicedata = stripe.Invoice.retrieve(session.invoice or '')

            payment_data = {
                "email": session.customer_email,
                "name": session.customer_details.name,
                "payment_gateway": "stripe",
                "payment_id": session.payment_intent,
                "amount": session.amount_total,
                "currency": session.currency.upper(),
                "status": "succeeded",
                "created_at": datetime.fromtimestamp(session.created).strftime('%Y-%m-%d %H:%M:%S'),
                "details": {
                    "stripe": {
                        "checkout_session_id": session.id,
                        "customer_id": session.customer,
                        "payment_method": session.payment_method_types[0]
                    }
                },
                "invoice_id": invoicedata.id if invoicedata else None,
                "invoice_url": invoicedata.hosted_invoice_url if invoicedata else None
            }
            payment_id = create_payment_record(payment_data)

            return {"saved": True}
        else:
            raise HTTPException(status_code=400, detail="Payment not completed")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Stripe PaymentDB API Failed: {e}")
        raise HTTPException(status_code=500, detail="Error In Stripe PaymentDB API")

@app.post("/api/razorpay")
async def handle_razorpay_payment(payment: RazorpayPayment):
    try:
        razorpay_client = razorpay.Client(auth=(os.getenv("RAZORPAY_API_KEY"), os.getenv("RAZORPAY_API_SECRET")))
        payment_details = razorpay_client.payment.fetch(payment.payment_id)
        if payment_details['status'] == 'captured':
            created_at = datetime.fromtimestamp(payment_details['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            payment_data = {
                "email": payment_details['notes']['email'],
                "name": payment_details['notes']['name'],
                "payment_gateway": "razorpay",
                "payment_id": payment.payment_id,
                "amount": payment_details['amount'],
                "currency": payment_details['currency'].upper(),
                "status": "captured",
                "created_at": created_at,
                "details": {
                    "razorpay": {
                        "order_id": payment.order_id,
                        "payment_method": payment_details['method']
                    }
                },
                "invoice_id": None  # Assuming no invoice ID for Razorpay
            }
            payment_id = create_payment_record(payment_data)
            paymailtoacc(payment_id=payment.payment_id, payment_amount=payment_details['amount']/100, payment_date=created_at, client_name=payment_details['notes']['name'],client_email=payment_details['notes']['email'], client_address=payment_details['notes']['address'], service_plan=payment_details['notes']['serviceName'], client_gst=payment_details['notes'].get('gst', None))
            if payment_id:
                return {"saved": True}
            else :
                raise HTTPException(status_code=400, detail="Payment record not captured in DB")
        else:
            raise HTTPException(status_code=400, detail="Payment not captured")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Razorpay PaymentDB API Failed: {e}")
        raise HTTPException(status_code=500, detail="Error In Razorpay PaymentDB API")

@app.get("/api/user/payments", response_model=List[dict])
async def get_user_payments(email: str = Depends(get_current_user)):
    try:
        query = {"email": email}
        payments = get_payments(query)
        return serialize_payments(payments)
    except Exception as e:
        logging.error(f"Error in getting payments: {e}")
        raise HTTPException(status_code=500, detail="Error processing payments")

@app.get("/api/fetch/payId/{payment_id}")
async def get_payment_details(payment_id: str):
    try:
        query = {"_id": ObjectId(payment_id)}
        payments_collection = MongoConnectPaymentDB()
        payment = payments_collection.find_one(query)

        if payment:
            try:
                serialized_payment = serialize_payments_with_id(payment)
                logging.debug(f"Serialized payment: {serialized_payment}")
                return serialized_payment
            except Exception as e:
                logging.error(f"Error in serializing payment: {e}")
                raise HTTPException(status_code=500, detail="Error processing payment details")
        else:
            raise HTTPException(status_code=404, detail="Payment not found")
    except Exception as e:
        logging.error(f"Error in getting payment details by Id: {e}")
        raise HTTPException(status_code=500, detail="Error processing payment details by Id")


conversational_rag_chain = RAG_Loader()

@app.post("/api/chat", response_model=MessageResponse)
async def chat_endpoint(message_request: MessageRequest, email: str = Depends(get_current_user)):
    """
    Handles incoming chat requests from users.

    Args:
        message_request (MessageRequest): The incoming chat message request.
        email (str): The email of the user sending the request. Defaults to the current user.

    Returns:
        MessageResponse: A response to the user's chat message.

    Raises:
        HTTPException: If an error occurs while processing the chat request.
    """
    session_id = message_request.session_id
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        message = message_request.message
        response = await handle_message(session_id, message)

        return MessageResponse(session_id=session_id, response=response)
    except Exception as e:
        logging.error(f"Error in Chat API: {e}")
        raise HTTPException(status_code=500, detail="Error in chat_endpoint_fn")


async def handle_message(session_id: str, message: str) -> str:
    """
    Handles incoming chat messages by invoking the conversational RAG chain.

    Args:
        session_id (str): The unique identifier for the user's session.
        message (str): The incoming chat message.

    Returns:
        str: The response to the user's chat message.
    """
    response = conversational_rag_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    return response


@app.get("/api/user/session")
async def get_session_id(email: str = Depends(get_current_user)):
    """
    Handles GET requests to retrieve a user's session ID.

    Args:
        email (str): The email address of the user. Defaults to the current user.

    Returns:
        dict: A dictionary containing the user's session ID.
    """
    return {"session_id": generate_session_id(email)}


@app.put("/api/user/session/update")
async def update_session_id(email: str = Depends(get_current_user)):
    """
    Handles PUT requests to update a user's session ID.

    Args:
        email (str): The email address of the user. Defaults to the current user.

    Returns:
        dict: A dictionary containing the updated session ID.
    """
    return {"session_id": update_user_session_id(email)}

@app.get("/api/user/conversation/")
async def get_user_session_id(email: str = Depends(get_current_user)):
    """
    Handles GET requests to retrieve a user's conversation history.

    Args:
        email (str): The email address of the user. Defaults to the current user.

    Returns:
        dict: A dictionary containing the user's session ID and conversation history.
    """
    session_id = generate_session_id(email)
    messages = get_conversation_by_session_id(session_id)
    return {"session_id": session_id, "conversation":messages}

if __name__ == "__main__":
    import uvicorn

    print("Starting webserver...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("DEBUG", False),
        log_level=os.getenv('LOG_LEVEL', "info"),
        proxy_headers=True,
        ws_ping_interval=None,
        ws_ping_timeout=None
    )
