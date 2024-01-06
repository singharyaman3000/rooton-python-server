from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pymongo
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise as sk
import json
from dotenv import load_dotenv
import re
import openai
import time
from cachetools import TTLCache, cached

load_dotenv()
cache = TTLCache(maxsize=1000000, ttl=86400)

app = FastAPI()

# Pydantic models for request and response
class CourseRequest(BaseModel):
    fos: str
    level: str
    dictionary: dict
    email: str

class CourseResponse(BaseModel):
    data: dict

# Function to fetch all data (similar to your script)
@cached(cache)
def fetch_all_data(database, collection):
    # Connect to MongoDB (Specify your MongoDB URI)
    MONGODB_URI = os.getenv('MONGODB_URI')
    # print("MONGODB_URI",os.getenv('MONGODB_URI'))
    client = pymongo.MongoClient(MONGODB_URI)

    db = client[database]
    collection = db[collection]

    cursor = collection.find({})
    documents = list(cursor)
    results = [doc for doc in documents]

    client.close()
    return results

async def preload_cache():
    try:
        # Preload data for each collection
        fetch_all_data("test", "courses")
        fetch_all_data("test", "userdetails")
        print("caching done")
        # Add more collections as needed
    except Exception as e:
        print(f"Error during cache preloading: {e}")

# Register the preload_cache function to run at startup
app.add_event_handler("startup", preload_cache)


def process_budget(budget_str):
    """Extract the lower and upper fee limits from the budget string."""
    if "to above" in budget_str:
        lower_limit = int(budget_str.split(" ")[0])
        upper_limit = int('999999')
    else:
        lower_limit, upper_limit = map(int, budget_str.split("-"))
    return lower_limit, upper_limit

def priorty(dataframe, dictionary, selected_fos):
    results=[]
    df_copy = dataframe.copy()
    for i in range(len(dictionary), 0, -1):
        df_copy = dataframe.copy()

        sub_dict = dict(list(dictionary.items())[0:i])
        if len(sub_dict)>0:
            comp_str=""
            for i,key in enumerate(sub_dict):

                if i==len(sub_dict)-1:
                    if i==0:
                        comp_str+="(df_copy['{0}'].str.contains('|'.join({1}), case=False))".format(key,sub_dict[key])
                    else:
                        if isinstance(sub_dict[key], (int, float)):
                            comp_str+= "(df_copy['{0}']<={1})".format(key,sub_dict[key])
                        else:
                            comp_str+= "(df_copy['{0}'].str.lower()=='{1}')".format(key,sub_dict[key])
                elif i<=len(sub_dict)-2:
                    if i==0:
                        comp_str+="(df_copy['{0}'].str.contains('|'.join({1}), case=False)) & ".format(key,sub_dict[key])
                    else:
                        if isinstance(sub_dict[key], (int, float)):
                            comp_str+= "(df_copy['{0}']<={1}) & ".format(key,sub_dict[key])
                        else:
                            comp_str+= "(df_copy['{0}'].str.lower()=='{1}') & ".format(key,sub_dict[key])
                else:
                    if isinstance(sub_dict[key], (int, float)):
                        comp_str+= "(df_copy['{0}']<={1})".format(key,sub_dict[key])
                    else:
                        comp_str+= "(df_copy['{0}'].str.lower()=='{1}')".format(key,sub_dict[key])
            # print(comp_str)
            df = df_copy[eval(comp_str)]

            counts = df['Title'].str.count('|'.join(selected_fos), flags=re.IGNORECASE)

            df1 = df.assign(score=counts).sort_values('score', ascending=False).drop('score', axis=1)
            results.append(df1)

    final_result = pd.concat(results, ignore_index=True)
    final_finalResult=pd.concat([final_result,dataframe], ignore_index=True)

    final_finalResult.drop_duplicates(subset='_id',keep='first', inplace=True)
    lo_bhai = final_finalResult.filter(['FieldOfStudy','Province','Institute','Length','IeltsOverall',"DuolingoOverall","PteOverall","Intake",'City','Campus','Title','Level','Fee'], axis=1)

    return final_finalResult

def calibre_checker(df, Imarks):
    noteligible = pd.DataFrame(columns=df.columns)

# Iterate through each row of the original DataFrame
    for index, row in df.iterrows():
        # Check if the condition is satisfied
        if row['IeltsOverall'] > Imarks:
            # If not satisfied, drop the row and append it to the dropped DataFrame
            noteligible = pd.concat(
                [noteligible, row.to_frame().T], ignore_index=True)
            df.drop(index, inplace=True)
    return df, noteligible

def cleandict(my_dict):
    for key in list(my_dict.keys()):
        # Check if the value of the key is an empty string, zero or None
        if my_dict[key] in ["", 0, None]:
            # If the value meets the specified criteria, remove the key
            del my_dict[key]
    return my_dict


def core_model(fos, level,college_df):
    if (level != "false" or fos != "false"):

        college_df["IeltsOverall"] = college_df["IeltsOverall"].astype(
            float).round(1)

        college_df['Length'] = college_df['Length'].astype('str').str.extractall(
            '(\d+)').unstack().fillna('').sum(axis=1).astype(int)
        college_df['FeeText'] = college_df['Fee']
        college_df["Fee"] = college_df.Fee.str.extract('(\d+)')

        college_df['Fee'] = college_df['Fee'].astype(int)
        college_df[['FieldOfStudy', 'Province', 'InstituteName', 'Title', 'Level']] = college_df[[
            'FieldOfStudy', 'Province', 'InstituteName', 'Title', 'Level']].astype(str)

        college_df[['Length', 'Fee']] = college_df[[
            'Length', 'Fee']].astype(int)

        comb_frame = college_df["FieldOfStudy"].astype(
            str) + " " + college_df["Title"].astype(str) + " " + college_df["Level"].astype(str)

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(comb_frame)
        if level == None:
            Query_Input = fos
        else:
            Query_Input = fos+" "+level
        Input_transform = vectorizer.transform([Query_Input])

        cosine_similarities = sk.pairwise_kernels(Input_transform, X).flatten()

        related_docs_indices = cosine_similarities.argsort()[:-300:-1]

        df = pd.DataFrame(columns=college_df.columns)
        point = 0

        for index in related_docs_indices:
            program = (college_df.iloc[index])
            temp = []
            for v in program.values:
                temp.append(v)
            df.loc[point] = temp
            point = point+1
        return df
    else:
        return {"Error": "Please Select Atleast Single Option"}



@app.post("/recommend_courses", response_model=CourseResponse)
async def recommend_courses(request: CourseRequest):
    print("recieved Request", request)
    start = time.time()
    my_dict = request.dictionary
    received_dictionary = {}
    for key in my_dict:
        new_key = key
        if key == 'FieldOfStudy':
            new_key = 'Title'
        elif key == 'Budget':
            new_key = 'Fee'
        elif key == 'Duration':
            new_key = 'Length'
        elif key == 'Intake':
            new_key = 'Seasons'
        received_dictionary[new_key] = my_dict[key]
    received_dictionary['Length'] = int(received_dictionary.get('Length', 0) or 0)
    received_dictionary['Fee'] = int(received_dictionary.get('Fee', 0) or 0)
    if received_dictionary['Fee']!=0:
        fee_lower, fee_max = process_budget(received_dictionary['Fee'])
        received_dictionary['FeeLower'] = fee_lower
        received_dictionary['FeeMax'] = fee_max
        received_dictionary['Fee'] = int(received_dictionary['FeeMax'])
        received_dictionary.pop('FeeMax')
        received_dictionary.pop('FeeLower')

    received_dictionary.update(my_dict)
    received_dictionary['Length'] = int(received_dictionary['Length'])

    received_dictionary.pop('FieldOfStudy')
    received_dictionary.pop('Budget')
    received_dictionary.pop('Duration')
    received_dictionary.pop('Intake')
    value = fetch_all_data("test", "courses")
    college_df = pd.DataFrame(value)
    userdetails = fetch_all_data("test", "userdetails")
    selected_useremail = request.email
    selected_fos = received_dictionary['Title']
    selected_level = received_dictionary['Level'].lower()

    if selected_fos == "":
        response = {"Error": "Please Select A Field Of Study"}
        print(response)
        exit()
    openai.api_key = os.getenv("OPEN_AI_KEY")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="List 18 courses related to " +
        selected_fos +
        " that can be studied in canada",
        temperature=0.7,
        max_tokens=200,
    )
    selected_fos = response.choices[0].text + " " + selected_fos

    title = selected_fos
    x = title.replace(",", " ")
    input_words = x.split()
    input_words = set(input_words)

    input_words = set(re.findall(r'[a-zA-Z]+', ' '.join(input_words)))

    input_words = set([w.lower() for w in input_words])

    joining_words = {'and', 'or', 'for', 'in', 'the', 'of', 'on', 'to', 'a', 'an'}
    input_words = input_words - joining_words

    selected_fos = input_words
    received_dictionary['Title'] = selected_fos

    received_dictionary['Level'] = received_dictionary['Level'].lower()
    received_dictionary['Province'] = received_dictionary['Province'].lower()
    received_dictionary['Seasons'] = received_dictionary['Seasons'].lower()
    dictionary = received_dictionary
    dictionary = cleandict(dictionary)

    selected_fos = ' '.join(selected_fos)

    fill = core_model(selected_fos, selected_level, college_df)

    fill['_id'] = fill['_id'].astype(str)
    for i in range(len(fill['Intake'])):
        for j in range(len(fill['Intake'][i])):
            if fill['Intake'][i][j] is not None and '_id' in fill['Intake'][i][j]:
                del fill['Intake'][i][j]['_id']

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
        {"Seasons": seasons, "Status": statuses, "Deadline": deadlines})
    fill = pd.concat([fill.drop('Intake', axis=1), intake_df], axis=1)
    if(len(dictionary)==1):
        recommended_course_names = fill
    else:
        recommended_course_names = priorty(fill, dictionary, selected_fos)

    recommended_course_names.drop(["__v", 'Language', 'IeltsReading', 'IeltsWriting', 'IeltsSpeaking', 'IeltsListening', 'PteReading', 'PteWriting', 'PteSpeaking', 'PteListening', "Country"], axis=1, inplace=True)

    for u in userdetails:
        if u['email'] == selected_useremail:
            IELTS_O = u['Scores']['IELTS']['Overall']
            break
    eligible, noteligible = calibre_checker(recommended_course_names, IELTS_O)

    eligible['Length'] = eligible['Length'].apply(
        lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months" if x >= 12 else f"{x} Months")
    noteligible['Length'] = noteligible['Length'].apply(
        lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months" if x >= 12 else f"{x} Months")
    eligible = eligible.reindex(columns=['CreatedOn', 'FieldOfStudy', 'InstituteName', 'Title', 'Level', 'Length', 'ApplicationFee', 'FeeText', 'Seasons', 'Status', 'Deadline',
                                'Percentage', 'Backlog', 'Gap', 'Campus', 'IeltsOverall', "PteOverall", 'TOEFLOverall', "DuolingoOverall", 'GRE', 'GMAT', 'City', 'Province', 'Visa Chances', 'PR'])
    noteligible = noteligible.reindex(columns=['CreatedOn', 'FieldOfStudy', 'InstituteName', 'Title', 'Level', 'Length', 'ApplicationFee', 'FeeText', 'Seasons', 'Status',
                                      'Deadline', 'Percentage', 'Backlog', 'Gap', 'Campus', 'IeltsOverall', "PteOverall", 'TOEFLOverall', "DuolingoOverall", 'GRE', 'GMAT', 'City', 'Province', 'Notes'])

    data = []

    eligible.fillna("N/A", inplace=True)
    eligible =  eligible.head(31)

    recommendData = eligible.to_dict("records")
    end = time.time()
    response_time = (end - start)
    print(response_time)
    # print(json_data)
    
    return CourseResponse(data={"message": "Course recommendations generated successfully","recommended_courses": recommendData,
                                "response_time": response_time})

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
