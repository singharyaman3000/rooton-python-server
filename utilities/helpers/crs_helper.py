
from imports import pd, openai, os, HTTPException, re, sk, traceback, TfidfVectorizer

pd.options.mode.chained_assignment = None

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

