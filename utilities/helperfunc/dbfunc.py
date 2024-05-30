import os
import pymongo
from dotenv import load_dotenv

load_dotenv()


def fetch_collection(database, collection):
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(MONGODB_URI)

    db = client[database]
    collection = db[collection]

    return collection


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

        update_profile_filled = (
            len(update_data.get("educationalExperiences", [])) >= 1 and
            len(update_data.get("englishCredentialInfo", [])) >= 1 and
            len(update_data.get("workExperiences", [])) >= 1
        )

        # Update profileFilled based on conditions
        update_operations = {}
        if update_profile_filled:
            update_data["profileFilled"] = True

        # Prepare update operations
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

