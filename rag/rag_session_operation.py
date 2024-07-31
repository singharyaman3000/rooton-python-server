import json
import uuid
from rag.rag_loader import get_session_history
from utilities.helpers.db_helper import perform_database_operation
from langchain.schema import HumanMessage, AIMessage


def generate_session_id(email):
    database = "test"
    collection_name = "rag-session-collection"

    # Check if email exists
    query = {"email": email}
    existing_user = perform_database_operation(database, collection_name, "read", query)

    if existing_user:
        return existing_user[0].get("session_id")

    # Generate new session ID
    session_id = str(uuid.uuid4())

    # Create a new document with email and session ID
    document = {"email": email, "session_id": session_id}
    perform_database_operation(database, collection_name, "create", document)

    return session_id


def update_user_session_id(email):
    database = "test"
    collection_name = "rag-session-collection"
    session_id = str(uuid.uuid4())
    query = {"email": email}
    perform_database_operation(
        database, collection_name, "update", query, {"session_id": session_id}
    )

    return session_id


def get_conversation_by_session_id(session_id: str) -> list:
    message_history = get_session_history(session_id)
    convo_messages = message_history.messages
    json_messages = []
    for message in convo_messages:
        if isinstance(message, HumanMessage):
            json_messages.append({"type": "human", "data": {"content": message.content}})
        elif isinstance(message, AIMessage):
            json_messages.append({"type": "ai", "data": {"content": message.content}})

    return json_messages
    
