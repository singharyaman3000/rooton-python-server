from imports import *
from rich import print
from .load_document import load_documents, cache_vectorstore_and_embeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Returns the chat message history for the given session ID.

    Args:
        session_id (str): The ID of the session.

    Returns:
        BaseChatMessageHistory: The chat message history for the session.

    Raises:
        None

    Note:
        This function uses the `store` dictionary to store and retrieve chat message histories.
        If the session ID is not found in the `store`, a new `UpstashRedisChatMessageHistory` object is created with the provided session ID and stored in the `store`.
        The `url` and `token` for the `UpstashRedisChatMessageHistory` object are obtained from the environment variables `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`, respectively.
        The `ttl` for the `UpstashRedisChatMessageHistory` object is set to 36000.
    """
    if session_id not in store:
        store[session_id] = UpstashRedisChatMessageHistory(
            url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"), ttl=36000, session_id=session_id
        )
    return store[session_id]

def RAG_Loader():
    """
    Initializes and returns a runnable RAG chain for answering questions about Permanent Residency (PR) in Canada through the Ontario immigration nominee program (OINP).

    The chain is composed of several components:
    - A history-aware retriever that uses a contextualized question prompt to formulate standalone questions from chat history and user input.
    - A question-answer chain that uses a specialized prompt to answer user queries accurately, providing precise CRS scores and evaluating eligibility for PR.

    The function takes no parameters and returns a RunnableWithMessageHistory instance, which can be used to execute the RAG chain with a given input message and chat history.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    vectorstore = cache_vectorstore_and_embeddings(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant specialized in answering questions about Permanent Residency (PR) in Canada through the Ontario immigration nominee program (OINP).
    Use the following retrieved context to answer user queries accurately.

    Always respond in the same language in which the user asks the query. if the answer is not available in provided context, state I don't know.

    if user asks to provide CRS scores provide a precise number by calculating every aspect which is required to provide a precise score. if required information is missing from the user's side, ask them to provide information. So that precise score can be given.

    CONTEXT: {context}

    When a user inquires about their chances or eligibility for Permanent Residency (PR) in Canada through the Ontario immigration nominee program (OINP), engage in a question-and-answer session to gather necessary information.
    Evaluate previous interactions to ensure you have all relevant details.
    Once you are confident in your understanding, provide a precise & concise answer with supporting insights.
    Keep the conversation clear and focused, summarizing key points in a concise manner and offering actionable advice when appropriate."""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    runnable_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return runnable_rag_chain
