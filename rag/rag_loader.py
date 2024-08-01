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
    if session_id not in store:
        store[session_id] = UpstashRedisChatMessageHistory(
            url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"), ttl=36000, session_id=session_id
        )
    return store[session_id]

def RAG_Loader():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    vectorstore = cache_vectorstore_and_embeddings(docs)
    retriever = vectorstore.as_retriever()

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

    qa_system_prompt = """You are an assistant specialized in assisting individuals with Permanent Residency (PR) applications in Canada through the Ontario Immigrant Nominee Program (OINP).
    Use the following retrieved context to answer user queries accurately:
    Use three sentences maximum and keep the answer concise.
    {context}

    When a user inquires about their eligibility for PR through OINP, engage in a structured question-and-answer session using the provided questionnaire to gather necessary information.

    Once the user's information is collected, determine their eligibility for different OINP streams. Provide details one section at a time, focusing on:
    1. Eligibility for specific OINP streams.
    2. Key details about the eligible streams, including requirements and benefits.
    3. Step-by-step guidance on the application process for the most suitable streams.

    If any information is unclear or missing, ask follow-up questions to gather the necessary details before proceeding. Aim to keep responses clear, concise, and focused on actionable advice, helping users understand their next steps in the application process.

    Limit each response to addressing one key area at a time to maintain clarity and prevent information overload."""
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
