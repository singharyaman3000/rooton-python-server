from imports import *
from rich import print

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from rag.load_document import load_documents, load_documents_with_recursive_chunking,cache_vectorstore_and_embeddings_from_text, cache_vectorstore_and_embeddings_from_docs

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = UpstashRedisChatMessageHistory(
            url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"), ttl=36000, session_id=session_id
        )
    return store[session_id]

def RAG_Loader():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    docs = load_documents()  # Load your documents here
    print(f"Loaded {len(docs)} documents")

    vectorstore = cache_vectorstore_and_embeddings_from_docs(docs)
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
        get_session_history,  # Assume this function is defined elsewhere
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return runnable_rag_chain
