from imports import *
from . import load_documents
from . import cache_vectorstore_and_embeddings_from_docs
from langchain.chains import RefineDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

def RAG_Loader_with_refined_chaining():
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print(f"Loaded and chunked {len(splits)} chunks from {len(docs)} documents")
    vectorstore = cache_vectorstore_and_embeddings_from_docs(splits)

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
    )
    document_variable_name = "context"

    qa_system_prompt = """You are an assistant specialized in answering questions about Permanent Residency (PR) in Canada through the Ontario immigration nominee program (OINP).
    Use the following retrieved context to answer user queries accurately.

    Always respond in the same language in which the user asks the query. if the answer is not available in provided context, state I don't know.

    if user asks to provide CRS scores provide a precise number by calculating every aspect which is required to provide a precise score. if required information is missing from the user's side, ask them to provide information. So that precise score can be given.

    CONTEXT: {context}

    When a user inquires about their chances or eligibility for Permanent Residency (PR) in Canada through the Ontario immigration nominee program (OINP), engage in a question-and-answer session to gather necessary information.
    Evaluate previous interactions to ensure you have all relevant details.
    Once you are confident in your understanding, provide a precise & concise answer with supporting insights.
    Keep the conversation clear and focused, summarizing key points in a concise manner and offering actionable advice when appropriate."""

    prompt = PromptTemplate.from_template(qa_system_prompt)

    initial_llm_chain = LLMChain(llm=llm, prompt=prompt)
    initial_response_name = "prev_response"
    prompt_refine = PromptTemplate.from_template(
        "Here's your first summary: {prev_response}. "
        "Now add to it based on the following context: {context}"
    )
    refine_llm_chain = LLMChain(llm=llm, prompt=prompt_refine)

    chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        return_intermediate_steps=True
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chain
        | StrOutputParser()
    )
    return rag_chain


