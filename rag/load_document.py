from imports import *
import shutil

docs_cache = TTLCache(maxsize=1000000, ttl=86400)

@cached(cache=docs_cache)
def load_documents():
    print("Loading documents from source")
    URLS = os.getenv("FEEDING_URL").split(",")

    # Separate URLs by parsing rules
    default_urls = [url for url in URLS if "ontario.ca" in url]
    special_urls = [url for url in URLS if "ontario.ca" not in url]

    # Load documents using WebBaseLoader for default URLs
    default_loader = WebBaseLoader(
        web_paths=(default_urls,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("body-field", "row intro")
            )
        ),
    )
    # If necessary, flatten the list of web paths
    if isinstance(default_loader.web_paths[0], list):
        default_loader.web_paths = [url for sublist in default_loader.web_paths for url in sublist]

    default_documents = default_loader.load()

    # Load documents using WebBaseLoader for special URLs
    special_loader = WebBaseLoader(
        web_paths=(special_urls,),
        bs_kwargs=dict()  # No specific parsing rules
    )
    # If necessary, flatten the list of web paths
    if isinstance(special_loader.web_paths[0], list):
        special_loader.web_paths = [url for sublist in special_loader.web_paths for url in sublist]

    special_documents = special_loader.load()

    # Combine documents from both loaders
    documents = default_documents + special_documents

    return documents


def cache_vectorstore_and_embeddings(docs):
    DB_PATH = "vectorstores/db/"
    if os.path.exists(DB_PATH):
        shutil.rmtree("vectorstores")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, persist_directory=DB_PATH, embedding=embeddings)
    return vectorstore
