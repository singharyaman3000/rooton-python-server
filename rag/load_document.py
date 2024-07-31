from imports import *
import shutil

docs_cache = TTLCache(maxsize=1000000, ttl=86400)

@cached(cache=docs_cache)
def load_documents():
    print("Loading documents from source")
    URLS = os.getenv("FEEDING_URL1").split(",")

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(URLS,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("body-field", "row intro")
            )
        ),
    )
    # Flatten the list of lists into a single list of URLs
    flattened_web_paths = [url for sublist in loader.web_paths for url in sublist]
    loader.web_paths = flattened_web_paths  # Update the loader's web_paths
    return loader.load()


def cache_vectorstore_and_embeddings(docs):
    DB_PATH = "vectorstores/db/"
    if os.path.exists(DB_PATH):
        shutil.rmtree("vectorstores")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, persist_directory=DB_PATH, embedding=embeddings)
    return vectorstore
