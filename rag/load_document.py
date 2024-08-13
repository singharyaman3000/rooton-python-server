from imports import *
import shutil
from langchain.text_splitter import MarkdownTextSplitter

docs_cache = TTLCache(maxsize=1000000, ttl=86400)

@cached(cache=docs_cache)
def load_documents():
    print("Loading documents from source")
    URLS = os.getenv("FEEDING_URL").split(",")

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

@cached(cache=docs_cache)
def load_documents_with_markdown():
    print("Loading documents from source")
    URLS = os.getenv("FEEDING_URL").split(",")

    # Load the contents from the web
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
    loader.web_paths = flattened_web_paths
    documents = loader.load()

    # Initialize a Markdown text splitter
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Split each document into chunks
    chunked_documents = []
    for document in documents:
        # Assuming document is an object with a 'text' attribute
        # If documents are plain strings, replace document.text with document
        text_content = document.text if hasattr(document, 'text') else document
        
        # Split text into chunks
        chunks = text_splitter.split_text(text_content)
        
        # Store the chunks
        chunked_documents.extend(chunks)

    return chunked_documents


# Function to get and chunk content with specific classes from a URL
def get_content_from_urls(urls):
  content = ''
  try:
      for url in urls:
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.content, 'html.parser')

        # Find all elements with the specified classes
        content_elements = soup.find_all(class_=("body-field", "row intro"))
        content += "\n".join([element.get_text(strip=True) for element in content_elements])

      return content
  except requests.exceptions.RequestException as e:
      print(f"Failed to retrieve {url}: {e}")
      return ''

@cached(cache=docs_cache)
def load_documents_with_recursive_chunking():
    URLS = os.getenv("FEEDING_URL").split(",")

    texts = get_content_from_urls(URLS)

    # Initialize a RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,  # Indicates that separators are not regex patterns
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]  # Hierarchy of separators
    )
    chunk_text = text_splitter.split_text(texts)
    print(f"Loaded and chunked {len(chunk_text)} chunks from documents")
    return chunk_text

@cached(cache=docs_cache)
def load_documents_with_semantic_chunking():
    print("Loading documents from source")

    # Load the contents from the web
    documents = []
    for url in URLS:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract semantic sections
        sections = []
        for element in soup.find_all(['.body-field','.row .intro']):
            text = element.get_text(strip=True)
            if text:
                sections.append(text)

        # Combine sections into a single document
        full_text = "\n\n".join(sections)
        documents.append(full_text)

    return documents


def cache_vectorstore_and_embeddings_from_docs(docs):
    DB_PATH = "vectorstores_docs/db/"
    if os.path.exists(DB_PATH):
        shutil.rmtree("vectorstores_docs")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, persist_directory=DB_PATH, embedding=embeddings)
    return vectorstore

def cache_vectorstore_and_embeddings_from_text(texts):
    DB_PATH = "vectorstores_texts/db/"
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree("vectorstores_texts")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=texts, persist_directory=DB_PATH, embedding=embeddings)
    return vectorstore