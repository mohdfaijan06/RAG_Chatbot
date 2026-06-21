import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings ## vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS #FAISS: Facebook AI Similarity Search

DB_PATH = "faizan_db" 

def load_document():
    loader=PyPDFLoader('data/Faizan_About_Me.pdf')
    document = loader.load()
    return document

# The document is a list of Document objects, we need to extract the page_content from each
# texts = [doc.page_content for doc in document]
def chunking(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 220, chunk_overlap = 50)
    chunk_document = text_splitter.split_documents(document)
    return chunk_document

def create_vector_db(api_key, chunk_document, model = 'text-embedding-ada-002'):
    # this line is creating FAISS db, using converted chunks documents into vector (or embedding) with the help of text-embedding-ada-002 (or any other model you can use)
    embeddings = OpenAIEmbeddings(model = model, api_key = api_key)
    db = FAISS.from_documents(chunk_document, embeddings) #
    return db

def get_db_implementation(api_key):
    if os.path.exists(DB_PATH):
        print("Loading existing FAISS index...")
        db = FAISS.load_local(DB_PATH, OpenAIEmbeddings(api_key=api_key, model='text-embedding-ada-002'), allow_dangerous_deserialization=True)
        return db

    document = load_document()
    chunk_document = chunking(document)
    db = create_vector_db(api_key, chunk_document)
    db.save_local(DB_PATH)
    return db