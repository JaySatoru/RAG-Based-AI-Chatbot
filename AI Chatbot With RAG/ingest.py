from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

documents = []

for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader("documents/" + file)
        documents.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# Generate embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Documents successfully stored!")