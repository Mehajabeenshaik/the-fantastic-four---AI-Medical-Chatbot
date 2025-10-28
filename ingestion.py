# ingestion.py - Advanced Hybrid Setup (Amrutha's Core Deliverable)

import os
import shutil
import pickle # For saving the BM25 chunk list
from dotenv import load_dotenv

# LangChain components for loading, splitting, and storage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Configuration ---
load_dotenv() # Load environment variables
DATA_DIR = "./data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./chroma_db"
CHUNKS_FILE = "chunks.pkl"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150 

# --- 2. Data Loading and Chunking ---
print("Starting Advanced Data Loading and preparation...")

try:
    # Check for data
    if not os.listdir(DATA_DIR):
        print(f"ERROR: The '{DATA_DIR}' folder is empty. Please place the medical PDFs inside.")
        exit()

    # Load Documents
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
        
    print(f"Loaded {len(documents)} document pages from {len(os.listdir(DATA_DIR))} files.")

    # Text Splitting (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created a total of {len(chunks)} chunks.")

except Exception as e:
    print(f"An error occurred during loading/chunking: {e}")
    exit() 
    
# --- 3. Vectorization & Storage (ChromaDB - Dense Index) ---

print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
print("--- DIAGNOSTIC: Attempting to initialize embeddings now ---")
# This will download the model the first time it runs
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # <-- ADD THIS ARGUMENT
)
# Clear old database for a fresh run
if os.path.exists(CHROMA_DB_PATH):
    print(f"Clearing old database at {CHROMA_DB_PATH}...")
    shutil.rmtree(CHROMA_DB_PATH) 

# Create and Persist the Vector Store
print("Creating ChromaDB and generating embeddings... This may take a few minutes.")
vectorstore = Chroma.from_documents(
    documents=chunks,              
    embedding=embeddings,          
    persist_directory=CHROMA_DB_PATH 
)
vectorstore.persist()
print(f"✅ ChromaDB (Semantic Index) successfully persisted to {CHROMA_DB_PATH}")

# --- 4. Saving the Chunks for BM25 Index ---

# Save the list of chunks (documents) for the BM25 retriever
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Chunks list saved to {CHUNKS_FILE} for BM25 retrieval.")
print("\n--- INGESTION COMPLETE. READY FOR RAG CHAIN ASSEMBLY (Step 3) ---")