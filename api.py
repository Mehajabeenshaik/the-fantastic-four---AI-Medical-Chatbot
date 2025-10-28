import os
import pickle
from dotenv import load_dotenv

# --- FastAPI Imports ---
from fastapi import FastAPI
from pydantic import BaseModel
# ---------------------------

# LangChain Imports for RAG Chain
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain.schema import format_document

# Hybrid Search (BM25) Imports
from rank_bm25 import BM25Okapi

# --- Configuration & Initialization ---

# 1. Load environment variables
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3") 
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNKS_FILE_PATH = os.getenv("CHUNKS_FILE_PATH", "chunks.pkl")

# 2. Define the LLM Prompt Template for Citation-Backed Answers
RAG_PROMPT_TEMPLATE = """
You are a highly accurate, citation-backed medical assistant. Your task is to answer the user's question ONLY based on the provided context.
You MUST provide the source citation in the format: [Document Name - Page Number] after every piece of information that came from the context.

If the context does not contain the answer, you MUST state, "The provided documents do not contain specific information regarding this topic." Do not use external knowledge.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
"""

# --- FastAPI Initialization ---
app = FastAPI(title="MediCare RAG API")

# Pydantic model for the API request body
class QueryRequest(BaseModel):
    query: str

# Pydantic model for the API response body (including answer and sources)
class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]

# --- Core RAG Logic Functions ---

def load_documents_for_bm25():
    """Loads the documents from the .pkl file to create the BM25 index."""
    try:
        with open(CHUNKS_FILE_PATH, 'rb') as f:
            documents = pickle.load(f)
        return documents
    except FileNotFoundError:
        print(f"CRITICAL WARNING: The required file '{CHUNKS_FILE_PATH}' is missing. Did you run 'python ingestion.py' first?")
        return []
    except Exception as e:
        print(f"Error loading {CHUNKS_FILE_PATH}: {e}")
        return []

def format_docs_for_llm(docs):
    """Formats the retrieved documents into a single string with citations for the LLM prompt."""
    formatted_string = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown Document')
        page_number = doc.metadata.get('page', 'Unknown Page')
        doc_name = os.path.basename(source)
        # Use a consistent citation format for the LLM
        citation = f"[{doc_name} - Page {page_number}]" 
        formatted_string += f"--- Chunk {i+1} {citation} ---\n{doc.page_content}\n"
    return formatted_string

def extract_sources(docs):
    """Extracts citation data from the retrieved documents for the API response."""
    sources_list = []
    seen_citations = set()
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Document')
        page_number = doc.metadata.get('page', 'Unknown Page')
        doc_name = os.path.basename(source)
        citation_key = (doc_name, page_number)
        
        if citation_key not in seen_citations:
            sources_list.append({
                "document_name": doc_name,
                "page_number": page_number,
                "chunk_content": doc.page_content
            })
            seen_citations.add(citation_key)
    return sources_list

# --- Initialization happens once at startup ---

print("Starting RAG System Initialization...")

# 1. Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5") 

# 2. Load Vector Store (ChromaDB)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH, 
    embedding_function=embeddings
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Load Documents for BM25 (Keyword) Index
documents = load_documents_for_bm25()

# 4. Create the BM25 retriever (Sparse)
bm25_retriever_base = BM25Okapi(
    [doc.page_content for doc in documents], 
    b=0.75, 
    k1=1.2, 
)

# Wrapper function for BM25
def bm25_function(query):
    """The original BM25 wrapper function."""
    retrieved_texts = bm25_retriever_base.get_top_n(query, documents, n=5)
    
    unique_docs = []
    seen_content = set()
    for doc in retrieved_texts:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    return unique_docs

# --- CRITICAL FIX: Wrap the function as a Runnable ---
bm25_runnable = RunnableLambda(bm25_function)
# ----------------------------------------------------

# 5. Create the Hybrid Retriever (Ensemble)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_runnable], 
    weights=[0.5, 0.5]
)
print("RAG System Initialization Complete.")

# 6. Initialize LLM (Connects to Ollama)
print(f"Connecting to LLM: {LLM_MODEL} (via Ollama)")
llm = Ollama(model=LLM_MODEL)

# 7. Create RAG Chain (using the corrected LCEL structure)
prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# This chain accepts a string as 'question' (via the map below) and returns an answer string.
# It also stores the retrieved documents for later source extraction.
rag_chain_retrieval_and_generation = (
    # 1. Map the input (which is the query string) to two keys:
    #    - 'context': The documents retrieved by the retriever (which accepts a string query).
    #    - 'question': The original query string.
    RunnablePassthrough.assign(
        context=lambda x: ensemble_retriever.invoke(x) # x is the query string passed from the endpoint
    )
    # 2. Format the context and pass the question to the prompt template
    | {
        "context": lambda x: format_docs_for_llm(x["context"]), # Format documents for the LLM
        "question": lambda x: x["question"], # Pass the original question
    }
    # 3. Invoke LLM
    | prompt
    | llm
    | StrOutputParser()
)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "MediCare RAG API is running. Go to /docs for the query endpoint."}

@app.post("/query", response_model=RAGResponse)
async def query_rag_system(request: QueryRequest):
    """
    Accepts a user query and returns a citation-backed answer using the RAG pipeline.
    """
    question = request.query
    
    # 1. Retrieve the context documents (only retriever used here to get docs for sources)
    retrieved_docs = ensemble_retriever.invoke(question)

    # 2. Invoke the chain for the answer (passes the question string to the chain)
    # The chain handles retrieving context internally (which is why the previous version failed).
    # We invoke the generation chain with the question string.
    answer = rag_chain_retrieval_and_generation.invoke(question)

    # 3. Extract clean sources for the final JSON response
    sources = extract_sources(retrieved_docs)
    
    # 4. Return the structured response
    return RAGResponse(answer=answer, sources=sources)