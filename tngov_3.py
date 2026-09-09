import json
import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

VECTOR_DB_DIR = "faiss_index_tn_schemes"


def load_scraped_documents(file_path="tn_agriculture_schemes.json"):
    """Load json records into standard LangChain Document format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing '{file_path}'. Please run Step 2 scraper first.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        entities = item.get("structured_entities", {})
        content = item.get("unstructured_content", "")
        
        # Build composite page content enriched with key context
        full_text = (
            f"Scheme Name: {entities.get('scheme_name', '')}\n"
            f"Department: {entities.get('department', '')}\n"
            f"Target Group: {entities.get('beneficiary_type', '')}\n"
            f"Eligibility: {entities.get('eligibility', '')}\n"
            f"Subsidies: {entities.get('subsidy_details', '')}\n"
            f"Details: {content}"
        )

        metadata = {
            "source_url": item.get("source_url", ""),
            "scheme_name": entities.get("scheme_name", "Unknown")
        }

        documents.append(Document(page_content=full_text, metadata=metadata))

    return documents


def build_and_save_vector_store():

    load_dotenv()
    print("Loading scraped document records...")
    raw_docs = load_scraped_documents()
    print(f"Loaded {len(raw_docs)} documents.")

    # Split text into chunks optimized for scheme retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Generated {len(chunks)} text chunks.")

    print("Generating OpenAI Embeddings and building FAISS Vector Index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save index locally for Streamlit backend initialization
    vector_store.save_local(VECTOR_DB_DIR)
    print(f"Vector Store successfully created and saved at './{VECTOR_DB_DIR}'")


if __name__ == "__main__":
    build_and_save_vector_store()