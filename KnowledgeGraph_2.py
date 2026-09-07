import os
from typing import List


# 1. Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Document Loading & Chunking Imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Graph Extraction Imports
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

# Official Updated Neo4j Integration Imports
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain


# ==========================================
# STEP 1: Document Loader & Text Chunking
# ==========================================
def load_and_chunk_documents(
    source_type: str, 
    source_path_or_text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[Document]:
    """Loads raw text or files and splits them into chunks."""
    documents = []
    
    if source_type == "text_string":
        documents = [Document(page_content=source_path_or_text, metadata={"source": "user_input"})]
    elif source_type == "txt_file":
        loader = TextLoader(source_path_or_text, encoding="utf-8")
        documents = loader.load()
    elif source_type == "pdf":
        loader = PyPDFLoader(source_path_or_text)
        documents = loader.load()
    else:
        raise ValueError("Invalid source_type. Choose 'text_string', 'txt_file', or 'pdf'.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[STEP 1] Created {len(chunks)} document chunk(s).")
    return chunks


# ==========================================
# STEPS 2 & 3: Entity Extraction & Relationship Mapping
# ==========================================
def extract_graph_elements(documents: List[Document]):
    """Extracts entity nodes and relationships via LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Database", "Company", "Language", "Framework", "Application"],
        allowed_relationships=["DEVELOPED_BY", "USES", "INTEGRATED_WITH", "USED_FOR"]
    )

    print("[STEPS 2 & 3] Extracting entities and relationships via LLM...")
    return llm_transformer.convert_to_graph_documents(documents)


# ==========================================
# STEP 4A: Store in Neo4j AuraDB
# ==========================================
def store_in_neo4j(graph_documents, url, username, password) -> Neo4jGraph:
    """Connects to Neo4j and loads the graph documents."""
    print("[STEP 4A] Connecting to Neo4j AuraDB...")
    graph = Neo4jGraph(url=url, username=username, password=password)

    print("[STEP 4A] Writing nodes and edges to database...")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    graph.refresh_schema()
    return graph


# ==========================================
# STEP 5: Natural Language Querying (KG-RAG)
# ==========================================
def query_knowledge_graph(graph: Neo4jGraph, question: str):
    """
    Translates a natural language question into a Cypher query, 
    executes it against Neo4j, and generates a natural language answer.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize GraphCypherQAChain from langchain_neo4j
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )

    print(f"\n[STEP 5] Querying Knowledge Graph...")
    print(f"Question: '{question}'")
    
    # Run the query chain
    response = chain.invoke({"query": question})
    return response["result"]


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # Validate .env configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")

    if not all([openai_key, neo4j_uri, neo4j_user, neo4j_pass]):
        raise ValueError("Missing environment variables in .env file. Please check configuration.")

    # 1. Corpus Data
    sample_corpus = """
    Graph databases are ideal for Knowledge Graph RAG applications. 
    Neo4j is an open-source graph database management system developed by Neo4j Inc.
    It uses Cypher as its declarative graph query language. 
    LangChain is a framework designed to simplify the creation of applications using large language models.
    LangChain provides integrations with Neo4j to store nodes and edges for RAG applications.
    """

    # 2. Run Pipeline Steps 1 - 4
    chunks = load_and_chunk_documents("text_string", sample_corpus, chunk_size=300, chunk_overlap=50)
    graph_docs = extract_graph_elements(chunks)
    graph_conn = store_in_neo4j(graph_docs, neo4j_uri, neo4j_user, neo4j_pass)

    # 3. Step 5 Execution: Ask Natural Language Questions
    questions = [
        "Who developed Neo4j?",
        "What query language does Neo4j use?",
        "How is LangChain connected to Neo4j?"
    ]

    print("\n=================== QUERYING STAGE ===================")
    for q in questions:
        answer = query_knowledge_graph(graph_conn, q)
        print(f"Answer: {answer}\n" + "-"*50)