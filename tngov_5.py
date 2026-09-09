import os
import json
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS  # FIX 1: Added missing FAISS import
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
#from  langchain_neo4j import GraphCypherQAChain, Neo4jGraph
#from langchain_community.graphs import Neo4jGraph
#from langchain_community.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env if present
load_dotenv()

# Configuration Settings
VECTOR_DB_DIR = "faiss_index_tn_schemes"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://a249ad00.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "a249ad00")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "9R_2TZNIiuFkh-DdoCX8Zk1VTZ4ncQ1zCJV-_sEOi34")


class GraphRAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        # 1. Initialize Vector Retriever
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = FAISS.load_local(
            VECTOR_DB_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # 2. Initialize Neo4j Knowledge Graph QA Chain
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=self.graph,
            verbose=False,
            allow_dangerous_requests=True
        )

        # 3. Final Synthesis Prompt
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """You are an expert AI assistant on Tamil Nadu Government Agriculture Schemes.
Answer the user's question using BOTH the structured Knowledge Graph data and the unstructured Vector context provided.

Knowledge Graph Context (Structured Entities & Relationships):
{graph_context}

Vector Search Context (Unstructured Scheme Details):
{vector_context}

User Question: {question}

Provide a clear, helpful, and well-structured answer. If specific requirements (like eligibility or documents) are found, present them using bullet points.
"""
        )

    def query(self, user_question: str) -> dict:
        """Executes hybrid retrieval: queries Neo4j via Cypher and FAISS via embeddings."""
        # A. Query Graph (Cypher generation & execution)
        graph_response = ""
        try:
            graph_res = self.cypher_chain.invoke({"query": user_question})
            graph_response = graph_res.get("result", "")
        except Exception as e:
            graph_response = f"Graph retrieval error/fallback: {str(e)}"

        # B. Query Vector Store
        vector_docs = self.retriever.invoke(user_question)
        vector_context = "\n\n".join([f"[{doc.metadata.get('scheme_name')}]: {doc.page_content}" for doc in vector_docs])

        # C. Synthesize Final Response
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        synthesis_chain = self.synthesis_prompt | llm

        final_answer = synthesis_chain.invoke({
            "question": user_question,
            "graph_context": graph_response,
            "vector_context": vector_context
        })

        return {
            "answer": final_answer.content,
            "graph_context": graph_response,
            "vector_context": vector_context,
            "source_documents": vector_docs
        }


def init_page():
    st.set_page_config(
        page_title="TN Farmers Scheme GraphRAG Assistant",
        page_icon="🌾",
        layout="wide"
    )
    st.title("🌾 Tamil Nadu Farmers Scheme GraphRAG Assistant")
    st.caption("Powered by Neo4j Knowledge Graph, FAISS Vector Search, and OpenAI")

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_rag_result" not in st.session_state:
        st.session_state.last_rag_result = None
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None


def run_custom_evaluator(question: str, answer: str, graph_context: str, vector_context: str, api_key: str) -> dict:
    """Computes RAG quality metrics (Faithfulness, Relevance, Precision) using LLM-as-a-Judge."""
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    eval_prompt = ChatPromptTemplate.from_template(
        """You are an objective AI evaluator assessing a RAG system for Tamil Nadu Government Farmer Schemes.
Evaluate the output using these three metrics:

1. Faithfulness (0.0 to 1.0): Is the answer fully derived from the provided context without hallucinations?
2. Answer Relevance (0.0 to 1.0): Does the answer directly address the user's question?
3. Context Precision (0.0 to 1.0): How relevant and useful are the retrieved graph and vector contexts?

User Question: {question}
Generated Answer: {answer}
Graph Context: {graph_context}
Vector Context: {vector_context}

Respond ONLY in valid raw JSON format as follows:
{{
  "faithfulness": 0.95,
  "answer_relevance": 0.90,
  "context_precision": 0.85
}}"""
    )

    eval_chain = eval_prompt | eval_llm | StrOutputParser()
    raw_response = eval_chain.invoke({
        "question": question,
        "answer": answer,
        "graph_context": graph_context,
        "vector_context": vector_context
    })

    try:
        clean_json = raw_response.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}


def main():
    init_page()

    # API Key Handling
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API Key or set the `OPENAI_API_KEY` environment variable.")
            return

    # Initialize GraphRAG engine once
    if st.session_state.rag_engine is None:
        with st.spinner("Initializing Neo4j Graph & Vector Store connection..."):
            try:
                st.session_state.rag_engine = GraphRAGEngine(api_key=api_key)
                st.sidebar.success("✅ GraphRAG Engine Connected!")
            except Exception as e:
                st.sidebar.error(f"Engine Connection Error: {e}")
                return

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Handler
    if prompt := st.chat_input("Ask about TN Farmer schemes, eligibility, subsidies, or application steps..."):
        # Display user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process Query through GraphRAG
        with st.chat_message("assistant"):
            with st.spinner("Searching Neo4j Knowledge Graph & Vector Store..."):
                result = st.session_state.rag_engine.query(prompt)
                
                answer = result["answer"]
                st.markdown(answer)

                # Store result for side-by-side inspection & evaluation
                st.session_state.last_rag_result = {
                    "question": prompt,
                    "answer": answer,
                    "graph_context": result["graph_context"],
                    "vector_context": result["vector_context"]
                }
                st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar: Context Inspector & Evaluation Panel
    if st.session_state.last_rag_result:
        res = st.session_state.last_rag_result
        
        with st.sidebar:
            st.divider()
            st.header("🔍 Context Inspector")
            
            with st.expander("🕸️ Neo4j Knowledge Graph Context"):
                st.write(res["graph_context"] if res["graph_context"] else "No direct graph relationship matched.")

            with st.expander("📄 Vector Search Context"):
                st.text(res["vector_context"])

            st.divider()
            st.header("📊 RAG Evaluation Panel")
            if st.button("Run Evaluation Metrics"):
                with st.spinner("Calculating quality metrics..."):
                    eval_scores = run_custom_evaluator(
                        question=res["question"],
                        answer=res["answer"],
                        graph_context=res["graph_context"],
                        vector_context=res["vector_context"],
                        api_key=api_key
                    )

                    # FIX 2: Explicitly scope columns to st.sidebar
                    col1, col2, col3 = st.sidebar.columns(3)
                    col1.metric("Faithfulness", f"{eval_scores.get('faithfulness', 0) * 100:.0f}%")
                    col2.metric("Relevance", f"{eval_scores.get('answer_relevance', 0) * 100:.0f}%")
                    col3.metric("Precision", f"{eval_scores.get('context_precision', 0) * 100:.0f}%")


if __name__ == "__main__":
    main()