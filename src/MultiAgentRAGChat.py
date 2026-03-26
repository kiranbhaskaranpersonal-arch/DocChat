import streamlit as st
import os
from typing import TypedDict, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION & STATE ---
# We use a Pydantic-like State for LangGraph
class AgentState(TypedDict):
    question: str
    context: str
    answer: str

# Initialize Local LLM and Embeddings
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db_path = "./chroma_db"

# --- 2. THE AGENTS (NODES) ---

def retrieval_agent(state: AgentState):
    """Searches the Vector DB for relevant context."""
    print("--- AGENT: RETRIEVER ---")
    db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    docs = db.similarity_search(state["question"], k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generator_agent(state: AgentState):
    """Generates a logical answer based ONLY on the retrieved context."""
    print("--- AGENT: GENERATOR ---")
    prompt = f"""
    You are a professional assistant. Use the context below to answer the user's question.
    If the answer isn't in the context, say you don't know.
    
    Context: {state['context']}
    Question: {state['question']}
    Answer:
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- 3. THE GRAPH (ORCHESTRATION) ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("generate", generator_agent)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
rag_app = workflow.compile()

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("📂 Multi-Agent Document Intelligence")

with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Vectorize Document") and uploaded_file:
        with st.spinner("Processing..."):
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and Split
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            # Create Vector DB
            Chroma.from_documents(chunks, embeddings, persist_directory=vector_db_path)
            st.success("Document successfully indexed!")

# Main Chat Interface
user_query = st.text_input("Ask a question about your documents:")

if user_query:
    with st.spinner("Agents are collaborating..."):
        inputs = {"question": user_query, "context": "", "answer": ""}
        result = rag_app.invoke(inputs)
        
        st.markdown("### 🤖 Agent Response")
        st.write(result["answer"])
        
        with st.expander("View Retrieved Context"):
            st.info(result["context"])