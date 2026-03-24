import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re


def process_documents(uploaded_files,reranker,embedding_model, base_url):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process files
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # Text splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)

    # Check if documents were loaded successfully
    if not texts:
        st.error("No text content found in the uploaded documents. Please check if the files are valid.")
        return

    text_contents = [doc.page_content for doc in texts]

    # 🚀 Hybrid Retrieval Setup
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    
    # Vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # BM25 store
    bm25_retriever = BM25Retriever.from_texts(
        text_contents, 
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )

    # Ensemble retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    # Store in session
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,  # Now using the global reranker variable
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts)  # Store Knowledge Graph
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    # ✅ Debugging: Print Knowledge Graph Nodes & Edges
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
