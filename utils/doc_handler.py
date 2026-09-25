import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from utils.advanced_rag import (
    contextualize_chunk,
    summarize_document,
    generate_suggested_questions,
    parallel_map,
)
from utils import store
from utils.loaders import LOADERS, SUPPORTED_TYPES, load_path, split_documents
import os
import tempfile

# Cap contextual-retrieval LLM calls so huge uploads don't hang forever
MAX_CONTEXTUAL_CHUNKS = int(os.getenv("MAX_CONTEXTUAL_CHUNKS", "300"))


def reset_documents():
    """Unload the active knowledge base from the session (files stay on disk)."""
    st.session_state.documents_loaded = False
    st.session_state.retrieval_pipeline = None
    st.session_state.manifest = None
    st.session_state.processing = False
    st.session_state.suggested_questions = []
    st.session_state.semantic_cache = []


def _load_file(file):
    """Load one uploaded file into Documents tagged with file name + 1-based page."""
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"unsupported file type '{ext}'")
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file.getbuffer())
        return load_path(path, display_name=file.name)
    finally:
        os.remove(path)


def _apply_contextual_retrieval(texts, documents, llm_uri, llm_model):
    """Prepend an LLM-generated situating sentence to each chunk (Anthropic technique)."""
    per_source_text = {}
    for d in documents:
        src = d.metadata.get("source", "doc")
        per_source_text[src] = per_source_text.get(src, "") + "\n" + d.page_content

    with st.spinner(f"Summarizing {len(per_source_text)} document(s)…"):
        names = list(per_source_text)
        summaries = dict(zip(names, parallel_map(
            lambda n: summarize_document(per_source_text[n], llm_uri, llm_model), names)))

    total = min(len(texts), MAX_CONTEXTUAL_CHUNKS)
    if len(texts) > total:
        st.warning(
            f"Contextual Retrieval applied to the first {total} of {len(texts)} chunks "
            f"(raise MAX_CONTEXTUAL_CHUNKS to cover more)."
        )
    progress = st.progress(0.0, text="Contextualizing chunks…")
    done = 0

    def _work(chunk):
        return contextualize_chunk(chunk.page_content, summaries.get(chunk.metadata.get("source"), ""),
                                   llm_uri, llm_model)

    # Batches keep the progress bar moving while calls run in parallel
    batch = 8
    for start in range(0, total, batch):
        part = texts[start:start + batch]
        for chunk, ctx in zip(part, parallel_map(_work, part)):
            if ctx:
                chunk.metadata["context"] = ctx
                chunk.metadata["original_text"] = chunk.page_content
                chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
        done += len(part)
        progress.progress(done / total, text=f"Contextualizing chunks… {done}/{total}")
    progress.empty()
    return texts


def load_collection_into_session(name, reranker, embedding_model, base_url):
    """Load a saved knowledge base from disk into the session. Returns True on success."""
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    try:
        loaded = store.load_collection(name, embeddings)
    except Exception as e:
        st.error(f"Could not load knowledge base '{name}': {e}")
        return False
    if loaded is None:
        return False
    vector_store, chunks, manifest = loaded
    if manifest.get("embedding_model") not in (None, embedding_model):
        st.warning(
            f"'{name}' was indexed with {manifest['embedding_model']} but the current "
            f"embedding model is {embedding_model}. Results will be poor; re-index or switch models."
        )
    st.session_state.retrieval_pipeline = store.build_pipeline(vector_store, chunks, reranker)
    st.session_state.manifest = manifest
    st.session_state.suggested_questions = manifest.get("suggested_questions", [])
    st.session_state.documents_loaded = True
    st.session_state.semantic_cache = []
    return True


def process_documents(uploaded_files, reranker, embedding_model, base_url,
                      llm_model=None, enable_contextual=False, collection=store.DEFAULT_COLLECTION):
    """Index uploaded files into `collection`, adding to it if it's already loaded."""
    st.session_state.processing = True
    manifest = st.session_state.get("manifest") or {"embedding_model": embedding_model, "files": []}
    known = {f["name"] for f in manifest.get("files", [])}

    documents, added = [], []
    for file in uploaded_files:
        if file.name in known:
            st.info(f"Skipped {file.name}: already in this knowledge base.")
            continue
        try:
            docs = _load_file(file)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
            continue
        if not docs:
            st.warning(f"No extractable text in {file.name} (scanned PDF?).")
            continue
        documents.extend(docs)
        added.append(file.name)

    if not documents:
        st.session_state.processing = False
        return False

    texts = split_documents(documents)

    # 🚀 Contextual Retrieval — enrich each chunk before embedding/indexing
    llm_uri = f"{base_url}/api/generate"
    if enable_contextual and llm_model:
        texts = _apply_contextual_retrieval(texts, documents, llm_uri, llm_model)

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    pipeline = st.session_state.get("retrieval_pipeline")
    try:
        if pipeline:
            vector_store = pipeline["vector_store"]
            vector_store.add_documents(texts)
            chunks = pipeline["doc_chunks"] + texts
        else:
            vector_store = FAISS.from_documents(texts, embeddings)
            chunks = texts
    except Exception as e:
        st.error(f"Failed to build vector store: {e}")
        st.session_state.processing = False
        return False

    for name in added:
        manifest["files"].append({
            "name": name,
            "chunks": sum(1 for t in texts if t.metadata.get("source") == name),
            "contextual": bool(enable_contextual),
        })

    if llm_model:
        sample = "\n".join(d.page_content for d in documents[:6])
        questions = generate_suggested_questions(sample, llm_uri, llm_model)
        if questions:
            manifest["suggested_questions"] = questions

    try:
        manifest = store.save_collection(collection, vector_store, chunks, manifest)
    except Exception as e:
        st.warning(f"Indexed, but could not save to disk: {e}")

    st.session_state.retrieval_pipeline = store.build_pipeline(vector_store, chunks, reranker)
    st.session_state.manifest = manifest
    st.session_state.suggested_questions = manifest.get("suggested_questions", [])
    st.session_state.semantic_cache = []
    st.session_state.documents_loaded = True
    st.session_state.processing = False
    return True
