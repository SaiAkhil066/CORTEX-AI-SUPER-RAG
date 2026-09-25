"""
Persistent knowledge bases ("collections") and retrieval-pipeline assembly.

Each collection lives in INDEX_DIR/<name>/:
    index.faiss / index.pkl   FAISS vector index (LangChain save_local format)
    chunks.json               chunk texts + metadata (rebuilds BM25 + graph)
    manifest.json             embedding model, files, suggested questions

Streamlit-free so it can be reused by the eval harness and tests.
"""
from datetime import datetime, timezone
import json
import os
import re
import shutil

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi

from utils.build_graph import build_knowledge_graph
from utils.routing import build_source_index

INDEX_DIR = os.getenv("INDEX_DIR", "indexes")
DEFAULT_COLLECTION = "default"


def sanitize_name(name):
    """Collection names become directory names: keep them boring."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-").lower()
    return cleaned[:48] or DEFAULT_COLLECTION


def collection_path(name, root=None):
    return os.path.join(root or INDEX_DIR, sanitize_name(name))


def list_collections(root=None):
    root = root or INDEX_DIR
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "manifest.json"))
    )


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_collection(name, vector_store, chunks, manifest, root=None):
    path = collection_path(name, root)
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([{"text": c.page_content, "metadata": c.metadata} for c in chunks], f)
    manifest = {**manifest, "updated": _now()}
    manifest.setdefault("created", manifest["updated"])
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_collection(name, embeddings, root=None):
    """Return (vector_store, chunks, manifest), or None if it doesn't exist."""
    path = collection_path(name, root)
    if not os.path.isfile(os.path.join(path, "manifest.json")):
        return None
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(path, "chunks.json"), encoding="utf-8") as f:
        chunks = [Document(page_content=c["text"], metadata=c["metadata"]) for c in json.load(f)]
    # index.pkl is written by save_collection above, never by a third party.
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store, chunks, manifest


def delete_collection(name, root=None):
    path = collection_path(name, root)
    if os.path.isdir(path):
        shutil.rmtree(path)


def build_pipeline(vector_store, chunks, reranker=None, k=20, bm25_weight=0.4):
    """Hybrid BM25 + FAISS ensemble, plus the entity graph, over the chunks.

    Each leg returns `k` candidates; the fused list is cut to the answer's
    context budget later, after reranking. Fusion is weighted RRF, where a
    leg's score is weight / (60 + rank), so with 0.4/0.6 the vector leg's top
    results outrank BM25-only hits and BM25 mostly reorders. That is deliberate:
    on FinanceBench, raising the BM25 weight pulled in look-alike passages from
    the wrong filings and lowered page-level hit rate (25% → 13% at 0.6).
    """
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split(),
    )
    bm25_retriever.k = k
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": k})],
        weights=[bm25_weight, 1 - bm25_weight],
    )
    return {
        "ensemble": ensemble,
        "vector_store": vector_store,
        "reranker": reranker,
        "knowledge_graph": build_knowledge_graph(chunks),
        "doc_chunks": chunks,
        "source_index": build_source_index(chunks),
        "k": k,
    }
