"""
Query-time retrieval: query rewriting → (RAG-Fusion | HyDE) hybrid search →
GraphRAG merge → neural reranking → CRAG grading.

Stateless: everything comes in through `pipeline` and `config`, so the
Streamlit app and the eval harness share exactly the same code path.
"""
import time

from utils.build_graph import retrieve_from_graph
from utils.advanced_rag import (
    _ollama_generate,
    condense_query,
    generate_query_variants,
    reciprocal_rank_fusion,
    grade_documents,
    parallel_map,
)

DEFAULT_CONFIG = {
    "enable_bm25": True,          # False = vector-only search (naive RAG baseline)
    "enable_condense": True,
    "enable_hyde": True,
    "enable_fusion": False,
    "enable_graph_rag": True,
    "enable_reranking": True,
    "enable_crag": False,
    "max_contexts": 3,
}


def expand_query(query, uri, model):
    """HyDE: append a hypothetical answer to improve recall."""
    expansion = _ollama_generate(
        uri, model,
        f"Generate a concise hypothetical answer (2-3 sentences) to help retrieve relevant documents for: {query}",
        temperature=0.0, timeout=30,
    )
    return f"{query}\n{expansion}" if expansion else query


def retrieve_documents(query, uri, model, pipeline, config=None, chat_history=""):
    """Return (docs, trace). trace = {search_query, crag, timings}."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["enable_bm25"]:
        ensemble = pipeline["ensemble"]
    else:
        ensemble = pipeline["vector_store"].as_retriever(search_kwargs={"k": 5})
    timings = {}
    t = time.perf_counter()

    def lap(stage):
        nonlocal t
        now = time.perf_counter()
        timings[stage] = now - t
        t = now

    # ── 0. Resolve follow-ups ("what about the second one?") ──
    search_query = query
    if cfg["enable_condense"] and chat_history.strip():
        search_query = condense_query(query, chat_history, uri, model)
        lap("rewrite")

    # ── 1. Base retrieval (RAG-Fusion or single query) ──
    if cfg["enable_fusion"]:
        variants = generate_query_variants(search_query, uri, model, n=3)
        lap("variants")

        def _search(v):
            try:
                return ensemble.invoke(v)
            except Exception:
                return None

        ranked_lists = [r for r in parallel_map(_search, variants) if r]
        docs = reciprocal_rank_fusion(ranked_lists) if ranked_lists else []
    else:
        if cfg["enable_hyde"]:
            expanded = expand_query(search_query, uri, model)
            lap("hyde")
        else:
            expanded = search_query
        docs = list(ensemble.invoke(expanded))
    lap("search")

    # ── 2. GraphRAG merge ──
    if cfg["enable_graph_rag"] and pipeline.get("knowledge_graph") is not None:
        graph_docs = retrieve_from_graph(search_query, pipeline["knowledge_graph"], pipeline["doc_chunks"])
        existing = {d.page_content for d in docs}
        for gdoc in graph_docs:
            if gdoc.page_content not in existing:
                docs.append(gdoc)
                existing.add(gdoc.page_content)
        lap("graph")

    # ── 3. Neural reranking ──
    if cfg["enable_reranking"] and pipeline.get("reranker") and docs:
        pairs = [[search_query, d.page_content] for d in docs]
        scores = pipeline["reranker"].predict(pairs)
        docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        lap("rerank")

    candidates = docs[:cfg["max_contexts"]]

    # ── 4. Corrective RAG (CRAG): grade relevance, drop irrelevant ──
    crag = None
    if cfg["enable_crag"] and candidates:
        verdicts = grade_documents(search_query, candidates, uri, model)
        relevant = [d for d, ok in zip(candidates, verdicts) if ok]
        if relevant:
            crag = ("ok", len(relevant), len(candidates))
            candidates = relevant
        else:
            # Nothing graded relevant — flag low confidence, keep best guess
            crag = ("low", 0, len(candidates))
            candidates = candidates[:1]
        lap("crag")

    return candidates, {"search_query": search_query, "crag": crag, "timings": timings}
