"""
Query-time retrieval: query rewriting → (RAG-Fusion | HyDE) hybrid search →
GraphRAG merge → neural reranking → CRAG grading.

Stateless: everything comes in through `pipeline` and `config`, so the
Streamlit app and the eval harness share exactly the same code path.
"""
import time

import numpy as np

from utils.build_graph import retrieve_from_graph
from utils.routing import route
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
    "enable_routing": True,       # search only the files a question names
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


def routed_search(query, pipeline, sources, use_bm25=True):
    """Hybrid search restricted to chunks from `sources`, fused like the ensemble."""
    k = pipeline.get("k", 20)
    allowed = lambda md: md.get("source") in sources
    vs = pipeline["vector_store"]
    vec = vs.similarity_search(query, k=k, filter=allowed, fetch_k=min(vs.index.ntotal, 20000))
    if not use_bm25:
        return vec
    bm25 = pipeline["ensemble"].retrievers[0]
    scores = bm25.vectorizer.get_scores(bm25.preprocess_func(query))
    rows = [i for i in np.argsort(scores)[::-1] if allowed(bm25.docs[i].metadata)][:k]
    kw = [bm25.docs[i] for i in rows]
    w_bm25, w_vec = pipeline["ensemble"].weights
    fused, docs = {}, {}
    for weight, ranked in ((w_bm25, kw), (w_vec, vec)):
        for rank, d in enumerate(ranked, start=1):
            docs.setdefault(d.page_content, d)
            fused[d.page_content] = fused.get(d.page_content, 0.0) + weight / (60 + rank)
    return [docs[c] for c, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


def retrieve_documents(query, uri, model, pipeline, config=None, chat_history=""):
    """Return (docs, trace). trace = {search_query, crag, timings}."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["enable_bm25"]:
        ensemble = pipeline["ensemble"]
    else:
        ensemble = pipeline["vector_store"].as_retriever(search_kwargs={"k": pipeline.get("k", 20)})
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

    # ── 0b. Route to the files the question names ──
    sources = None
    if cfg["enable_routing"] and pipeline.get("source_index"):
        sources = route(search_query, pipeline["source_index"])

    def search(q):
        if sources:
            return routed_search(q, pipeline, sources, use_bm25=cfg["enable_bm25"])
        return list(ensemble.invoke(q))

    # ── 1. Base retrieval (RAG-Fusion or single query) ──
    if cfg["enable_fusion"]:
        variants = generate_query_variants(search_query, uri, model, n=3)
        lap("variants")

        def _search(v):
            try:
                return search(v)
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
        docs = search(expanded)
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

    return candidates, {"search_query": search_query, "crag": crag, "timings": timings,
                        "routed_to": sorted(sources) if sources else None}
