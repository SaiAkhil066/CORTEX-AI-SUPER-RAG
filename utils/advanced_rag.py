"""
Advanced RAG techniques: Contextual Retrieval, RAG-Fusion (RRF),
Corrective RAG (CRAG) grading, conversational query rewriting,
suggested questions and Semantic Caching helpers.

All functions are pure / stateless and degrade gracefully (return safe
fallbacks) if the LLM call fails, so the main pipeline never crashes.
"""
from concurrent.futures import ThreadPoolExecutor
import requests
import math
import re

# Ollama serves a few requests concurrently (OLLAMA_NUM_PARALLEL); more
# workers than that just queue up server-side.
LLM_WORKERS = 4


def _ollama_generate(uri, model, prompt, temperature=0.0, timeout=60):
    """Single non-streamed Ollama completion. Returns '' on failure."""
    try:
        r = requests.post(
            uri,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        return r.json().get("response", "").strip()
    except Exception:
        return ""


def parallel_map(fn, items, workers=LLM_WORKERS):
    """Order-preserving threaded map for I/O-bound LLM calls."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


# ─── 1. Contextual Retrieval (Anthropic) ──────────────────────────────────────
def summarize_document(text, uri, model, max_input=6000):
    """Short LLM summary of a document, used to situate its chunks.
    Falls back to the document's opening text if the LLM is unavailable."""
    head = text.strip()[:max_input]
    prompt = (
        "Summarize the following document in 3-4 sentences. Mention its type, "
        "subject, and the main entities it discusses.\n\n"
        f"<document>\n{head}\n</document>\n\nSummary:"
    )
    summary = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=90)
    return summary or head[:1500]


def contextualize_chunk(chunk_text, doc_summary, uri, model):
    """Generate a short situating context for a chunk given its document summary."""
    prompt = (
        "You are helping index a document for search.\n"
        f"<document_summary>\n{doc_summary}\n</document_summary>\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n\n"
        "Write a SINGLE short sentence (max 25 words) that situates this chunk "
        "within the document so it can be retrieved on its own. "
        "Output only that sentence, nothing else."
    )
    ctx = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=60)
    # Keep it tidy and bounded
    ctx = ctx.replace("\n", " ").strip()
    return ctx[:300]


# ─── 2. Conversational query rewriting ────────────────────────────────────────
def condense_query(query, chat_history, uri, model):
    """Rewrite a follow-up question into a standalone search query using the
    conversation so far. Returns the original query if there is no history
    or the LLM call fails."""
    if not chat_history.strip():
        return query
    prompt = (
        "Given the conversation and a follow-up question, rewrite the follow-up "
        "as a standalone question that can be understood without the conversation. "
        "Resolve pronouns and references. If it is already standalone, return it unchanged. "
        "Output only the rewritten question.\n\n"
        f"Conversation:\n{chat_history[-3000:]}\n\n"
        f"Follow-up question: {query}\n\nStandalone question:"
    )
    out = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=30)
    out = out.strip().splitlines()[0].strip().strip('"').strip() if out.strip() else ""
    return out if 3 <= len(out) <= 500 else query


# ─── 3. RAG-Fusion: multi-query + Reciprocal Rank Fusion ──────────────────────
def generate_query_variants(query, uri, model, n=3):
    """Return [original] + up to n reworded search queries."""
    prompt = (
        f"Generate {n} alternative search queries that capture different phrasings "
        f"or sub-aspects of the question below. One query per line, no numbering.\n\n"
        f"Question: {query}\n\nQueries:"
    )
    out = _ollama_generate(uri, model, prompt, temperature=0.4, timeout=45)
    variants = []
    for line in out.splitlines():
        v = _LIST_MARKER.sub("", line.strip()).strip().strip('"')
        # Models often open with "Here are three alternative queries:". Searching
        # with that line pulls the same generic passages into every question.
        if not v or v.endswith(":") or _PREAMBLE.match(v) or v.lower() == query.lower():
            continue
        variants.append(v)
    return [query] + variants[:n]


# "1. ", "2) ", "- ", "* ", "• " at the start of a line (not digits that belong to
# the text, like "3M" or "2022 revenue")
_LIST_MARKER = re.compile(r"^(?:\d{1,2}[.)]|[-*•])\s+")
_PREAMBLE = re.compile(r"^(here (are|is)|sure\b|okay\b|certainly\b|alternative (search )?quer)", re.I)


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked Document lists into one via RRF."""
    scores = {}
    doc_map = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ordered]


# ─── 4. Corrective RAG (CRAG): relevance grading ──────────────────────────────
def grade_document(query, doc_text, uri, model):
    """Return True if the document is relevant to the query (LLM judge)."""
    prompt = (
        "You are a strict relevance grader. Decide whether the document contains "
        "information useful to answer the question.\n"
        f"Question: {query}\n"
        f"Document: {doc_text[:1200]}\n\n"
        "Answer with a single word: 'yes' or 'no'."
    )
    ans = _ollama_generate(uri, model, prompt, temperature=0.0, timeout=30).lower()
    return ans.startswith("y") or "yes" in ans[:6]


def grade_documents(query, docs, uri, model):
    """Grade several documents concurrently. Returns a list of bools."""
    return parallel_map(lambda d: grade_document(query, d.page_content, uri, model), docs)


# ─── 5. Suggested questions ───────────────────────────────────────────────────
def generate_suggested_questions(sample_text, uri, model, n=3):
    """Propose n short questions a reader could ask about the documents."""
    prompt = (
        f"Read the document excerpt below and write {n} short, specific questions "
        "(max 12 words each) that the document can answer. "
        "One question per line, no numbering, no extra text.\n\n"
        f"<excerpt>\n{sample_text[:4000]}\n</excerpt>\n\nQuestions:"
    )
    out = _ollama_generate(uri, model, prompt, temperature=0.3, timeout=60)
    questions = []
    for line in out.splitlines():
        q = _LIST_MARKER.sub("", line.strip()).strip().strip('"')
        if q.endswith("?") and 8 <= len(q) <= 120:
            questions.append(q)
    return questions[:n]


# ─── 6. Semantic Cache helpers ────────────────────────────────────────────────
def cosine_similarity(a, b):
    """Cosine similarity between two equal-length float lists."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
