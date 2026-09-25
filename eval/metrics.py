"""Scoring helpers for the eval harness. Pure functions — no LLM, no I/O."""
import re


def is_relevant(doc, item):
    """Does a retrieved chunk satisfy a golden item's expectations?

    An item can pin the expected `source` file (and optionally `page`), list
    `keywords` that must all appear in the chunk, or both.
    """
    source, page, keywords = item.get("source"), item.get("page"), item.get("keywords")
    if not (source or keywords):
        return False
    meta = doc.metadata
    if source and meta.get("source") != source:
        return False
    if page is not None and meta.get("page") != page:
        return False
    if keywords:
        text = doc.page_content.lower()
        if not all(k.lower() in text for k in keywords):
            return False
    return True


def first_hit_rank(docs, item):
    """1-based rank of the first relevant chunk, or None."""
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, item):
            return rank
    return None


def summarize_retrieval(ranks):
    """Hit rate and MRR over a list of first-hit ranks (None = miss)."""
    n = len(ranks)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0}
    hits = [r for r in ranks if r is not None]
    return {
        "hit_rate": len(hits) / n,
        "mrr": sum(1.0 / r for r in hits) / n,
    }


def parse_score(text, low=1, high=5):
    """First integer in [low, high] found in an LLM judge reply, or None."""
    for m in re.findall(r"\d+", text or ""):
        v = int(m)
        if low <= v <= high:
            return v
    return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
