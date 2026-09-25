from langchain_core.documents import Document

from eval.metrics import is_relevant, first_hit_rank, summarize_retrieval, parse_score, mean


def _d(text, source="a.pdf", page=None):
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return Document(page_content=text, metadata=meta)


def test_is_relevant_source_page_keywords():
    item = {"source": "a.pdf", "page": 2, "keywords": ["Leave", "24 days"]}
    assert is_relevant(_d("annual leave is 24 days", page=2), item)
    assert not is_relevant(_d("annual leave is 24 days", page=3), item)
    assert not is_relevant(_d("annual leave is 24 days", source="b.pdf", page=2), item)
    assert not is_relevant(_d("annual leave", page=2), item)
    assert not is_relevant(_d("anything"), {"question": "no target"})


def test_rank_and_summary():
    docs = [_d("x"), _d("target here"), _d("target again")]
    assert first_hit_rank(docs, {"keywords": ["target"]}) == 2
    assert first_hit_rank(docs, {"keywords": ["missing"]}) is None
    s = summarize_retrieval([1, 2, None, None])
    assert s["hit_rate"] == 0.5 and abs(s["mrr"] - 0.375) < 1e-9
    assert summarize_retrieval([]) == {"hit_rate": 0.0, "mrr": 0.0}


def test_parse_score_and_mean():
    assert parse_score("Score: 4") == 4
    assert parse_score("10 out of 10, so 5") == 5
    assert parse_score("no idea") is None
    assert mean([1, None, 3]) == 2
    assert mean([None]) is None
