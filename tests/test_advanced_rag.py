from langchain_core.documents import Document

from utils import advanced_rag
from utils.advanced_rag import (
    reciprocal_rank_fusion, cosine_similarity, condense_query, generate_suggested_questions,
    grade_documents,
)


def _docs(*texts):
    return [Document(page_content=t) for t in texts]


def test_rrf_rewards_docs_ranked_high_in_many_lists():
    fused = reciprocal_rank_fusion([_docs("a", "b", "c"), _docs("b", "a", "d"), _docs("b", "e")])
    assert [d.page_content for d in fused][:2] == ["b", "a"]
    assert {d.page_content for d in fused} == {"a", "b", "c", "d", "e"}


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_condense_query_skips_llm_without_history(monkeypatch):
    monkeypatch.setattr(advanced_rag, "_ollama_generate", lambda *a, **k: 1 / 0)
    assert condense_query("what is X?", "", "uri", "m") == "what is X?"


def test_condense_query_uses_first_line_and_falls_back(monkeypatch):
    monkeypatch.setattr(advanced_rag, "_ollama_generate",
                        lambda *a, **k: '"What is the parental leave for secondary caregivers?"\nextra')
    assert condense_query("and for the other one?", "User: parental leave?", "u", "m") == \
        "What is the parental leave for secondary caregivers?"
    monkeypatch.setattr(advanced_rag, "_ollama_generate", lambda *a, **k: "")
    assert condense_query("and for the other one?", "User: hi", "u", "m") == "and for the other one?"


def test_suggested_questions_filters_non_questions(monkeypatch):
    monkeypatch.setattr(advanced_rag, "_ollama_generate", lambda *a, **k:
                        "Here are some questions:\n1. Who founded the company?\n- What is the leave policy?\nok?")
    assert generate_suggested_questions("text", "u", "m") == [
        "Who founded the company?", "What is the leave policy?"]


def test_grade_documents_preserves_order(monkeypatch):
    monkeypatch.setattr(advanced_rag, "_ollama_generate",
                        lambda uri, model, prompt, **k: "yes" if "keep" in prompt else "no")
    verdicts = grade_documents("q", _docs("keep 1", "drop", "keep 2", "drop"), "u", "m")
    assert verdicts == [True, False, True, False]


def test_query_variants_skip_model_preamble(monkeypatch):
    monkeypatch.setattr(advanced_rag, "_ollama_generate", lambda *a, **k:
                        "Here are three alternative search queries:\n1. 3M capex fiscal 2018\n"
                        "2. \"3M purchases of PP&E in 2018\"\nSure:\n3. 3M cash flow statement 2018")
    from utils.advanced_rag import generate_query_variants
    assert generate_query_variants("What was 3M's 2018 capex?", "u", "m", n=3) == [
        "What was 3M's 2018 capex?", "3M capex fiscal 2018", "3M purchases of PP&E in 2018",
        "3M cash flow statement 2018"]
