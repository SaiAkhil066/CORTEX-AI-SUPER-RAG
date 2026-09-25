import os

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS

from utils import store, advanced_rag
from utils.loaders import load_path, split_documents
from utils.retriever_pipeline import retrieve_documents

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "eval", "sample_docs", "northwind_handbook.md")
OFF = {"enable_condense": False, "enable_hyde": False, "enable_fusion": False,
       "enable_graph_rag": False, "enable_reranking": False, "enable_crag": False}


def _chunks():
    return split_documents(load_path(SAMPLE))


def _pipeline(chunks):
    return store.build_pipeline(FAISS.from_documents(chunks, DeterministicFakeEmbedding(size=32)), chunks)


def test_loader_tags_source_and_handles_uppercase_ext(tmp_path):
    upper = tmp_path / "NOTES.MD"
    upper.write_text("Hello Pune", encoding="utf-8")
    docs = load_path(str(upper))
    assert docs[0].metadata == {"source": "NOTES.MD"}
    assert all(c.metadata["source"] == "northwind_handbook.md" for c in _chunks())


def test_sanitize_name():
    assert store.sanitize_name("  My KB / 2026!! ") == "my-kb-2026"
    assert store.sanitize_name("///") == store.DEFAULT_COLLECTION
    assert store.sanitize_name("../../etc") == "etc"


def test_collection_roundtrip(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    chunks = _chunks()
    vs = FAISS.from_documents(chunks, emb)
    store.save_collection("My KB", vs, chunks, {"embedding_model": "fake", "files": []}, root=str(tmp_path))
    assert store.list_collections(root=str(tmp_path)) == ["my-kb"]

    vs2, chunks2, manifest = store.load_collection("my-kb", emb, root=str(tmp_path))
    assert [c.page_content for c in chunks2] == [c.page_content for c in chunks]
    assert chunks2[0].metadata["source"] == "northwind_handbook.md"
    assert manifest["embedding_model"] == "fake" and "updated" in manifest
    assert vs2.index.ntotal == len(chunks)

    store.delete_collection("my-kb", root=str(tmp_path))
    assert store.list_collections(root=str(tmp_path)) == []
    assert store.load_collection("my-kb", emb, root=str(tmp_path)) is None


def test_retrieval_finds_keyword_match_with_metadata():
    pipeline = _pipeline(_chunks())
    docs, trace = retrieve_documents("Daniel Okafor security team", "u", "m", pipeline,
                                     {**OFF, "max_contexts": 3})
    assert len(docs) <= 3
    assert any("Daniel Okafor" in d.page_content for d in docs)   # BM25 leg
    assert all(d.metadata.get("source") == "northwind_handbook.md" for d in docs)
    assert trace["search_query"] == "Daniel Okafor security team"
    assert "search" in trace["timings"]


def test_retrieval_rewrites_followups_and_applies_crag(monkeypatch):
    pipeline = _pipeline(_chunks())

    def fake_llm(uri, model, prompt, **k):
        if "Standalone question" in prompt:
            return "Who leads the security team at Northwind?"
        return "yes" if "Okafor" in prompt else "no"

    monkeypatch.setattr(advanced_rag, "_ollama_generate", fake_llm)
    docs, trace = retrieve_documents(
        "who leads it?", "u", "m", pipeline,
        {**OFF, "enable_condense": True, "enable_crag": True, "enable_graph_rag": True, "max_contexts": 4},
        chat_history="User: tell me about the security team",
    )
    assert trace["search_query"] == "Who leads the security team at Northwind?"
    assert trace["crag"][0] == "ok"
    assert docs and all("Okafor" in d.page_content for d in docs)


def test_build_pipeline_bm25_keeps_metadata():
    chunks = [Document(page_content="alpha beta", metadata={"source": "a.txt"}),
              Document(page_content="gamma delta", metadata={"source": "b.txt", "page": 2})]
    pipeline = store.build_pipeline(FAISS.from_documents(chunks, DeterministicFakeEmbedding(size=8)), chunks, k=1)
    bm25 = pipeline["ensemble"].retrievers[0]
    assert bm25.invoke("gamma")[0].metadata == {"source": "b.txt", "page": 2}
