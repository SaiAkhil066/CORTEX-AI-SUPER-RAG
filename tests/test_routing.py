from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS

from utils import store
from utils.routing import source_keys, build_source_index, route
from utils.retriever_pipeline import retrieve_documents

FILES = ["3M_2018_10K.pdf", "3M_2022_10K.pdf", "AMERICANEXPRESS_2022_10K.pdf",
         "JOHNSON_JOHNSON_2022_10K.pdf", "employee_handbook.pdf"]


def _index():
    return build_source_index([Document(page_content="", metadata={"source": f}) for f in FILES])


def test_source_keys():
    assert source_keys("AMERICANEXPRESS_2022_10K.pdf") == ({"americanexpress"}, {"2022"})
    names, years = source_keys("JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf")
    assert "johnsonjohnson" in names and years == {"2022"}


def test_route_by_name_and_year():
    idx = _index()
    assert route("What was 3M's capex in FY2018?", idx) == {"3M_2018_10K.pdf"}
    assert route("How has 3M's dividend changed?", idx) == {"3M_2018_10K.pdf", "3M_2022_10K.pdf"}
    assert route("American Express tax rate in 2022", idx) == {"AMERICANEXPRESS_2022_10K.pdf"}
    assert route("What does the employee handbook say about leave?", idx) == {"employee_handbook.pdf"}


def test_route_none_when_nothing_named():
    assert route("What is the parental leave policy?", _index()) is None


def test_routed_retrieval_stays_in_named_file():
    chunks = [Document(page_content=f"capital expenditure figures for the year {i}", metadata={"source": f, "page": i})
              for i, f in enumerate(FILES * 4)]
    pipe = store.build_pipeline(FAISS.from_documents(chunks, DeterministicFakeEmbedding(size=16)), chunks, k=5)
    docs, trace = retrieve_documents("3M capital expenditure 2022", "u", "m", pipe,
                                     {"enable_condense": False, "enable_hyde": False, "enable_graph_rag": False,
                                      "enable_reranking": False, "enable_crag": False, "max_contexts": 3})
    assert trace["routed_to"] == ["3M_2022_10K.pdf"]
    assert docs and all(d.metadata["source"] == "3M_2022_10K.pdf" for d in docs)
