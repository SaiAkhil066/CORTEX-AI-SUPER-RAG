from langchain_core.documents import Document

from utils.build_graph import extract_entities, build_knowledge_graph, retrieve_from_graph

CHUNKS = [
    Document(page_content="The company was founded by Priya Raman in Pune. This is great."),
    Document(page_content="Priya Raman later hired Tomas Keller to run Berlin."),
    Document(page_content="The weather in Austin is hot. It is sunny."),
]


def test_extract_entities_drops_sentence_starters():
    ents = extract_entities("The company is in Pune. This is NASA work. It was By Design.")
    assert "Pune" in ents and "NASA" in ents
    assert not {"The", "This", "It"} & ents


def test_graph_links_cooccurring_entities():
    G = build_knowledge_graph(CHUNKS)
    assert G.has_edge("Priya Raman", "Tomas Keller")
    assert G.nodes["Priya Raman"]["chunks"] == {0, 1}


def test_retrieve_from_graph_matches_whole_words_only():
    G = build_knowledge_graph(CHUNKS)
    # "is" / "the" must not match every node anymore
    assert retrieve_from_graph("what is the", G, CHUNKS) == []
    hits = retrieve_from_graph("Who is Tomas Keller?", G, CHUNKS)
    assert hits[0] is CHUNKS[1]
    assert CHUNKS[2] not in hits


def test_retrieve_from_graph_empty_graph():
    assert retrieve_from_graph("anything", build_knowledge_graph([]), []) == []
