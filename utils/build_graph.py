"""
Lightweight GraphRAG: an entity co-occurrence graph over document chunks.

Entities are capitalised phrases / acronyms (heuristic NER, no extra deps).
Two entities are linked when they appear in the same chunk; every node
remembers which chunks mention it, so graph hits map straight back to text.
"""
from itertools import combinations
import networkx as nx
import re

# Capitalised words that are sentence starters / function words, not entities.
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "there", "here", "it", "its",
    "is", "are", "was", "were", "be", "been", "i", "we", "you", "he", "she", "they",
    "our", "your", "their", "his", "her", "my", "me", "us", "them", "if", "in", "on",
    "at", "of", "for", "to", "from", "by", "with", "and", "or", "but", "not", "no",
    "as", "so", "all", "any", "each", "some", "such", "when", "where", "what", "which",
    "who", "whom", "why", "how", "also", "however", "therefore", "thus", "after",
    "before", "during", "while", "then", "than", "may", "can", "will", "shall",
    "should", "would", "could", "must", "do", "does", "did", "has", "have", "had",
    "yes", "note", "figure", "table", "section", "chapter", "page", "see", "one",
    "two", "first", "second", "new", "about", "into", "over", "under", "between",
}

# Acronyms (NASA, GPT4) or runs of Capitalised words (New York City).
_ENTITY_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Skip co-occurrence edges for chunks with more entities than this (keeps the
# graph from exploding on index pages / tables of contents).
_MAX_ENTITIES_PER_CHUNK = 40


def extract_entities(text):
    """Return the set of normalised entity strings found in text."""
    entities = set()
    for match in _ENTITY_RE.findall(text):
        words = [w for w in match.split() if w.lower() not in _STOPWORDS]
        if not words:
            continue
        entity = " ".join(words)
        if len(entity) >= 3 or entity.isupper():
            entities.add(entity)
    return entities


def _tokens(text):
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) >= 3}


def build_knowledge_graph(docs):
    G = nx.Graph()
    for idx, doc in enumerate(docs):
        entities = sorted(extract_entities(doc.page_content))
        for entity in entities:
            if entity not in G:
                G.add_node(entity, chunks=set(), tokens=_tokens(entity))
            G.nodes[entity]["chunks"].add(idx)
        if len(entities) <= _MAX_ENTITIES_PER_CHUNK:
            for a, b in combinations(entities, 2):
                weight = G[a][b]["weight"] + 1 if G.has_edge(a, b) else 1
                G.add_edge(a, b, weight=weight)
    return G


def retrieve_from_graph(query, G, doc_chunks, top_k=3):
    """Return chunks mentioning entities from the query or their strongest neighbours.

    Chunks are scored by how many matched entities they mention (direct matches
    count double), so the most entity-dense chunks come first.
    """
    q_tokens = _tokens(query)
    if not q_tokens or G.number_of_nodes() == 0:
        return []

    matched = [n for n, data in G.nodes(data=True)
               if data.get("tokens") and data["tokens"] <= q_tokens]
    if not matched:
        # Partial match: any whole-word overlap with a multi-word entity
        matched = [n for n, data in G.nodes(data=True) if data.get("tokens", set()) & q_tokens]
    if not matched:
        return []

    scores = {}
    for node in matched:
        for idx in G.nodes[node]["chunks"]:
            scores[idx] = scores.get(idx, 0) + 2
        neighbours = sorted(G[node].items(), key=lambda kv: kv[1].get("weight", 1), reverse=True)[:5]
        for neighbour, _ in neighbours:
            for idx in G.nodes[neighbour]["chunks"]:
                scores[idx] = scores.get(idx, 0) + 1

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [doc_chunks[idx] for idx, _ in ranked[:top_k] if idx < len(doc_chunks)]
