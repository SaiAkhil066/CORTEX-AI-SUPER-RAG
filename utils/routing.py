"""
Source routing: when a question names a document in the knowledge base
("3M", "American Express", "the handbook"), restrict search to that document.

In a collection of many similar files (annual reports, contracts, manuals),
most passages look alike, so plain search often lands in the wrong file.
File names usually carry the distinguishing name, so we match on those.
"""
import re

_YEAR = re.compile(r"(?:19|20)\d{2}")
# Filename parts that describe the document type rather than its subject
_GENERIC = {"10k", "10q", "8k", "earnings", "dated", "annual", "report", "final", "draft", "v1", "v2", "copy", "doc", "pdf"}


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def source_keys(source):
    """'AMERICANEXPRESS_2022_10K.pdf' → ({'americanexpress'}, {'2022'})."""
    stem = re.sub(r"\.[a-z0-9]+$", "", source, flags=re.I)
    parts = [p for p in re.split(r"[_\-\s.]+", stem) if p]
    years = {m for p in parts for m in _YEAR.findall(p)}
    names = set()
    for p in parts:
        n = _norm(_YEAR.sub("", p))
        n = re.sub(r"^q[1-4]$", "", n)
        if len(n) >= 2 and n not in _GENERIC and not n.isdigit():
            names.add(n)
    # Multi-part names like JOHNSON_JOHNSON → also match "johnsonjohnson"
    joined = _norm("".join(p for p in parts if not _YEAR.search(p) and _norm(p) not in _GENERIC))
    if len(joined) >= 4:
        names.add(joined)
    return names, years


def build_source_index(chunks):
    index = {}
    for c in chunks:
        src = c.metadata.get("source")
        if src and src not in index:
            index[src] = source_keys(src)
    return index


def route(query, index):
    """Return the set of sources the query names, or None if it names none.
    When several files match by name and the query mentions a year that some of
    them carry, keep only those."""
    q = _norm(query)
    q_years = set(_YEAR.findall(query))
    matched = {s for s, (names, _) in index.items() if any(n in q for n in names if len(n) >= 3 or n.isalnum() and any(ch.isdigit() for ch in n))}
    if not matched:
        return None
    if q_years:
        same_year = {s for s in matched if index[s][1] & q_years}
        if same_year:
            return same_year
    return matched
