"""File → Document loading, shared by the app and the eval harness."""
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".md": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
}
SUPPORTED_TYPES = [ext.lstrip(".") for ext in LOADERS]


def load_path(path, display_name=None):
    """Load a file into Documents tagged with its display name and 1-based page."""
    ext = os.path.splitext(display_name or path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"unsupported file type '{ext}'")
    docs = LOADERS[ext](path).load()
    name = display_name or os.path.basename(path)
    for d in docs:
        page = d.metadata.get("page")
        d.metadata = {"source": name}
        if isinstance(page, int):
            d.metadata["page"] = page + 1
    return [d for d in docs if d.page_content.strip()]


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
