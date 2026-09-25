"""
Cortex RAG evaluation harness.

Runs a golden question set through several retrieval configurations and
reports, per configuration:
  • hit rate / MRR  — did the expected source reach the top-k contexts?
  • correctness     — LLM judge, answer vs. reference answer (1-5)
  • faithfulness    — LLM judge, answer supported by retrieved context (1-5)
  • latency         — retrieval and generation seconds per question

Usage:
  python -m eval.run_eval --golden eval/golden_example.jsonl --docs path/to/docs
  python -m eval.run_eval --golden my_set.jsonl --collection contracts-2026 --retrieval-only

Golden set: JSONL, one object per line:
  {"question": "...", "answer": "reference answer",
   "source": "file.pdf", "page": 3, "keywords": ["optional", "must-appear terms"]}
`source`/`page`/`keywords` drive retrieval metrics; `answer` drives correctness.
"""
from datetime import datetime
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

from utils import store  # noqa: E402
from utils.loaders import SUPPORTED_TYPES, load_path, split_documents  # noqa: E402
from utils.retriever_pipeline import retrieve_documents  # noqa: E402
from utils.generation import build_prompt, format_context, estimate_num_ctx, ThinkStreamParser  # noqa: E402
from utils.advanced_rag import _ollama_generate  # noqa: E402
from eval.metrics import first_hit_rank, summarize_retrieval, parse_score, mean  # noqa: E402

load_dotenv(find_dotenv())

BASE = {
    "enable_condense": False,
    "enable_hyde": False,
    "enable_fusion": False,
    "enable_graph_rag": False,
    "enable_reranking": False,
    "enable_crag": False,
}
CONFIGS = {
    "hybrid": {},
    "+hyde": {"enable_hyde": True},
    "+graph": {"enable_graph_rag": True},
    "+rerank": {"enable_reranking": True},
    "+fusion": {"enable_fusion": True},
    "+crag": {"enable_crag": True},
    "app-default": {"enable_hyde": True, "enable_graph_rag": True, "enable_reranking": True},
    "full": {"enable_fusion": True, "enable_graph_rag": True, "enable_reranking": True, "enable_crag": True},
}

CORRECTNESS_PROMPT = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Score the candidate from 1 to 5:
5 = fully correct and complete, 4 = correct with minor omissions,
3 = partially correct, 2 = mostly wrong, 1 = wrong or no answer.
Reply with only the number."""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by its context.
Context:
{context}

Answer: {answer}

Score from 1 to 5 how well EVERY claim in the answer is supported by the context:
5 = fully supported, 3 = partly supported, 1 = unsupported or contradicts the context.
If the answer says the context doesn't contain the information and that is true, score 5.
Reply with only the number."""


def load_golden(path):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]
    for i, item in enumerate(items):
        if "question" not in item:
            raise ValueError(f"{path}:{i + 1}: missing 'question'")
    return items


def ingest(docs_dir, collection, embeddings, embedding_model):
    paths = [
        os.path.join(docs_dir, f) for f in sorted(os.listdir(docs_dir))
        if os.path.splitext(f)[1].lower().lstrip(".") in SUPPORTED_TYPES
    ]
    if not paths:
        raise SystemExit(f"No supported files ({', '.join(SUPPORTED_TYPES)}) in {docs_dir}")
    documents = []
    for p in paths:
        documents.extend(load_path(p))
    chunks = split_documents(documents)
    print(f"Indexing {len(paths)} file(s) → {len(chunks)} chunks into '{collection}'…")
    vector_store = FAISS.from_documents(chunks, embeddings)
    manifest = {
        "embedding_model": embedding_model,
        "files": [{"name": os.path.basename(p),
                   "chunks": sum(1 for c in chunks if c.metadata["source"] == os.path.basename(p)),
                   "contextual": False} for p in paths],
    }
    store.save_collection(collection, vector_store, chunks, manifest)


def load_reranker(model_name):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception as e:
        print(f"! Reranker unavailable ({e.__class__.__name__}); rerank configs will be skipped.")
        return None


def generate(question, docs, crag, uri, model):
    prompt = build_prompt(question, format_context(docs), crag_low=bool(crag and crag[0] == "low"))
    try:
        import requests
        r = requests.post(uri, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_ctx": estimate_num_ctx(prompt)},
        }, timeout=300)
        parser = ThinkStreamParser()
        parser.feed(r.json().get("response", ""))
        return parser.finish()[1]
    except Exception as e:
        return f"[generation failed: {e}]"


def run(args):
    base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    uri = f"{base_url}/api/generate"
    model = args.model or os.getenv("MODEL", "llama3.1:8b")
    judge = args.judge_model or model
    embedding_model = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    if args.docs and (args.reindex or args.collection not in store.list_collections()):
        ingest(args.docs, args.collection, embeddings, embedding_model)
    loaded = store.load_collection(args.collection, embeddings)
    if loaded is None:
        raise SystemExit(f"Knowledge base '{args.collection}' not found. Pass --docs DIR to build it.")
    vector_store, chunks, _ = loaded

    configs = {n: CONFIGS[n] for n in (args.configs or CONFIGS)}
    needs_reranker = any(c.get("enable_reranking") for c in configs.values())
    reranker = load_reranker(os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")) \
        if needs_reranker else None
    if reranker is None:
        configs = {n: c for n, c in configs.items() if not c.get("enable_reranking")}
    pipeline = store.build_pipeline(vector_store, chunks, reranker)

    golden = load_golden(args.golden)
    print(f"{len(golden)} questions × {len(configs)} configs · model={model} judge={judge}\n")

    results = {}
    for name, overrides in configs.items():
        cfg = {**BASE, **overrides, "max_contexts": args.k}
        rows = []
        for i, item in enumerate(golden, start=1):
            t0 = time.perf_counter()
            docs, info = retrieve_documents(item["question"], uri, model, pipeline, cfg)
            t_retrieve = time.perf_counter() - t0
            row = {
                "question": item["question"],
                "rank": first_hit_rank(docs, item),
                "retrieved": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
                "retrieve_s": t_retrieve,
            }
            if not args.retrieval_only:
                t1 = time.perf_counter()
                answer = generate(item["question"], docs, info.get("crag"), uri, model)
                row["generate_s"] = time.perf_counter() - t1
                row["answer"] = answer
                if item.get("answer"):
                    row["correctness"] = parse_score(_ollama_generate(uri, judge, CORRECTNESS_PROMPT.format(
                        question=item["question"], reference=item["answer"], answer=answer), timeout=120))
                row["faithfulness"] = parse_score(_ollama_generate(uri, judge, FAITHFULNESS_PROMPT.format(
                    context=format_context(docs)[:8000], answer=answer), timeout=120))
            rows.append(row)
            print(f"  [{name}] {i}/{len(golden)} rank={row['rank']}", end="\r")
        summary = summarize_retrieval([r["rank"] for r in rows if _has_retrieval_target(golden, r)])
        summary["retrieve_s"] = mean([r["retrieve_s"] for r in rows])
        if not args.retrieval_only:
            summary["correctness"] = mean([r.get("correctness") for r in rows])
            summary["faithfulness"] = mean([r.get("faithfulness") for r in rows])
            summary["generate_s"] = mean([r.get("generate_s") for r in rows])
        results[name] = {"config": cfg, "summary": summary, "rows": rows}
        print(" " * 60, end="\r")
        print(f"✓ {name}")

    report = render_report(results, args, model, judge, len(golden))
    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved to {out_dir}")


def _has_retrieval_target(golden, row):
    item = next(g for g in golden if g["question"] == row["question"])
    return bool(item.get("source") or item.get("keywords"))


def _fmt(v, pct=False, digits=2):
    if v is None:
        return "—"
    return f"{v:.0%}" if pct else f"{v:.{digits}f}"


def render_report(results, args, model, judge, n):
    lines = [
        f"# Cortex RAG eval — {args.collection}",
        "",
        f"{n} questions · top-{args.k} contexts · model `{model}` · judge `{judge}`",
        "",
    ]
    if args.retrieval_only:
        lines += ["| Config | Hit rate | MRR | Retrieve (s) |", "|---|---|---|---|"]
        for name, r in results.items():
            s = r["summary"]
            lines.append(f"| {name} | {_fmt(s['hit_rate'], True)} | {_fmt(s['mrr'])} | {_fmt(s['retrieve_s'], digits=1)} |")
    else:
        lines += ["| Config | Hit rate | MRR | Correctness (1-5) | Faithfulness (1-5) | Retrieve (s) | Generate (s) |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            s = r["summary"]
            lines.append(
                f"| {name} | {_fmt(s['hit_rate'], True)} | {_fmt(s['mrr'])} | {_fmt(s['correctness'])} | "
                f"{_fmt(s['faithfulness'])} | {_fmt(s['retrieve_s'], digits=1)} | {_fmt(s['generate_s'], digits=1)} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", required=True, help="JSONL golden question set")
    p.add_argument("--collection", default="eval", help="knowledge base to evaluate (default: eval)")
    p.add_argument("--docs", help="folder of documents to index into --collection if it doesn't exist")
    p.add_argument("--reindex", action="store_true", help="rebuild --collection from --docs")
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="subset of configs to run")
    p.add_argument("--k", type=int, default=5, help="contexts passed to the LLM (default: 5)")
    p.add_argument("--model", help="generation model (default: $MODEL)")
    p.add_argument("--judge-model", help="LLM judge model (default: same as --model)")
    p.add_argument("--retrieval-only", action="store_true", help="skip generation and LLM judging")
    p.add_argument("--out", default=os.path.join("eval", "results"), help="output folder")
    run(p.parse_args())


if __name__ == "__main__":
    main()
