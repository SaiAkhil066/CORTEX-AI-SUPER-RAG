"""
Cortex RAG evaluation harness.

Runs a golden question set through several retrieval configurations and
reports, per configuration:
  • hit rate / MRR  — did the expected source (and page) reach the top-k contexts?
  • doc hit rate    — did any chunk of the expected file reach the top-k?
  • accuracy        — share of answers the LLM judge scores ≥ 4/5 vs. the reference
  • correctness     — mean LLM-judge score, answer vs. reference answer (1-5)
  • faithfulness    — LLM judge, answer supported by retrieved context (1-5)
  • latency         — retrieval and generation seconds per question

Usage:
  python -m eval.run_eval --golden eval/golden_example.jsonl --docs path/to/docs
  python -m eval.run_eval --golden my_set.jsonl --collection contracts-2026 --retrieval-only

Golden set: JSONL, one object per line:
  {"question": "...", "answer": "reference answer",
   "source": "file.pdf", "page": 3, "keywords": ["optional", "must-appear terms"],
   "question_type": "optional grouping label"}
`source`/`page`/`keywords` drive retrieval metrics; `answer` drives correctness.

Long runs are safe to interrupt: indexing checkpoints every few files and
resumes where it stopped, and each config streams its rows to disk as it goes.
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
import requests  # noqa: E402

from utils import store  # noqa: E402
from utils.loaders import SUPPORTED_TYPES, load_path, split_documents  # noqa: E402
from utils.retriever_pipeline import retrieve_documents  # noqa: E402
from utils.generation import build_prompt, format_context, estimate_num_ctx, ThinkStreamParser  # noqa: E402
from utils.advanced_rag import _ollama_generate  # noqa: E402
from eval.metrics import (  # noqa: E402
    first_hit_rank, first_doc_hit_rank, summarize_retrieval, parse_score, mean,
)

load_dotenv(find_dotenv())

BASE = {
    "enable_bm25": True,
    "enable_condense": False,
    "enable_routing": False,
    "enable_hyde": False,
    "enable_fusion": False,
    "enable_graph_rag": False,
    "enable_reranking": False,
    "enable_crag": False,
}
CONFIGS = {
    "naive": {"enable_bm25": False},
    "hybrid": {},
    "+hyde": {"enable_hyde": True},
    "+graph": {"enable_graph_rag": True},
    "+rerank": {"enable_reranking": True},
    "+routing": {"enable_routing": True},
    "routing+rerank": {"enable_routing": True, "enable_reranking": True},
    "routing+rerank+fusion": {"enable_routing": True, "enable_reranking": True, "enable_fusion": True},
    "+fusion": {"enable_fusion": True},
    "+crag": {"enable_crag": True},
    "app-default": {"enable_routing": True, "enable_hyde": True, "enable_graph_rag": True, "enable_reranking": True},
    "full": {"enable_routing": True, "enable_fusion": True, "enable_graph_rag": True,
             "enable_reranking": True, "enable_crag": True},
}
ACCURATE_AT = 4   # judge score counted as a correct answer

CORRECTNESS_PROMPT = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Score the candidate from 1 to 5:
5 = fully correct and complete, 4 = correct with minor omissions,
3 = partially correct, 2 = mostly wrong, 1 = wrong or no answer.
Numbers must match the reference (small rounding differences are fine).
Reply with only the number."""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by its context.
Context:
{context}

Answer: {answer}

Score from 1 to 5 how well EVERY claim in the answer is supported by the context:
5 = fully supported, 3 = partly supported, 1 = unsupported or contradicts the context.
If the answer says the context doesn't contain the information and that is true, score 5.
Reply with only the number."""


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ollama_up(base_url):
    try:
        return requests.get(f"{base_url}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def load_golden(path):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]
    for i, item in enumerate(items):
        if "question" not in item:
            raise ValueError(f"{path}:{i + 1}: missing 'question'")
    return items


def ingest(docs_dir, collection, embeddings, embedding_model, reindex=False, checkpoint_every=5, batch=256):
    """Index every supported file in docs_dir into `collection`. Resumable: files
    already recorded in the collection's manifest are skipped."""
    names = [f for f in sorted(os.listdir(docs_dir))
             if os.path.splitext(f)[1].lower().lstrip(".") in SUPPORTED_TYPES]
    if not names:
        raise SystemExit(f"No supported files ({', '.join(SUPPORTED_TYPES)}) in {docs_dir}")

    if reindex:
        store.delete_collection(collection)
    loaded = store.load_collection(collection, embeddings)
    if loaded:
        vector_store, chunks, manifest = loaded
    else:
        vector_store, chunks, manifest = None, [], {"embedding_model": embedding_model, "files": []}
    done = {f["name"] for f in manifest["files"]}
    todo = [n for n in names if n not in done]
    if not todo:
        log(f"'{collection}' already indexed: {len(done)} files, {len(chunks)} chunks")
        return
    log(f"Indexing {len(todo)} file(s) into '{collection}' ({len(done)} already done)")

    t0, embedded = time.perf_counter(), 0
    for i, name in enumerate(todo, start=1):
        try:
            new = split_documents(load_path(os.path.join(docs_dir, name)))
        except Exception as e:
            log(f"  ! skipped {name}: {e}")
            continue
        for s in range(0, len(new), batch):
            part = new[s:s + batch]
            if vector_store is None:
                vector_store = FAISS.from_documents(part, embeddings)
            else:
                vector_store.add_documents(part)
        chunks.extend(new)
        embedded += len(new)
        manifest["files"].append({"name": name, "chunks": len(new), "contextual": False})
        rate = embedded / (time.perf_counter() - t0)
        remaining = sum(1 for _ in todo[i:])
        log(f"  [{i}/{len(todo)}] {name}: {len(new)} chunks · {rate:.1f} chunks/s · "
            f"~{remaining * (embedded / i) / max(rate, 0.01) / 60:.0f} min left")
        if i % checkpoint_every == 0 or i == len(todo):
            store.save_collection(collection, vector_store, chunks, manifest)
    log(f"Indexed '{collection}': {len(manifest['files'])} files, {len(chunks)} chunks")


def load_reranker(model_name):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception as e:
        log(f"! Reranker unavailable ({e.__class__.__name__}: {e}); rerank configs will be skipped.")
        return None


def generate(question, docs, crag, uri, model):
    prompt = build_prompt(question, format_context(docs), crag_low=bool(crag and crag[0] == "low"))
    try:
        r = requests.post(uri, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_ctx": estimate_num_ctx(prompt)},
        }, timeout=900)
        parser = ThinkStreamParser()
        parser.feed(r.json().get("response", ""))
        return parser.finish()[1]
    except Exception as e:
        return f"[generation failed: {e}]"


def summarize(rows, golden_by_q, retrieval_only):
    targeted = [r for r in rows if golden_by_q[r["question"]].get("source") or golden_by_q[r["question"]].get("keywords")]
    s = summarize_retrieval([r["rank"] for r in targeted])
    docs = [r for r in targeted if golden_by_q[r["question"]].get("source")]
    s["doc_hit_rate"] = summarize_retrieval([r["doc_rank"] for r in docs])["hit_rate"] if docs else None
    s["retrieve_s"] = mean([r["retrieve_s"] for r in rows])
    s["n"] = len(rows)
    if not retrieval_only:
        scores = [r.get("correctness") for r in rows if r.get("correctness") is not None]
        s["accuracy"] = sum(1 for v in scores if v >= ACCURATE_AT) / len(scores) if scores else None
        s["correctness"] = mean(scores)
        s["faithfulness"] = mean([r.get("faithfulness") for r in rows])
        s["generate_s"] = mean([r.get("generate_s") for r in rows])
    return s


def run(args):
    base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    uri = f"{base_url}/api/generate"
    model = args.model or os.getenv("MODEL", "llama3.1:8b")
    judge = args.judge_model or model
    embedding_model = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    if not ollama_up(base_url):
        raise SystemExit(f"Ollama is not reachable at {base_url}. Start it with `ollama serve` and retry.")

    if args.docs:
        ingest(args.docs, args.collection, embeddings, embedding_model, reindex=args.reindex)
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
    log(f"Building retrieval pipeline over {len(chunks)} chunks…")
    pipeline = store.build_pipeline(vector_store, chunks, reranker)

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[:args.limit]
    golden_by_q = {g["question"]: g for g in golden}
    mode = "retrieval-only" if args.retrieval_only else "full"
    log(f"{len(golden)} questions × {len(configs)} configs ({mode}) · model={model} judge={judge}")

    results = {}
    if args.resume:
        out_dir = args.resume
        if os.path.isfile(os.path.join(out_dir, "results.json")):
            with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
                results = {n: r for n, r in json.load(f).items() if n in configs}
        log(f"Resuming in {out_dir} ({len(results)} config(s) already complete)")
    else:
        tag = args.tag or f"{args.collection}-{mode}"
        out_dir = os.path.join(args.out, f"{datetime.now():%Y%m%d-%H%M%S}-{tag}")
        os.makedirs(out_dir, exist_ok=True)
        log(f"Writing results to {out_dir}")

    for name, overrides in configs.items():
        if name in results:
            continue
        cfg = {**BASE, **overrides, "max_contexts": args.k}
        rows, t_cfg = [], time.perf_counter()
        rows_path = os.path.join(out_dir, f"rows-{name.replace('+', 'plus-')}.jsonl")
        if args.resume and os.path.isfile(rows_path):
            with open(rows_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            rows = [r for r in rows if r["question"] in golden_by_q]
            if rows:
                log(f"  [{name}] resuming after {len(rows)} saved question(s)")
        done_q = {r["question"] for r in rows}
        resumed = len(rows)
        with open(rows_path, "a" if resumed else "w", encoding="utf-8") as rows_file:
            for i, item in enumerate(golden, start=1):
                if item["question"] in done_q:
                    continue
                t0 = time.perf_counter()
                try:
                    docs, info = retrieve_documents(item["question"], uri, model, pipeline, cfg)
                except Exception as e:
                    if not ollama_up(base_url):
                        # Don't record misses for an outage: stop, so --resume picks up here
                        log(f"  ! Ollama is unreachable at {base_url}; stopping. "
                            f"Start it, then rerun with --resume {out_dir}")
                        sys.exit(3)
                    log(f"  ! [{name}] retrieval failed on q{i}: {e}")
                    docs, info = [], {}
                row = {
                    "question": item["question"],
                    "question_type": item.get("question_type"),
                    "rank": first_hit_rank(docs, item),
                    "doc_rank": first_doc_hit_rank(docs, item),
                    "retrieved": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
                    "retrieve_s": time.perf_counter() - t0,
                    "crag": info.get("crag"),
                }
                if not args.retrieval_only:
                    t1 = time.perf_counter()
                    answer = generate(item["question"], docs, info.get("crag"), uri, model)
                    row["generate_s"] = time.perf_counter() - t1
                    row["answer"] = answer
                    row["reference"] = item.get("answer")
                    if item.get("answer"):
                        row["correctness"] = parse_score(_ollama_generate(uri, judge, CORRECTNESS_PROMPT.format(
                            question=item["question"], reference=item["answer"], answer=answer), timeout=600))
                    row["faithfulness"] = parse_score(_ollama_generate(uri, judge, FAITHFULNESS_PROMPT.format(
                        context=format_context(docs)[:8000], answer=answer), timeout=600))
                rows.append(row)
                rows_file.write(json.dumps(row, default=str) + "\n")
                rows_file.flush()
                if i % 10 == 0 or i == len(golden):
                    el, new = time.perf_counter() - t_cfg, len(rows) - resumed
                    per_q = el / max(new, 1)
                    log(f"  [{name}] {i}/{len(golden)} · {per_q:.1f}s/q · ~{per_q * (len(golden) - i) / 60:.0f} min left")

        summary = summarize(rows, golden_by_q, args.retrieval_only)
        by_type = {}
        for qtype in sorted({r["question_type"] for r in rows if r.get("question_type")}):
            by_type[qtype] = summarize([r for r in rows if r.get("question_type") == qtype],
                                       golden_by_q, args.retrieval_only)
        results[name] = {"config": cfg, "summary": summary, "by_type": by_type}
        log(f"✓ {name}: hit {summary['hit_rate']:.0%} · doc-hit {summary['doc_hit_rate'] or 0:.0%} · MRR {summary['mrr']:.2f}"
            + ("" if args.retrieval_only else f" · accuracy {summary['accuracy'] or 0:.0%}"))

        # Rewrite the report after every config so a partial run is still readable
        report = render_report(results, args, model, judge, len(golden), len(chunks))
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report)

    print("\n" + report)
    log(f"Saved to {out_dir}")


def _fmt(v, pct=False, digits=2):
    if v is None:
        return "—"
    return f"{v:.0%}" if pct else f"{v:.{digits}f}"


def _table(results, retrieval_only, key=None):
    if retrieval_only:
        lines = ["| Config | Page hit | Doc hit | MRR | Retrieve (s) |", "|---|---|---|---|---|"]
    else:
        lines = ["| Config | Page hit | Doc hit | MRR | Accuracy | Correctness (1-5) | Faithfulness (1-5) | Retrieve (s) | Generate (s) |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["by_type"].get(key) if key else r["summary"]
        if not s:
            continue
        row = f"| {name} | {_fmt(s['hit_rate'], True)} | {_fmt(s['doc_hit_rate'], True)} | {_fmt(s['mrr'])} |"
        if not retrieval_only:
            row += f" {_fmt(s['accuracy'], True)} | {_fmt(s['correctness'])} | {_fmt(s['faithfulness'])} |"
        row += f" {_fmt(s['retrieve_s'], digits=1)} |"
        if not retrieval_only:
            row += f" {_fmt(s['generate_s'], digits=1)} |"
        lines.append(row)
    return lines


def render_report(results, args, model, judge, n, n_chunks):
    lines = [
        f"# Cortex RAG eval — {args.collection}",
        "",
        f"{n} questions · {n_chunks} chunks · top-{args.k} contexts · model `{model}`"
        + ("" if args.retrieval_only else f" · judge `{judge}`"),
        "",
        "*Page hit*: a chunk from the cited page is in the top-k. *Doc hit*: any chunk from the right file is."
        + ("" if args.retrieval_only else f" *Accuracy*: judge score ≥ {ACCURATE_AT}/5."),
        "",
        *_table(results, args.retrieval_only),
        "",
    ]
    types = sorted({t for r in results.values() for t in r["by_type"]})
    for qtype in types:
        lines += [f"### {qtype}", "", *_table(results, args.retrieval_only, qtype), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", required=True, help="JSONL golden question set")
    p.add_argument("--collection", default="eval", help="knowledge base to evaluate (default: eval)")
    p.add_argument("--docs", help="folder of documents to index into --collection (resumes if partial)")
    p.add_argument("--reindex", action="store_true", help="rebuild --collection from --docs from scratch")
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="subset of configs to run")
    p.add_argument("--k", type=int, default=5, help="contexts passed to the LLM (default: 5)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--model", help="generation model (default: $MODEL)")
    p.add_argument("--judge-model", help="LLM judge model (default: same as --model)")
    p.add_argument("--retrieval-only", action="store_true", help="skip generation and LLM judging")
    p.add_argument("--tag", help="suffix for the results folder name")
    p.add_argument("--resume", metavar="DIR", help="continue an interrupted run in this results folder")
    p.add_argument("--out", default=os.path.join("eval", "results"), help="output folder")
    run(p.parse_args())


if __name__ == "__main__":
    main()
