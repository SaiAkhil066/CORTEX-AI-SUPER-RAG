"""
Prepare the FinanceBench open-source sample for the Cortex eval harness.

  python -m eval.datasets.financebench            # download + convert
  python -m eval.run_eval --golden eval/data/financebench/golden.jsonl \
      --docs eval/data/financebench/pdfs --collection financebench

Downloads the 150 public questions and only the SEC filings they reference
(84 PDFs, not the full 368) into eval/data/financebench/ (gitignored), then
writes a golden set in the harness format:

  {"question", "answer", "source": "<doc>.pdf", "page": <1-based page>, ...}

FinanceBench: Islam et al. 2023, https://arxiv.org/abs/2311.11944
Data: https://github.com/patronus-ai/financebench — annotations are CC-BY-NC-4.0,
so they are downloaded on demand rather than redistributed with this repo.
"""
import argparse
import json
import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{RAW}/data/financebench_open_source.jsonl"
PDF_URL = f"{RAW}/pdfs/{{name}}.pdf"
OUT_DIR = os.path.join("eval", "data", "financebench")


def _download(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return False
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    os.replace(tmp, path)
    return True


def convert(rows, page_offset):
    """FinanceBench rows → harness golden items."""
    golden = []
    for r in rows:
        ev = (r.get("evidence") or [{}])[0]
        page = ev.get("evidence_page_num")
        item = {
            "question": r["question"],
            "answer": r["answer"],
            "source": f"{r['doc_name']}.pdf",
            "id": r["financebench_id"],
            "question_type": r["question_type"],
            "question_reasoning": r.get("question_reasoning"),
        }
        if page not in (None, ""):
            item["page"] = int(page) + page_offset
        golden.append(item)
    return golden


def detect_page_offset(rows, pdf_dir, samples=8):
    """FinanceBench page numbers may be 0- or 1-based; the harness uses 1-based.
    Check which offset puts the evidence text on the cited page."""
    from pypdf import PdfReader

    def norm(s):
        return " ".join(s.split()).lower()

    votes = {0: 0, 1: 0}
    checked = 0
    for r in rows:
        ev = (r.get("evidence") or [{}])[0]
        text, page = ev.get("evidence_text", ""), ev.get("evidence_page_num")
        path = os.path.join(pdf_dir, f"{r['doc_name']}.pdf")
        if not text or page in (None, "") or not os.path.isfile(path):
            continue
        probe = norm(text)[:80]
        reader = PdfReader(path)
        for offset in (0, 1):
            idx = int(page) + offset - 1          # → 0-based index into the PDF
            if 0 <= idx < len(reader.pages) and probe in norm(reader.pages[idx].extract_text() or ""):
                votes[offset] += 1
        checked += 1
        if checked >= samples:
            break
    print(f"page-offset check over {checked} samples: {votes}")
    return max(votes, key=votes.get)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--limit", type=int, help="only the first N questions (and their PDFs)")
    args = p.parse_args()

    pdf_dir = os.path.join(args.out, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    qpath = os.path.join(args.out, "financebench_open_source.jsonl")
    _download(QUESTIONS_URL, qpath)
    with open(qpath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    docs = sorted({r["doc_name"] for r in rows})
    print(f"{len(rows)} questions over {len(docs)} filings")
    for i, name in enumerate(docs, start=1):
        fresh = _download(PDF_URL.format(name=name), os.path.join(pdf_dir, f"{name}.pdf"))
        print(f"  [{i}/{len(docs)}] {name}.pdf {'downloaded' if fresh else 'cached'}", flush=True)

    offset = detect_page_offset(rows, pdf_dir)
    golden = convert(rows, offset)
    gpath = os.path.join(args.out, "golden.jsonl")
    with open(gpath, "w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(golden)} items to {gpath} (page offset +{offset})")


if __name__ == "__main__":
    sys.exit(main())
