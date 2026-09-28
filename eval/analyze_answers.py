"""
Stricter scoring for an answer run, independent of the LLM judge's leniency.

  python -m eval.analyze_answers eval/results/<answers-folder>

For each config it reports:
  judge accuracy   share of answers the LLM judge scored >= 4/5 (as run_eval does)
  refusal rate     answers saying the information isn't in the sources
  strict accuracy  judged >= 4 AND not a refusal
  numeric match    for questions whose reference is a number (e.g. "$1577.00",
                   "16.5%"), whether the answer contains it within 2% tolerance;
                   no LLM involved
Small local judges often give refusals a passing score, which is why the strict
numbers exist. FinanceBench itself reports "incorrect or refused" together.
"""
import json
import os
import re
import sys

REFUSAL = re.compile(
    r"(couldn'?t|could not|cannot|can'?t|unable to|not able to) (find|locate|determine|provide|calculate|answer)"
    r"|no (information|mention|data|details) (about|on|regarding|for)"
    r"|(is|are) not (provided|mentioned|available|included|given) in"
    r"|do(es)? not (contain|provide|mention|include) (any |the |enough )?(information|data|details|figures)",
    re.I,
)
NUMBER = re.compile(r"-?\(?\$?\d[\d,]*\.?\d*\)?\s*(%|percent|billion|million|bn|mn)?", re.I)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6}


def is_refusal(answer):
    return bool(REFUSAL.search(answer or ""))


def _numbers(text):
    """All numbers in text, both as written and scaled by million/billion."""
    out = []
    for m in NUMBER.finditer(text or ""):
        raw = m.group(0)
        digits = re.sub(r"[^\d.\-]", "", raw.replace("(", "-").replace(")", ""))
        if not digits or digits in {"-", ".", "-."}:
            continue
        try:
            v = float(digits)
        except ValueError:
            continue
        unit = (m.group(1) or "").lower()
        out.append(v)
        if unit in SCALE:
            out.append(v * SCALE[unit])
    return out


def reference_number(reference):
    """The reference answer's number, if the reference is essentially just a number."""
    ref = (reference or "").strip()
    if not ref or len(ref) > 25:
        return None
    nums = _numbers(ref)
    return nums[0] if nums else None


def numeric_match(answer, reference, tol=0.02):
    target = reference_number(reference)
    if target is None:
        return None
    for v in _numbers(answer):
        for cand in (v, v / 1e3, v * 1e3, v / 1e6, v * 1e6):   # tolerate unit shifts (millions vs billions)
            if target == 0 and cand == 0:
                return True
            if target and abs(abs(cand) - abs(target)) <= tol * abs(target):
                return True
    return False


def analyze(folder):
    report = {}
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("rows-") and name.endswith(".jsonl")):
            continue
        config = name[5:-6].replace("plus-", "+")
        rows = [json.loads(line) for line in open(os.path.join(folder, name), encoding="utf-8") if line.strip()]
        if not rows or "answer" not in rows[0]:
            continue
        judged = [r for r in rows if r.get("correctness") is not None]
        refusals = [r for r in rows if is_refusal(r["answer"])]
        strict = [r for r in judged if r["correctness"] >= 4 and not is_refusal(r["answer"])]
        numeric = [numeric_match(r["answer"], r.get("reference")) for r in rows]
        numeric = [m for m in numeric if m is not None]
        report[config] = {
            "n": len(rows),
            "judge_accuracy": sum(1 for r in judged if r["correctness"] >= 4) / len(judged) if judged else None,
            "refusal_rate": len(refusals) / len(rows),
            "strict_accuracy": len(strict) / len(judged) if judged else None,
            "numeric_n": len(numeric),
            "numeric_match": sum(numeric) / len(numeric) if numeric else None,
            "refusals_judged_correct": sum(1 for r in refusals if (r.get("correctness") or 0) >= 4),
        }
    return report


def main():
    folder = sys.argv[1]
    report = analyze(folder)
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = ["| Config | Judge accuracy (lenient) | Refused | Strict accuracy | Numeric match (n) |",
             "|---|---|---|---|---|"]
    for c, s in report.items():
        lines.append(f"| {c} | {pct(s['judge_accuracy'])} | {pct(s['refusal_rate'])} | {pct(s['strict_accuracy'])} "
                     f"| {pct(s['numeric_match'])} ({s['numeric_n']}) |")
    text = "\n".join(lines)
    print(text)
    for c, s in report.items():
        print(f"{c}: {s['refusals_judged_correct']} refusals were scored >= 4 by the judge")
    with open(os.path.join(folder, "strict_report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
