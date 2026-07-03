"""
Step 2 — Build a golden eval set with NO pre-existing ground truth.

This is the honest way to test a doc you were handed with no Q&A pairs
(LinkedIn backlog gap #11). Two phases:

  GENERATE  — pull sentences that state an exact specification (a number
              with units) and turn each into a cloze question by blanking
              the value. Cloze keeps the question unambiguous: there is
              exactly one right string.

  VERIFY    — keep a question ONLY if its answer:
                (a) actually appears on its cited page, and
                (b) is *unique* across the whole manual (appears on one
                    page only), so retrieval to that page is decisive and
                    the question isn't secretly answerable from elsewhere.

The verify step is what makes a generated set trustworthy. Generation is
cheap; verification is the product.

An optional LLM rephrase (--llm) turns each cloze into a natural question.
It is off by default so the repo runs offline with zero spend.

Output: eval/golden.jsonl
    {"qid","question","answer","page","source_sentence","style"}
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"
OUT = ROOT / "eval" / "golden.jsonl"

# A "spec value": a decimal/fraction measurement with a real unit.
# (?<![\d.,]) stops us grabbing "140 rpm" out of "2,140 rpm".
# The unit must be followed by a word boundary so "58 Volt(age)" (a list
# label) can't masquerade as a 58-volt spec.
VALUE = re.compile(
    r"(?P<val>"
    r"(?<![\d.,])\d+\.\d+\s*(?:inch|in\.|in|mm|psi|volts?|amperes?|amp|rpm|lb-ft|lb\.-ft|ft-lb)\b"
    r"|(?<![\d.,])\d+\s*\d*/\d+\s*(?:inch|in\.|in)\b"
    r"|(?<![\d.,])\d{2,4}\s*(?:psi|rpm)\b"
    r")",
    re.I,
)

STOP_SENTENCE = re.compile(r"figure|table of contents|^\s*$", re.I)

# Parts-list / bill-of-material lines ("Screw, cap, hex-head, 3/8-16 X 1 5/8 in")
# repeat near-identically across dozens of pages. Their dimension is real but the
# surrounding context is boilerplate, so they make ambiguous questions. Drop them
# and keep true specification statements (clearances, pressures, voltages, limits).
BOM_LINE = re.compile(r"\b(screw|bolt|washer|nut|gasket),?\s*(cap|lock|flat|plain|hex)", re.I)


def load_corpus():
    rows = []
    with CORPUS.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def sentences(text):
    # crude splitter; manuals are terse so this is fine
    for s in re.split(r"(?<=[.;:])\s+|\n", text):
        s = s.strip()
        if 25 <= len(s) <= 240 and not STOP_SENTENCE.search(s):
            yield s


def norm(v):
    return re.sub(r"\s+", " ", v).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--llm", action="store_true", help="rephrase cloze -> natural question via API")
    args = ap.parse_args()

    rows = load_corpus()

    # ---- global frequency of each value string, for the uniqueness test
    global_counts = Counter()
    per_page_text = {}
    for r in rows:
        per_page_text.setdefault(r["page"], "")
        per_page_text[r["page"]] += " " + r["text"]
    for page, txt in per_page_text.items():
        seen = set()
        for m in VALUE.finditer(txt):
            v = norm(m.group("val"))
            if v not in seen:      # count each value once per page
                seen.add(v)
                global_counts[v] += 1

    # ---- GENERATE candidates
    candidates = []
    for r in rows:
        for s in sentences(r["text"]):
            if BOM_LINE.search(s):
                continue
            m = VALUE.search(s)
            if not m:
                continue
            val = m.group("val").strip()
            cloze = s.replace(val, "_____", 1)
            candidates.append(
                {
                    "page": r["page"],
                    "answer": val,
                    "answer_norm": norm(val),
                    "source_sentence": s,
                    "question": f"Per TM 5-3805-237-35, fill in the exact specification: \"{cloze}\"",
                    "style": "cloze",
                }
            )

    # ---- VERIFY: answer present on page AND globally unique
    verified, used_pages = [], set()
    for c in candidates:
        page_txt = norm(per_page_text.get(c["page"], ""))
        if c["answer_norm"] not in page_txt:
            continue                       # (a) not actually on cited page
        if global_counts.get(c["answer_norm"], 0) != 1:
            continue                       # (b) not unique -> ambiguous
        if c["page"] in used_pages:
            continue                       # spread across the manual
        verified.append(c)
        used_pages.add(c["page"])
        if len(verified) >= args.max:
            break

    if args.llm:
        verified = llm_rephrase(verified)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for i, c in enumerate(verified, 1):
            rec = {
                "qid": f"q{i:02d}",
                "question": c["question"],
                "answer": c["answer"],
                "page": c["page"],
                "source_sentence": c["source_sentence"],
                "style": c["style"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"candidates: {len(candidates)}  verified+unique: {len(verified)}  -> {OUT}")
    for c in verified:
        print(f"  p{c['page']:>3}  {c['answer']:<22}  {c['source_sentence'][:70]}")


def llm_rephrase(items):
    """Optional: turn cloze into a natural question. Requires OPENAI_API_KEY.
    Kept behind a flag so the default path is offline + free."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for c in items:
        prompt = (
            "Rewrite this fill-in-the-blank spec as a direct natural-language "
            "question whose answer is exactly the blanked value. Keep the component "
            f"name. Return only the question.\n\n{c['question']}"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        c["question"] = r.choices[0].message.content.strip()
        c["style"] = "natural"
    return items


if __name__ == "__main__":
    main()
