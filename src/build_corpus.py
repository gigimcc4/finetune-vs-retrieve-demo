"""
Step 1 — Build the corpus.

Reads the equipment manual PDF and turns it into page-anchored chunks.
Each chunk keeps its source page so every later answer is traceable
back to a line in the manual. This is the whole point of the demo:
knowledge you can cite, not knowledge you hope the weights absorbed.

Output: data/corpus.jsonl  (one JSON object per chunk)
    {"id", "page", "text"}
"""
import json
import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pip install pypdf")

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "TM-5-3805-237-35_grader.pdf"
OUT = ROOT / "data" / "corpus.jsonl"

# ~700 char chunks keep a single spec + its context together without
# blowing up the retrieval index. Tune for your own manual.
CHUNK_CHARS = 700
OVERLAP = 120


def clean(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def chunk_page(text: str):
    text = clean(text)
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + CHUNK_CHARS])
        i += CHUNK_CHARS - OVERLAP
    return out


def main():
    reader = PdfReader(str(PDF))
    n = 0
    with OUT.open("w") as f:
        for pageno, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            for j, ch in enumerate(chunk_page(raw)):
                if len(ch) < 40:
                    continue
                rec = {"id": f"p{pageno}_c{j}", "page": pageno, "text": ch}
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"pages: {len(reader.pages)}  chunks: {n}  -> {OUT}")


if __name__ == "__main__":
    main()
