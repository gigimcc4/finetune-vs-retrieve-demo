"""
Step 3a — The RETRIEVE path (knowledge lever).

Index the manual's chunks, and for each golden question pull the chunks
most likely to contain the answer. Two things get measured:

  page_hit  — did the correct source page surface in the top-k? (retrieval)
  answer_in_context — is the exact spec string sitting in what we retrieved?

If answer_in_context is true, a competent reader model will quote it. That
is the point: the fact was *available at question time*. We did not ask the
model to have memorized it.

Default retriever is TF-IDF (scikit-learn) so this runs offline with zero
downloads. Pass --embed to use sentence-transformers instead (nicer recall,
needs a model download). Pass --llm to actually let a model read the context
and answer, scoring exact-match end to end.

    python src/retrieval_demo.py            # offline, free
    python src/retrieval_demo.py --embed    # dense retrieval
    python src/retrieval_demo.py --llm      # full read+answer (needs API key)
"""
import argparse
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"
GOLDEN = ROOT / "eval" / "golden.jsonl"
K = 4


def load(path):
    return [json.loads(l) for l in path.open()]


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def query_text(g):
    """What we actually send to the retriever.

    The stored question wraps the spec in instruction boilerplate
    ("Per TM..., fill in..."). That boilerplate is identical across every
    question, so it adds no signal and dilutes the distinctive terms. We
    retrieve on the spec context itself (the blanked sentence), which is
    the realistic query a user's app would build.
    """
    m = re.search(r'"(.*?)"', g["question"], re.S)
    body = m.group(1) if m else g["question"]
    return body.replace("_____", " ").strip()


def build_tfidf(texts):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    return lambda q: cosine_similarity(vec.transform([q]), mat)[0]


def build_embed(texts):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return lambda q: util.cos_sim(
        model.encode(q, convert_to_tensor=True, normalize_embeddings=True), emb
    )[0].cpu().numpy()


def llm_answer(question, context):
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY with the exact spec value from the context. "
                "If it is not present, say 'NOT FOUND'.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return r.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    corpus = load(CORPUS)
    golden = load(GOLDEN)
    texts = [c["text"] for c in corpus]
    score = build_embed(texts) if args.embed else build_tfidf(texts)

    page_hits = ans_in_ctx = llm_correct = 0
    print(f"RETRIEVE path  (retriever={'embed' if args.embed else 'tfidf'}, k={K})\n")
    for g in golden:
        sims = score(query_text(g))
        top = sorted(range(len(corpus)), key=lambda i: sims[i], reverse=True)[:K]
        pages = {corpus[i]["page"] for i in top}
        context = "\n---\n".join(corpus[i]["text"] for i in top)
        page_hit = g["page"] in pages
        in_ctx = norm(g["answer"]) in norm(context)
        page_hits += page_hit
        ans_in_ctx += in_ctx
        line = f"  {g['qid']}  page_hit={int(page_hit)}  answer_in_context={int(in_ctx)}  (gold p{g['page']} = {g['answer']})"
        if args.llm:
            pred = llm_answer(g["question"], context)
            ok = norm(g["answer"]) in norm(pred)
            llm_correct += ok
            line += f"  | model='{pred}' {'OK' if ok else 'MISS'}"
        print(line)

    n = len(golden)
    print(f"\n  page_hit@{K}:          {page_hits}/{n}")
    print(f"  answer_in_context@{K}: {ans_in_ctx}/{n}")
    if args.llm:
        print(f"  end-to-end exact:     {llm_correct}/{n}")
    print("\n  Takeaway: the facts are retrievable and quotable. Knowledge = retrieval.")


if __name__ == "__main__":
    main()
