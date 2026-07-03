# Fine-tune ≠ knowledge injection

> "I fine-tuned the model on our docs and it still doesn't know our data."

I hear this a lot. This repo settles it, then builds the thing people actually
wanted — in two parts, on a document you could swap for your own tonight.

**The one-line diagnostic:** if the problem is *"it doesn't know my data,"*
don't fine-tune — retrieve. Fine-tune changes **behavior** (format, tone,
how it follows instructions). Retrieval supplies **knowledge** (facts you can
look up and cite). Two different levers.

**The arc:**

- **Part I — Prove it.** Fine-tuning on the docs can't recall a spec; retrieval
  quotes it exactly. No generative model required to make the point.
- **Part II — Build on it.** Put a reader on top of retrieval and *chat with the
  book*: plain-English questions, exact cited answers, "not in the manual" when
  it isn't there.

The corpus for both is a real 385-page equipment maintenance manual — a U.S.
Army technical manual for a motorized road grader (public domain). It reads like
a private company's service binder: torque specs, clearances, voltages, wear
limits. Chosen precisely because the base model has *not* memorized it, so a
correct answer has to come from what we add — no contamination hiding the result.

---

# Part I — Prove it: fine-tune ≠ knowledge

We auto-build a **golden eval set of exact-spec questions** and verify each one
against its source page (see `PROCESS.md` for how, and for every wrong turn along
the way). Then we compare two ways to get the manual's facts to the point of use:

| Approach | Lever | Exact-spec recall |
|---|---|---|
| Fine-tune on the docs, ask closed-book | behavior | ~0 / 12 *(run on GPU — see notes)* |
| Retrieve the spec from the manual (search) | knowledge | **12 / 12** *(measured, offline, no model)* |

The fine-tuned model comes out *sounding* like a manual while confidently
inventing numbers. Retrieval quotes the exact line. Same base model.

### Wait — where's the model? (You don't need one to make this point.)

Part I uses **no generative model at all** — no LLM, no agent, no API key. The
retrieval is classic information retrieval (TF-IDF search over the manual), and
it scores 12/12 on surfacing the exact spec. Said bluntly: the "it doesn't know
our data" problem was a **search problem**, and search solved it. A bigger,
agentic, or fine-tuned generative model was never the missing piece.

The generative model earns its place in **Part II**, but only as a *reader* that
phrases and cites what search already found — behavior, not knowledge. It's why
classic ML/IR still earns its seat in the LLM era: the cheapest fix is often not
a model at all. (For agents: one that needs a fact should *call a retriever*, not
lean on the model's memory. The context window is RAM, not storage.)

### Run Part I

Everything installs into a **project-local `.venv`** — nothing touches your
system Python, so it runs the same on every laptop. One command:

```bash
./setup.sh          # creates .venv, installs pinned deps, verifies (12/12)
```

Then, in that shell (or after `source .venv/bin/activate` in a new one):

```bash
python src/build_corpus.py       # PDF -> page-anchored chunks
python src/build_evalset.py      # generate spec questions, verify each
python src/retrieval_demo.py     # RETRIEVE path — offline, free, 12/12
python src/compare.py            # the scoreboard
```

Prefer make? `make setup`, then `make retrieval`, `make compare`, etc.

The **fine-tune half needs a GPU** (free Colab T4 is enough). Install the
extras, then run it:

```bash
./setup.sh --optional            # adds torch + transformers + peft + ...
python src/finetune.py --smoke   # wiring check
python src/finetune.py           # real small LoRA run, then closed-book eval
python src/compare.py --finetune # both paths in one table
```

Other optional flags: `retrieval_demo.py --embed` (dense retrieval),
`--llm` on retrieval/eval scripts (real reader / natural-language questions,
needs `OPENAI_API_KEY`). Both require `./setup.sh --optional`.

---

# Part II — Build on it: chat with the book

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gigimcc4/finetune-vs-retrieve-demo/blob/main/notebooks/chat_with_the_book.ipynb)

`notebooks/chat_with_the_book.ipynb` puts the *reader* layer on top of Part I's
retrieval so you can ask the manual questions in plain English and get **exact,
cited** answers — or an honest "not in the manual" when the fact isn't there.

```
your question
  -> retrieve the most relevant pages   (Part I — search, no model)
  -> a reader LLM answers ONLY from those pages, and cites them
  -> a gate checks the answer is grounded; if not, it abstains
  -> reply + page citation + a source-based trust flag
```

Retrieval still does the *knowing*; the reader only phrases and cites the answer.
Two backends: **Claude** (your Anthropic key) or **local gemma via Ollama** (no
key, uses the Colab GPU). Terminal-style chat loop, page citation and trust flag
on every reply. This is the auditable shape of "chat with our docs" — the
opposite of fine-tuning a model and hoping it memorized the numbers.

---

## Honest notes

- The **retrieval numbers are measured** (12/12, plain TF-IDF). The **fine-tune
  numbers are not run in this repo's CI** — that needs a GPU. The script is
  complete and runs on Colab; the expected closed-book result is near zero, and
  you should confirm it yourself rather than take my word.
- Retrieval isn't magic either. On the raw question it scored 4/12; the fixes
  that got it to 12/12 (query cleaning, dropping ambiguous parts-list questions)
  are documented in `PROCESS.md`. Retrieval quality is *engineering*, not a
  free win — that's part of the point.
- The Part II notebook's Claude/gemma cells run for real on Colab; the retrieval
  and grounding-gate logic around them is what's been tested here.
- Swap `data/` for your own manual and re-run. If your questions are
  "what does our doc say about X," this is the whole pattern.

> Reproducibility note: pinned versions live in `requirements.txt` (core) and
> `requirements-optional.txt` (heavy/GPU). The core path was verified from a
> clean venv on Python 3.10 — see `PROCESS.md` for why a venv, not a global
> `pip install`, is the difference between "runs for me" and "runs for everyone."

## What's here

```
data/       the manual (PDF) + provenance/license  -> corpus.jsonl
eval/       golden.jsonl  (verified exact-spec questions)
src/        Part I — build_corpus, build_evalset, retrieval_demo, finetune, compare
notebooks/  Part II — chat_with_the_book.ipynb  (grounded, cited chat — Claude or gemma)
PROCESS.md  the decision log: dataset pivots, the eval bugs, the retrieval fixes
```
