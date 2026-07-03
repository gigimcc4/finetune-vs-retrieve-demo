# Process log — the wrong turns, and why we changed course

The clean result in the README is not how the work actually went. The value is
in the detours. Here they are, in order.

## 1. Dataset: the obvious choice was the wrong choice

First instinct was a big, famous QA dataset — MS MARCO (~1M questions),
TriviaQA (~650K). Large and tempting.

**Why we abandoned it.** Those are built on web/Wikipedia text the base model
already saw in pretraining. If the model answers "correctly" after fine-tuning,
you cannot tell whether it *learned from your fine-tune* or *already knew it*.
That contamination silently invalidates the whole experiment — the exact "vibes"
trap this demo exists to refute.

**The rule that fixed it.** The corpus must contain facts the base model does
NOT already know. That means proprietary or obscure.

**Then a second pivot, from the user.** Proprietary finance data (FinanceBench)
worked technically, but the goal is a viewer thinking *"we could do this with
ours."* A 10-K only lands for finance people. A **legacy equipment maintenance
manual** lands for nearly everyone — it's the platonic "our internal ops docs."
We landed on a public-domain U.S. Army technical manual for a road grader:
uncontaminated, universally relatable, dense with checkable numbers.

## 2. Eval generation: the regex was too greedy

First pass at pulling "spec sentences" produced garbage like:

- `58 Volt` — actually lifted from *"58 Voltage regulator relay,"* a numbered
  list item, not a 58-volt spec.
- `140 rpm` — lifted from *"2,140 rpm,"* because the comma broke the digit run.

**Fix.** Tighten the value pattern: a negative lookbehind `(?<![\d.,])` so we
don't slice a number out of a bigger number, and require a word boundary after
the unit so `58 Volt(age)` can't masquerade as a measurement. Candidates dropped
from noisy-480 to clean-425, and the bad extractions disappeared.

## 3. Verification catches ambiguity, not mistakes

The verifier keeps a question only if its answer appears on the cited page **and**
is unique across the whole manual. That guarantees the question isn't secretly
answerable from another page — good.

**What it does NOT catch:** a value that was mis-parsed but still happens to be
unique. Uniqueness ≠ correctness. Lesson baked into the repo: automated
verification is necessary but not sufficient; a human still spot-checks the final
set. (This is LinkedIn backlog gap #11 in miniature — building evals with no
ground truth is real work, not a prompt.)

## 4. Retrieval: it did NOT work on the first try

Ran RAG expecting a clean win. Got **4/12**. Two separate problems:

**(a) The query carried dead weight.** Every question was wrapped in identical
boilerplate ("Per TM 5-3805-237-35, fill in the exact specification..."). That
boilerplate is constant across all questions, so it adds zero signal and dilutes
the distinctive terms in the TF-IDF vector. Fix: retrieve on the spec sentence
itself, not the instruction wrapper. → **8/12.**

**(b) Some questions were bad questions.** The remaining misses were all
parts-list lines — "Screw, cap, hex-head, 3/8-16 X 1 5/8 in" — which repeat
near-identically on dozens of pages. Genuinely ambiguous; no retriever should be
expected to pin the one right page. Fix: filter bill-of-material lines out of the
eval set at generation time, keeping true specification statements (clearances,
pressures, voltages, wear limits). → **12/12.**

The honest read: retrieval quality is engineering. The 4→8→12 climb is the part
most "just use RAG" takes leave out.

## 5. What we could NOT run here

The fine-tune half needs a GPU; this environment has none, so those numbers are
**not** reported as measured. The script is complete and runs on a free Colab T4.
Rather than fabricate a result, the repo states the expected outcome (near-zero
closed-book recall) and asks you to confirm it. That restraint is the whole brand.

## 6. "Runs for me" vs "runs for everyone" — use a venv

A pinned `requirements.txt` is necessary but not sufficient. If setup installs
those pins into the user's **system Python** (directly, or via
`pip install --break-system-packages`), you get collisions with whatever they
already have, and the interpreter that *installed* the packages can differ from
the one that later *runs* the script — the classic `ModuleNotFoundError` on a
fresh laptop.

The fix is a **project-local `.venv`**: an isolated interpreter that owns its
own pinned packages, so there is nothing to collide with and nothing to clean up
but a folder. `setup.sh` here creates it (with a `virtualenv` fallback for
minimal machines whose stdlib `venv`/`ensurepip` is missing) and then *proves*
the install by building the corpus + eval set and printing the scoreboard.

Verified: from a clean checkout, `virtualenv .venv` → install pinned core →
run pipeline → 12/12, with `pip list` showing the deps living inside the venv,
not the system.

## Scoreboard of the journey

| stage | change | result |
|---|---|---|
| dataset v1 | MS MARCO / TriviaQA | rejected: contamination |
| dataset v2 | FinanceBench | rejected: not relatable enough |
| dataset v3 | Army equipment manual | kept |
| eval v1 | loose regex | noisy extractions (58 Volt, 140 rpm) |
| eval v2 | tightened regex | clean values |
| eval v3 | drop parts-list lines | true specs only |
| retrieval v1 | raw question query | 4/12 |
| retrieval v2 | clean query | 8/12 |
| retrieval v3 | clean eval set | 12/12 |
