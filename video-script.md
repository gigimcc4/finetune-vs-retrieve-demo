# Video script — "Fine-tune ≠ knowledge"

**Length:** 75–90 sec. **One claim, one live demo, one question.**
**Format:** talking head to open + close, screen-share for the middle.
**Voice:** first person, no hype, no AI attribution.

---

### 0:00–0:12 — Hook (talking head)
> "'I fine-tuned the model on our docs and it still doesn't know our data.'
> I hear this a lot — usually after a painful weekend and a real GPU bill.
> Here's the part that saves us the weekend."

**Shot:** you, direct to camera. Lower-third: *Fine-tune ≠ knowledge.*

### 0:12–0:28 — The frame (screen-share: the manual)
> "This is a 385-page equipment maintenance manual. Think of it as your ops
> binder — torque specs, clearances, wear limits. I want the model to know it."

**Shot:** scroll the PDF fast; pause on a page with a clear number
(e.g. valve clearance 0.009 inch). Circle the number.

### 0:28–0:50 — The wrong lever (screen-share: terminal)
> "Naive move: fine-tune on the docs. So I did — LoRA, small model, the whole
> thing. Now I ask it, closed-book: what's the spec on this part?"

**Shot:** run `finetune.py` output (pre-recorded). Show the model answering
with a confident WRONG number.
> "Confident. Wrong. Training on prose taught it to *sound* like the manual.
> It did not install a lookup table for a torque spec. That's the trap:
> fine-tuning changed behavior, not knowledge."

### 0:50–1:10 — The right lever (screen-share: terminal)
> "Leave the model alone. Put the manual in a retriever. Same question —
> but first pull the actual page, then answer."

**Shot:** run `retrieval_demo.py`; show **12/12**, answer quoted exactly.
> "Exact spec, straight off the page. Same base model. The difference wasn't
> a better model — it was giving it the page at question time."

### 1:10–1:25 — The rule + CTA (talking head)
> "So the diagnostic I give every team: if the problem is 'it doesn't know my
> data,' don't fine-tune — retrieve. Fine-tune for behavior, retrieve for
> knowledge. Code's in the comments — swap in your own manual and run it tonight."

**End card:** repo name + *"Fine-tune for behavior. Retrieve for knowledge."*

---

## Shot list / capture checklist
- [ ] Talking-head open + close (same framing, so it cuts clean)
- [ ] PDF scroll, pause + circle on one numeric spec
- [ ] Pre-record `finetune.py` closed-book miss (needs the Colab run)
- [ ] Pre-record `retrieval_demo.py` 12/12 hit
- [ ] Terminal font bumped to ~18pt for mobile legibility
- [ ] Captions burned in (most LinkedIn video is watched muted)
- [ ] First 3 seconds must carry the hook text on-screen (autoplay-muted)

## Post pairing
Drop the existing post (`2026-06-30-post-finetuning-not-knowledge.md`) as the
caption. Pin a comment: "Repo + the messy process log (what broke, what I
changed): <link>."
