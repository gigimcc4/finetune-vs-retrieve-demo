"""
Step 3b — The FINE-TUNE path (behavior lever, misused as a knowledge lever).

This reproduces the exact move behind the complaint we hear a lot:
"I fine-tuned the model on our docs and it still doesn't know our data."

We take a small open model and continue-pretrain it (LoRA) on the raw text
of the manual — the naive "train on our docs" approach. Then we ask the
golden spec questions CLOSED-BOOK (no retrieval) and score exact-match.

Expected result: near zero. The model comes out sounding more like a manual
(that is what fine-tuning changed — style/behavior) while confidently
inventing numbers, because gradient descent over prose does not install a
reliable lookup table for a torque spec.

RUNTIME: needs a GPU. Free Google Colab (T4) runs this on the default
0.5B model in a few minutes. On CPU it will be painfully slow — use --smoke
to sanity-check the wiring on a tiny slice, then move to Colab for the real run.

    python src/finetune.py --smoke          # wiring check, tiny
    python src/finetune.py                   # real small run (GPU)
    python src/finetune.py --base Qwen/Qwen2.5-0.5B
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"
GOLDEN = ROOT / "eval" / "golden.jsonl"
ADAPTER = ROOT / "artifacts" / "lora-manual"


def load(path):
    return [json.loads(l) for l in path.open()]


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def build_dataset(tokenizer, smoke):
    from datasets import Dataset

    chunks = [c["text"] for c in load(CORPUS)]
    if smoke:
        chunks = chunks[:64]
    ds = Dataset.from_dict({"text": chunks})

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=512)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tok, batched=True, remove_columns=["text"])


def train(base, smoke):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(tok, smoke)
    args = TrainingArguments(
        output_dir=str(ADAPTER),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1 if smoke else 3,   # more epochs = more "memorization" pressure
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    return model, tok


def evaluate_closed_book(model, tok):
    import torch

    golden = load(GOLDEN)
    correct = 0
    print("\nFINE-TUNE path  (closed-book, no retrieval)\n")
    for g in golden:
        prompt = g["question"] + "\nAnswer with the exact value:"
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=24, do_sample=False)
        pred = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = norm(g["answer"]) in norm(pred)
        correct += ok
        print(f"  {g['qid']}  gold={g['answer']:<12} model='{pred[:40]}'  {'OK' if ok else 'MISS'}")
    print(f"\n  closed-book exact: {correct}/{len(golden)}")
    print("  Takeaway: training on the docs taught STYLE, not recall. Knowledge != fine-tuning.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model, tok = train(args.base, args.smoke)
    evaluate_closed_book(model, tok)


if __name__ == "__main__":
    main()
