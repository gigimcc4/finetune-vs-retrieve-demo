"""
Step 4 — Side-by-side scoreboard.

Runs the RETRIEVE path always (offline, free). Runs the FINE-TUNE path only
with --finetune (needs a GPU). Prints the one table the whole demo is about.

    python src/compare.py                 # retrieval only
    python src/compare.py --finetune      # both (GPU)
"""
import argparse
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden.jsonl"


def n_golden():
    return sum(1 for _ in GOLDEN.open())


def run_retrieval():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "src" / "retrieval_demo.py")],
        capture_output=True, text=True,
    ).stdout
    m = re.search(r"answer_in_context@\d+:\s*(\d+)/(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true")
    args = ap.parse_args()

    total = n_golden()
    ret_correct, _ = run_retrieval()

    print("\n" + "=" * 52)
    print("  FINE-TUNE vs RETRIEVE  — exact-spec recall")
    print("=" * 52)
    print(f"  {'approach':<26}{'lever':<14}score")
    print("  " + "-" * 48)
    if args.finetune:
        import finetune as F
        model, tok = F.train("Qwen/Qwen2.5-0.5B", smoke=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            F.evaluate_closed_book(model, tok)
        m = re.search(r"closed-book exact:\s*(\d+)/(\d+)", buf.getvalue())
        ft = int(m.group(1)) if m else 0
        print(f"  {'fine-tune (closed-book)':<26}{'behavior':<14}{ft}/{total}")
    else:
        print(f"  {'fine-tune (closed-book)':<26}{'behavior':<14}run with --finetune (GPU)")
    print(f"  {'retrieve (search, no model)':<26}{'knowledge':<14}{ret_correct}/{total}")
    print("  " + "-" * 48)
    print("  Fine-tune for behavior. Retrieve for knowledge.\n")


if __name__ == "__main__":
    main()
