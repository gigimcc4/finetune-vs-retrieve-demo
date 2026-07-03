#!/usr/bin/env bash
# One-shot setup — creates a PROJECT-LOCAL venv so this runs the same on every
# machine. Nothing is installed into your system Python, so there is nothing to
# collide with and nothing to clean up but a folder.
#
#   ./setup.sh              # core (offline, free path)
#   ./setup.sh --optional   # also install dense-retrieval / fine-tune / LLM extras
#
# Re-running is safe.
set -euo pipefail

# --- pick a Python 3.10+ interpreter --------------------------------------
PYTHON_CMD=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$py" >/dev/null 2>&1 && \
     "$py" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
    PYTHON_CMD="$py"; break
  fi
done
[ -z "$PYTHON_CMD" ] && { echo "Need Python 3.10+ (brew install python@3.11)"; exit 1; }
echo "==> Using $($PYTHON_CMD --version)"

# --- create the venv (idempotent) -----------------------------------------
# Prefer the stdlib venv. On stripped-down systems ensurepip can be missing,
# so fall back to `virtualenv` (which bundles pip and never needs ensurepip).
if [ ! -d ".venv" ]; then
  if "$PYTHON_CMD" -m venv .venv 2>/dev/null; then
    echo "==> Created .venv (venv)"
  else
    echo "==> venv/ensurepip unavailable, falling back to virtualenv"
    "$PYTHON_CMD" -m pip install --user virtualenv >/dev/null 2>&1 || \
      "$PYTHON_CMD" -m pip install --user --break-system-packages virtualenv >/dev/null 2>&1
    "$PYTHON_CMD" -m virtualenv .venv
    echo "==> Created .venv (virtualenv)"
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- install INTO the venv (no --break-system-packages needed, ever) ------
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

if [ "${1:-}" = "--optional" ]; then
  echo "==> Installing optional extras (torch first, then the rest)"
  python -m pip install torch
  python -m pip install -r requirements-optional.txt
fi

# --- prove it works --------------------------------------------------------
echo "==> Building corpus + eval set to verify the install"
python src/build_corpus.py
python src/build_evalset.py >/dev/null
python src/compare.py

cat <<EOF

Setup complete. The venv is active in THIS shell.
In a new terminal, re-activate with:   source .venv/bin/activate
Then run any step, e.g.:                python src/retrieval_demo.py
EOF
