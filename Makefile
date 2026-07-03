# Convenience wrapper. Everything runs inside the project-local .venv.
PY := .venv/bin/python

.PHONY: setup corpus eval retrieval compare finetune clean

setup:            ## create venv + install core + verify
	./setup.sh

corpus:           ## PDF -> page-anchored chunks
	$(PY) src/build_corpus.py

eval:             ## generate + verify the golden spec questions
	$(PY) src/build_evalset.py

retrieval:        ## RETRIEVE path (offline, free)
	$(PY) src/retrieval_demo.py

compare:          ## the scoreboard
	$(PY) src/compare.py

finetune:         ## FINE-TUNE path (needs GPU + `./setup.sh --optional`)
	$(PY) src/finetune.py

clean:            ## remove venv + generated artifacts
	rm -rf .venv artifacts src/__pycache__ data/corpus.jsonl eval/golden.jsonl
