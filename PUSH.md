# Push this repo to GitHub

Run from your Mac Terminal (not from Claude — git needs your GitHub login).
The GitHub repo already exists and is empty, so a fresh init is cleanest.

```bash
cd ~/Documents/Claude/Projects/Linkedin/finetune-vs-retrieve-demo

# clear any partial git state
rm -rf .git

git init
git branch -M main
git config user.name "Jeanne McClure"
git config user.email "jmcclure@arsinnovate.com"
git add -A
git commit -m "Fine-tune != knowledge: Part I proof + Part II chat-with-the-book"
git remote add origin https://github.com/gigimcc4/finetune-vs-retrieve-demo.git
git push -u origin main
```

## Notes
- If push asks for a password, use a **GitHub personal access token**, not your
  account password (GitHub stopped accepting passwords over HTTPS).
  Create one: GitHub → Settings → Developer settings → Personal access tokens →
  Fine-grained token with `Contents: read/write` on this repo.
- `.gitignore` already excludes `.venv/` and generated `data/corpus.jsonl`.
- The 19 MB manual PDF **is** included (needed to run; well under GitHub's 100 MB).
- After pushing, the "Open in Colab" badge in the README goes live automatically.

## Later updates (after the first push)
```bash
git add -A
git commit -m "your message"
git push
```
