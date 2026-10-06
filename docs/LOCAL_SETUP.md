# Running AP Autopilot on your own machine

Everything runs offline with no account and no key. Grok is optional: it adds LLM extraction, the investigator, the VP and auditor agents, vision for scans, and written emails. Without a key the same pipeline runs with deterministic rules.

## 1. Install the prerequisites

**Python 3.12 or newer** (3.13 recommended).

- macOS: install from [python.org](https://www.python.org/downloads/), or `brew install python@3.13`. Check with `python3 --version`.
- Windows: install from [python.org](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** on the first screen. Check with `py --version` in PowerShell.

**Tesseract OCR (optional).** Lets offline mode read the scanned PDF and the phone photo. Without it, those two invoices are reported as skipped and everything else works.

- macOS: `brew install tesseract`
- Windows: install the UB Mannheim build from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki), then add `C:\Program Files\Tesseract-OCR` to PATH.
- Check with `tesseract --version` (open a new terminal first).

You do not need Graphviz, Docker or a database server. SQLite ships with Python.

## 2. Unzip and create a virtual environment

Unzip `acme-ap-autopilot.zip`, then open a terminal in that folder.

macOS / Linux:

```bash
cd ~/Downloads/acme-ap-autopilot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```powershell
cd $HOME\Downloads\acme-ap-autopilot
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

Your prompt now starts with `(.venv)`. Activate it again in every new terminal before running anything.

## 3. Check the build (offline)

```bash
pytest                      # about 200 tests, under a minute; 2 skip if Tesseract is missing
python main.py --eval       # 38 labelled invoices: expect 100% accuracy, 0 wrongful payments
```

On Windows use `python` everywhere this guide says `python` (inside the venv both work).

## 4. Run it

```bash
python main.py --invoice_path=data/invoices/invoice_1001.txt   # the brief's command
python main.py --batch data/invoices                            # the whole inbox, in arrival order
python main.py --review                                         # approve or reject held invoices
python main.py --outbox                                         # read and send drafted messages
python main.py --roi                                            # business case, three scenarios
python main.py --bench                                          # throughput before/after (simulated)
streamlit run app.py                                            # the console, opens http://localhost:8501
```

`inventory.db` is created on first run. `python main.py --reset-db` (or **Reset demo data** in the console) starts again from the seed data.

## 5. Add Grok (optional)

You do **not** need a Grok app subscription (SuperGrok or X Premium). Those plans are for the chat app. The API is a separate developer account at console.x.ai, billed per token from prepaid credits.

1. Go to [console.x.ai](https://console.x.ai) and sign up with an email, Google or X account.
2. Check **Billing / Credits**. If there is no promotional balance, add a small prepaid amount; $10 covers everything below with room to spare.
3. Open **API Keys**, create a key, and copy it (it is shown once).
4. Open **Models** and note the exact ID of a current reasoning model that accepts images, and optionally a cheaper fast model.
5. Edit `.env`:

   ```
   XAI_API_KEY=xai-...your key...
   XAI_MODEL=<model ID from the Models page>
   # optional routing:
   # XAI_MODEL_FAST=<cheaper fast model ID>
   # XAI_MODEL_REASONING=<reasoning model ID>
   ```

   `.env` is in `.gitignore`. Never commit it or paste the key anywhere public.

6. Verify, then run:

   ```bash
   python main.py --check-llm --vision     # one text call and one image call; prints the model and token use
   python main.py --eval --llm grok        # the labelled set on Grok
   python main.py --compare --runs 3       # writes eval/SCORECARD.md (rules vs Grok, 3 runs)
   python main.py --bench --live           # writes eval/BENCH.md with real latencies and tokens
   streamlit run app.py                    # pick "grok" in the sidebar
   ```

   With a key in `.env`, `LLM_PROVIDER=auto` uses Grok by default; `--llm offline` forces the rules-only mode.

**What it costs:** one pass over the 38 invoices is a few hundred LLM calls and well under a dollar at typical list prices. `--compare --runs 3` and `--bench --live` together are a few dollars. Check current prices on the Models page.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` (macOS) | Use `python3`, or activate the venv first. |
| `No module named ...` | The venv is not active: `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`). |
| `--check-llm` says model not found | Copy the exact model ID from console.x.ai Models into `XAI_MODEL`. |
| `--check-llm --vision` fails on the image call | That model is text-only. Set `XAI_VISION_MODEL` to a model that accepts images. |
| 401 / 403 from the API | Key wrong or out of credits: re-copy the key, check Billing. |
| 429 rate limit | The client retries with backoff; rerun if it still fails. |
| Scanned invoices show as skipped offline | Install Tesseract and open a new terminal, or use Grok vision. |
| Streamlit port 8501 busy | `streamlit run app.py --server.port 8502` |
| Old results in the console | **Reset demo data** in the sidebar, or `python main.py --reset-db`. |
