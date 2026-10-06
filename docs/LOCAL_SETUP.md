# Running AP Autopilot on your own machine

Everything runs offline with no account and no key. Grok is optional: it adds LLM extraction, the investigator, the VP and auditor agents, vision for scans, and written emails. Without a key the same pipeline runs with deterministic rules.

## 1. Install the prerequisites

**Python 3.12 or newer** (3.13 recommended).

- macOS: install from [python.org](https://www.python.org/downloads/), or `brew install python@3.13`. Check with `python3 --version`.
- Windows: install from [python.org](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** on the first screen. Check with `py --version` in PowerShell.

**Git**, to clone the repository. macOS: run `git --version` and accept the prompt to install the developer tools if asked. Windows: install from [git-scm.com](https://git-scm.com/download/win).

**Tesseract OCR (optional).** Lets offline mode read the scanned PDF and the phone photo. Without it, those two invoices are reported as skipped and everything else works.

- macOS: `brew install tesseract`
- Windows: install the UB Mannheim build from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki), then add `C:\Program Files\Tesseract-OCR` to PATH.
- Check with `tesseract --version` (open a new terminal first).

You do not need Graphviz, Docker or a database server. SQLite ships with Python.

## 2. Get the code and create a virtual environment

macOS / Linux:

```bash
git clone https://github.com/SauravKg1/galatiq-invoice-processing.git
cd galatiq-invoice-processing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```powershell
git clone https://github.com/SauravKg1/galatiq-invoice-processing.git
cd galatiq-invoice-processing
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

Your prompt now starts with `(.venv)`. Activate it again in every new terminal before running anything:

- macOS / Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\Activate.ps1`

If you use Anaconda and the prompt also shows `(base)`, run `conda deactivate` first, or conda's Python may