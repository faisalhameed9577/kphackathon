# Installation (Windows 10/11, VS Code, PowerShell)

## 1. Prerequisites
* Python 3.11 (3.12 / 3.13 also work): <https://www.python.org/downloads/> – tick **Add python.exe to PATH**.
* VS Code with the *Python* extension (optional but recommended).

## 2. Create the environment
Open a terminal in the project folder (`zetesis_ai`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 3. Configure keys

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in:

| Variable | Needed? | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Recommended | <https://aistudio.google.com/apikey> |
| `GEMINI_MODEL` | Optional | Leave empty for the default; set a current model name if you see "model not found". |
| `NCBI_API_KEY` | Optional | NCBI account settings → API Key Management (raises limit 3 → 10 req/s). |
| `NCBI_EMAIL` | Optional | Your email; NCBI asks API users to identify themselves. |

Never commit `.env`. It is already listed in `.gitignore`.

## 4. Initialise the database

```powershell
python -m database.db
```

(The app also does this automatically on first start.)

## 5. Run the app

```powershell
streamlit run app.py
```

The browser opens at <http://localhost:8501>.

## 6. Run the tests

```powershell
pytest
```

All tests use mocks and synthetic fixtures; no network or API key is needed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` not found | Activate the venv (step 2) or run `python -m streamlit run app.py`. |
| `width="stretch"` error | Upgrade Streamlit: `pip install -U "streamlit>=1.50"`. |
| ChromaDB install fails | Update pip, make sure you use 64-bit Python, retry `pip install chromadb`. |
| Gemini "model not found" | Set `GEMINI_MODEL` in `.env` to a current model. |
| Search says a source failed | Check internet/proxy; try again later. The other source still works. |
| Want data elsewhere | Set `ZETESIS_DATA_DIR` in `.env`. |
