# Osteopathic Examination Web App

An adaptive (desktop, mobile, tablet) Flask + Bootstrap web application for conducting a comprehensive osteopathic examination using the five WHO models. The app uses an LLM (Ollama, OpenAI, or Gemini) to generate a clinical summary, differential/working diagnosis, and an osteopathic treatment plan. It now includes optional user authentication, saved history, and basic admin features.

![input](input.png)
![output](output.png)

## Features
- Responsive Bootstrap UI
- Internationalization (i18n) with per-page language selector (English, German, French, Greek, Russian, Spanish, Chinese, Japanese)
- Optional user accounts: self‑registration, login/logout with simple math CAPTCHA
- Admin user management: view, block/unblock, delete users
- Saved analysis history (SQLite) with list and detail views
- Captures patient demographics and findings per five WHO models
- Upload a ready JSON file to prefill the form (or load built-in demo cases)
- Selectable LLM backend via environment (.env): OLLAMA, OPENAI, GEMINI
- Generates: 
  - Concise Summary
  - Diagnosis (working + differentials)
  - Osteopathic treatment plan (prioritized, contraindications, follow-up)

## Quickstart

1. Ensure Python 3.13 is installed.
2. Create and activate a virtual environment.
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy configuration and set provider/keys:
   ```bash
   cp .env.example .env
   # Edit .env to set LLM_PROVIDER and related keys
   ```
5. Run the app:
   ```bash
   python app.py
   ```
6. Open http://localhost:8000 in your browser.
   - If no users exist yet, you will be redirected to Register to create the first account (it becomes admin).
   - If a user already exists (or you bootstrapped one via .env), you will need to log in.

### Prefill via JSON upload
- Prepare a JSON file with these keys (all optional except chief_complaint for analysis):
  - name, age, sex, chief_complaint, history, red_flags, vitals, goals,
    biomechanical, respiratory, metabolic, neurological, behavioral
- On the top-right of the form, use "Load JSON into form" to upload your file and prefill fields.
- Or use the "Load demo" menu to load one of the built-in examples from `demo-data/`.

Examples available (plus localized variants):
- demo-data/low-back-pain.json (+ low-back-pain-de/es/fr/el/ru/zh/ja.json)
- demo-data/asthma-rib-dysfunction.json
- demo-data/chronic-fatigue-metabolic.json

## Authentication and roles (optional)
- Authentication is disabled until at least one user exists in the database.
- The first registered user becomes an admin automatically.
- You can bootstrap the first admin by setting AUTH_USERNAME and AUTH_PASSWORD in .env before first run (only if the users table is empty).
- Admins can manage users at /admin/users (block/unblock/delete).
- Login and registration forms use a simple math CAPTCHA.

## Data storage & history
- SQLite database file path is configured via DB_PATH (default: data/app.db). The file is created on first run.
- Each analysis is saved to the History. Use the “History” link in the navbar to review past runs and open their details.

## Environment variables (.env)
See `.env.example`. Key items:
- LLM_PROVIDER: OLLAMA | OPENAI | GEMINI
- OLLAMA_BASE_URL, OLLAMA_MODEL
- OPENAI_API_KEY, OPENAI_MODEL
- GEMINI_API_KEY, GEMINI_MODEL
- FLASK_DEBUG, FLASK_SECRET_KEY, HOST, PORT
- AUTH_USERNAME, AUTH_PASSWORD (optional; auto-create first admin if users table empty)
- DB_PATH (SQLite database path)

## Notes
- For Ollama, install and run Ollama locally and pull the specified model (e.g., `ollama pull llama3.1`).
- For OpenAI or Gemini, set API keys appropriately.
- This app is for educational/assistive purposes; not a substitute for clinical judgment.
