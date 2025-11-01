from __future__ import annotations

import json
from pathlib import Path
from functools import wraps
from urllib.parse import urlparse
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, g

from config import get_settings
from services.llm_client import LLMClient, build_prompt, parse_sections
from services.i18n import t, get_supported_languages, resolve_lang_from_request_args, language_name

app = Flask(__name__)
settings = get_settings()
app.config['SECRET_KEY'] = settings.flask_secret_key

# Fields expected in the form/JSON
FORM_FIELDS = [
    'name', 'age', 'sex', 'chief_complaint', 'history', 'red_flags', 'vitals', 'goals',
    'biomechanical', 'respiratory', 'metabolic', 'neurological', 'behavioral'
]

BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / 'demo-data'

# --- Database setup (SQLite) ---
DB_PATH = (BASE_DIR / settings.db_path).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    db = getattr(g, '_db', None)
    if db is None:
        # Note: isolation_level=None enables autocommit by default; keep default and commit explicitly
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


def close_db(e=None):  # noqa: unused-argument
    db = getattr(g, '_db', None)
    if db is not None:
        db.close()
        g._db = None


@app.teardown_appcontext
def teardown_db(exception):  # noqa: unused-argument
    close_db()


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user TEXT,
            lang TEXT,
            provider TEXT,
            model TEXT,
            payload_json TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            response_text TEXT NOT NULL
        )
        """
    )
    db.commit()


# Flask 3 removed before_first_request; initialize DB on the first incoming request instead.
_db_init_done = False

@app.before_request
def _ensure_db():
    global _db_init_done
    if not _db_init_done:
        init_db()
        _db_init_done = True


def _auth_enabled() -> bool:
    return bool(settings.auth_username) and bool(settings.auth_password)


def _is_logged_in() -> bool:
    return bool(session.get('auth', False))


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _auth_enabled():
            return view_func(*args, **kwargs)
        if _is_logged_in():
            return view_func(*args, **kwargs)
        # Preserve lang and next
        lang = resolve_lang_from_request_args(request.args)
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for('login', next=next_url, lang=lang))
    return wrapper


def _is_safe_next(next_url: str | None) -> bool:
    if not next_url:
        return False
    # Disallow absolute URLs to prevent open redirects
    parsed = urlparse(next_url)
    return not parsed.netloc and (parsed.path or parsed.query)


@app.route('/login', methods=['GET', 'POST'])
def login():
    lang = resolve_lang_from_request_args(request.args if request.method == 'GET' else request.form)
    def tn(key):
        return t(key, lang)

    if not _auth_enabled():
        return redirect(url_for('index', lang=lang))

    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == (settings.auth_username or '') and password == (settings.auth_password or ''):
            session['auth'] = True
            session['user'] = username
            next_url = request.args.get('next') or request.form.get('next')
            if _is_safe_next(next_url):
                return redirect(next_url)
            return redirect(url_for('index', lang=lang))
        else:
            flash('Invalid username or password.', 'danger')
            # fall through to render template

    next_url = request.args.get('next') if _is_safe_next(request.args.get('next')) else url_for('index', lang=lang)
    return render_template('login.html', settings=settings, lang=lang, langs=get_supported_languages(), tn=tn, next_url=next_url)


@app.route('/logout', methods=['POST', 'GET'])
def logout():
    lang = resolve_lang_from_request_args(request.args if request.method == 'GET' else request.form)
    session.clear()
    return redirect(url_for('login', lang=lang))


def _filter_payload(data: dict) -> dict:
    payload: dict[str, str] = {}
    for key in FORM_FIELDS:
        val = data.get(key, '') if isinstance(data, dict) else ''
        if isinstance(val, (int, float)):
            val = str(val)
        elif not isinstance(val, str):
            val = ''
        payload[key] = val
    return payload


def _list_demo_files() -> list[dict]:
    demos = []
    if DEMO_DIR.exists():
        for p in sorted(DEMO_DIR.glob('*.json')):
            demos.append({'name': p.stem, 'filename': p.name})
    return demos


@app.route('/', methods=['GET'])
@login_required
def index():
    lang = resolve_lang_from_request_args(request.args)
    demos = _list_demo_files()
    # translator lambda bound to lang
    def tn(key):
        return t(key, lang)
    return render_template('index.html', settings=settings, payload={}, demos=demos, lang=lang, langs=get_supported_languages(), tn=tn)


@app.route('/load-json', methods=['GET', 'POST'])
@login_required
def load_json():
    try:
        lang = resolve_lang_from_request_args(request.args if request.method == 'GET' else request.form)
        demos = _list_demo_files()
        def tn(key):
            return t(key, lang)
        if request.method == 'GET':
            demo = request.args.get('demo')
            if demo:
                candidate = (DEMO_DIR / f"{Path(demo).stem}.json").resolve()
                # Ensure the path is inside demo dir
                if not str(candidate).startswith(str(DEMO_DIR.resolve())) or not candidate.exists():
                    flash(tn('err.demo_not_found'), 'warning')
                    return redirect(url_for('index', lang=lang))
                with candidate.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                payload = _filter_payload(data)
                return render_template('index.html', settings=settings, payload=payload, demos=demos, lang=lang, langs=get_supported_languages(), tn=tn)
            # If no demo specified, just redirect
            return redirect(url_for('index', lang=lang))

        # POST: file upload
        file = request.files.get('json_file')
        if not file or file.filename == '':
            flash(tn('err.json_choose'), 'warning')
            return redirect(url_for('index', lang=lang))
        try:
            raw = file.read().decode('utf-8')
            data = json.loads(raw)
        except Exception:
            flash(tn('err.json_invalid'), 'danger')
            return redirect(url_for('index', lang=lang))
        payload = _filter_payload(data)
        return render_template('index.html', settings=settings, payload=payload, demos=demos, lang=lang, langs=get_supported_languages(), tn=tn)
    except Exception as e:
        flash(f"{t('err.load_json', lang)} {e}", 'danger')
        return redirect(url_for('index', lang=resolve_lang_from_request_args(request.args)))


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    form = request.form
    lang = resolve_lang_from_request_args(form)
    payload = {
        'name': form.get('name', '').strip(),
        'age': form.get('age', '').strip(),
        'sex': form.get('sex', '').strip(),
        'chief_complaint': form.get('chief_complaint', '').strip(),
        'history': form.get('history', '').strip(),
        'red_flags': form.get('red_flags', '').strip(),
        'vitals': form.get('vitals', '').strip(),
        'goals': form.get('goals', '').strip(),
        'biomechanical': form.get('biomechanical', '').strip(),
        'respiratory': form.get('respiratory', '').strip(),
        'metabolic': form.get('metabolic', '').strip(),
        'neurological': form.get('neurological', '').strip(),
        'behavioral': form.get('behavioral', '').strip(),
    }

    # translator for templates
    def tn(key):
        return t(key, lang)

    # basic validation
    if not payload['chief_complaint']:
        flash(tn('err.chief_required'), 'warning')
        return redirect(url_for('index', lang=lang))

    prompt = build_prompt(payload, lang=lang)
    client = LLMClient()
    resp = client.generate(prompt)

    if not resp.ok:
        flash(f"LLM error: {resp.error}", 'danger')
        return redirect(url_for('index', lang=lang))

    # Persist request/response to DB
    try:
        provider = settings.llm_provider
        if provider == 'OLLAMA':
            model = settings.ollama_model or 'llama3.1'
        elif provider == 'OPENAI':
            model = settings.openai_model or 'gpt-4o-mini'
        elif provider == 'GEMINI':
            model = settings.gemini_model or 'gemini-1.5-flash'
        else:
            model = ''
        db = get_db()
        db.execute(
            "INSERT INTO history (created_at, user, lang, provider, model, payload_json, prompt_text, response_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(timespec='seconds'),
                session.get('user'),
                lang,
                provider,
                model,
                json.dumps(payload, ensure_ascii=False),
                prompt,
                resp.text,
            ),
        )
        db.commit()
    except Exception as e:
        # Non-fatal: just log via flash for now
        flash(f"Warning: failed to store history: {e}", 'warning')

    sections = parse_sections(resp.text)
    return render_template('result.html', payload=payload, sections=sections, raw=resp.text, settings=settings, lang=lang, langs=get_supported_languages(), tn=tn)


@app.route('/history', methods=['GET'])
@login_required
def history():
    lang = resolve_lang_from_request_args(request.args)
    def tn(key):
        return t(key, lang)
    db = get_db()
    rows = db.execute(
        "SELECT id, created_at, user, lang, provider, model FROM history ORDER BY id DESC LIMIT 100"
    ).fetchall()
    return render_template('history.html', rows=rows, settings=settings, lang=lang, langs=get_supported_languages(), tn=tn)


@app.route('/history/<int:item_id>', methods=['GET'])
@login_required
def history_detail(item_id: int):
    lang = resolve_lang_from_request_args(request.args)
    def tn(key):
        return t(key, lang)
    db = get_db()
    row = db.execute(
        "SELECT * FROM history WHERE id = ?",
        (item_id,)
    ).fetchone()
    if not row:
        flash('History item not found.', 'warning')
        return redirect(url_for('history', lang=lang))
    return render_template('history_detail.html', item=row, settings=settings, lang=lang, langs=get_supported_languages(), tn=tn)


if __name__ == '__main__':
    host = settings.host
    port = settings.port
    app.run(host=host, port=port, debug=settings.flask_debug)
