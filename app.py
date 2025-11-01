from __future__ import annotations

import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash

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
def index():
    lang = resolve_lang_from_request_args(request.args)
    demos = _list_demo_files()
    # translator lambda bound to lang
    def tn(key):
        return t(key, lang)
    return render_template('index.html', settings=settings, payload={}, demos=demos, lang=lang, langs=get_supported_languages(), tn=tn)


@app.route('/load-json', methods=['GET', 'POST'])
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

    sections = parse_sections(resp.text)
    return render_template('result.html', payload=payload, sections=sections, raw=resp.text, settings=settings, lang=lang, langs=get_supported_languages(), tn=tn)


if __name__ == '__main__':
    host = settings.host
    port = settings.port
    app.run(host=host, port=port, debug=settings.flask_debug)
