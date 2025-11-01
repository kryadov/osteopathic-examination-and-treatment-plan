from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore

from config import get_settings


@dataclass
class LLMResponse:
    ok: bool
    text: str
    error: Optional[str] = None


class LLMClient:
    def __init__(self):
        self.settings = get_settings()

    def generate(self, prompt: str) -> LLMResponse:
        provider = self.settings.llm_provider
        try:
            if provider == "OLLAMA":
                return self._ollama_generate(prompt)
            elif provider == "OPENAI":
                return self._openai_generate(prompt)
            elif provider == "GEMINI":
                return self._gemini_generate(prompt)
            else:
                return LLMResponse(False, "", error=f"Unsupported provider: {provider}")
        except Exception as e:
            return LLMResponse(False, "", error=str(e))

    # --- Providers ---
    def _ollama_generate(self, prompt: str) -> LLMResponse:
        base = self.settings.ollama_base_url or "http://localhost:11434"
        model = self.settings.ollama_model or "llama3.1"
        url = f"{base.rstrip('/')}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            return LLMResponse(False, "", error=f"Ollama HTTP {r.status_code}: {r.text}")
        data = r.json()
        text = data.get("response") or data.get("message") or ""
        return LLMResponse(True, text)

    def _openai_generate(self, prompt: str) -> LLMResponse:
        if OpenAI is None:
            return LLMResponse(False, "", error="openai package not available")
        api_key = self.settings.openai_api_key
        if not api_key:
            return LLMResponse(False, "", error="OPENAI_API_KEY is missing in environment")
        model = self.settings.openai_model or "gpt-4o-mini"
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise, clinically safe osteopathy assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(True, text)

    def _gemini_generate(self, prompt: str) -> LLMResponse:
        if genai is None:
            return LLMResponse(False, "", error="google-generativeai package not available")
        api_key = self.settings.gemini_api_key
        if not api_key:
            return LLMResponse(False, "", error="GEMINI_API_KEY is missing in environment")
        model_name = self.settings.gemini_model or "gemini-1.5-flash"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        # For safety, handle candidates
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = []
            for cand in resp.candidates:
                try:
                    parts.append(cand.content.parts[0].text)
                except Exception:
                    pass
            text = "\n".join(p for p in parts if p)
        return LLMResponse(True, text or "")


from services.i18n import language_name, normalize_lang

def build_prompt(payload: Dict[str, Any], lang: str | None = None) -> str:
    # Construct a structured instruction for the LLM
    name = payload.get("name", "Unknown")
    age = payload.get("age", "")
    sex = payload.get("sex", "")
    cc = payload.get("chief_complaint", "")
    history = payload.get("history", "")
    red_flags = payload.get("red_flags", "")
    vitals = payload.get("vitals", "")
    goals = payload.get("goals", "")

    models = {
        "Biomechanical/Structural": payload.get("biomechanical", ""),
        "Respiratory-Circulatory": payload.get("respiratory", ""),
        "Metabolic-Nutritional": payload.get("metabolic", ""),
        "Neurological": payload.get("neurological", ""),
        "Behavioral-Biopsychosocial": payload.get("behavioral", ""),
    }

    # Language instruction
    lcode = normalize_lang(lang or "en")
    lname = language_name(lcode)

    lines = []
    lines.append("You are an expert osteopathic clinician.")
    lines.append(f"Write the entire response in {lname}.")
    lines.append("Using the findings below, produce three sections: Summary, Diagnosis, Treatment Plan.")
    lines.append("Follow strictly the output template:")
    lines.append("Summary:\n- ...\n\nDiagnosis:\n- Working diagnosis\n- Differential diagnoses\n\nTreatment Plan:\n- Osteopathic techniques (with rationale)\n- Priorities and sequencing\n- Contraindications/precautions\n- Adjunct advice (exercise, lifestyle)\n- Follow-up plan and expected response")
    lines.append("Keep it concise, clinically safe, and do not exceed 400-600 words total.")
    lines.append("")

    lines.append("Patient:")
    lines.append(f"- Name: {name}")
    if age:
        lines.append(f"- Age: {age}")
    if sex:
        lines.append(f"- Sex: {sex}")
    if cc:
        lines.append(f"- Chief complaint: {cc}")
    if history:
        lines.append(f"- Relevant history: {history}")
    if red_flags:
        lines.append(f"- Red flags screened: {red_flags}")
    if vitals:
        lines.append(f"- Vitals/exam: {vitals}")
    if goals:
        lines.append(f"- Patient goals: {goals}")

    lines.append("\nWHO Five Models findings:")
    for k, v in models.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def parse_sections(text: str) -> Dict[str, str]:
    # naive splitter by headings
    sections = {"summary": "", "diagnosis": "", "plan": ""}
    if not text:
        return sections
    lower = text.lower()
    def find(h: str) -> int:
        return lower.find(h)
    s = find("summary")
    d = find("diagnosis")
    p = find("treatment plan")

    indices = [(s, "summary"), (d, "diagnosis"), (p, "plan")]
    indices = [(i, k) for i, k in indices if i != -1]
    if not indices:
        sections["summary"] = text.strip()
        return sections
    indices.sort()
    # slice between headings
    for idx, (start, key) in enumerate(indices):
        end = indices[idx + 1][0] if idx + 1 < len(indices) else len(text)
        chunk = text[start:end]
        # remove heading line
        lines = chunk.splitlines()
        if lines:
            lines = lines[1:]
        sections["summary" if key == "summary" else ("diagnosis" if key == "diagnosis" else "plan")] = "\n".join(lines).strip()
    return sections
