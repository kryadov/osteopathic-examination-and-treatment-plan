import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"OLLAMA", "OPENAI", "GEMINI"}


@dataclass
class Settings:
    llm_provider: str
    ollama_base_url: str | None
    ollama_model: str | None
    openai_api_key: str | None
    openai_model: str | None
    gemini_api_key: str | None
    gemini_model: str | None
    flask_debug: bool
    flask_secret_key: str
    host: str
    port: int


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "OLLAMA").upper().strip()
    if provider not in SUPPORTED_PROVIDERS:
        provider = "OLLAMA"

    return Settings(
        llm_provider=provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        flask_debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        flask_secret_key=os.getenv("FLASK_SECRET_KEY", "change-this-secret"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
    )
