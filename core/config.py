from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Groq
    groq_api_key: str = "not-set"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_models: list[str] = ["llama-3.1-8b-instant", "gemma2-9b-it"]

    # Embeddings — local sentence-transformers, no API key needed
    embedding_model: str = "all-MiniLM-L6-v2"

    # Twilio (optional — for WhatsApp webhook signature validation)
    twilio_account_sid:     str = ""
    twilio_auth_token:      str = ""
    twilio_whatsapp_number: str = ""

    # Google Sheets CRM (optional — falls back to mock when not set)
    google_service_account_json: str = ""
    google_sheet_id:             str = ""

    # Paths
    chroma_persist_dir: str = "./data/chroma_db"
    sqlite_db_path: str = "./data/checkpoints.db"
    kb_path: str = "./data/kb/autostream.md"

    # Conversation settings
    max_turns: int = 6
    summarize_after_turns: int = 4
    retrieval_k: int = 3


@lru_cache()
def get_settings() -> Settings:
    return Settings()
