from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", str(ROOT / "data/knowledge_base")))
if not KNOWLEDGE_DIR.is_absolute():
    KNOWLEDGE_DIR = ROOT / KNOWLEDGE_DIR
SOURCE_URL = os.getenv("USCIS_SOURCE_URL", "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "openai").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini" if LLM_PROVIDER == "groq" else LLM_PROVIDER).lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
TOP_K = int(os.getenv("TOP_K", "5"))


def provider_api_key() -> str:
    if LLM_PROVIDER == "gemini":
        return GEMINI_API_KEY
    if LLM_PROVIDER == "openai":
        return OPENAI_API_KEY
    if LLM_PROVIDER == "groq":
        return GROQ_API_KEY
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use gemini, openai, or groq.")


def provider_model() -> str:
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    if LLM_PROVIDER == "groq":
        return GROQ_MODEL
    return OPENAI_MODEL


def provider_embedding_key() -> str:
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_API_KEY
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_API_KEY
    return ""


def provider_embedding_model() -> str:
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_EMBEDDING_MODEL
    if EMBEDDING_PROVIDER == "openai":
        return EMBEDDING_MODEL
    return ""
