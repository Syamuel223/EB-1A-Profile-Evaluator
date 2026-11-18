"""Provider adapter for Gemini and OpenAI JSON generation and embeddings."""
from dataclasses import dataclass
from functools import lru_cache
import os
import time
from .config import EMBEDDING_PROVIDER, LLM_PROVIDER, provider_api_key, provider_model, provider_embedding_key, provider_embedding_model

_last_gemini_request = 0.0
_last_groq_request = 0.0


def _pace_gemini():
    """Keep sequential Gemini calls below common low-tier request rates."""
    global _last_gemini_request
    interval = max(0.0, float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "4")))
    elapsed = time.monotonic() - _last_gemini_request
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_gemini_request = time.monotonic()


def _pace_groq():
    """Space profile requests to fit low TPM on-demand tiers."""
    global _last_groq_request
    interval = max(0.0, float(os.getenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "60")))
    if _last_groq_request:
        elapsed = time.monotonic() - _last_groq_request
        if elapsed < interval:
            time.sleep(interval - elapsed)
    _last_groq_request = time.monotonic()


@dataclass
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    api_latency_seconds: float | None = None


def generate_json(system: str, user: str, model: str | None = None, max_output_tokens: int | None = None) -> Generation:
    key = provider_api_key()
    if not key:
        key_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}.get(LLM_PROVIDER, "provider API key")
        raise RuntimeError(f"Set {key_name} in .env to run model inference.")
    target = model or provider_model()
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=2, initial_delay=0.5, max_delay=2.0)))
        fallbacks = [x.strip() for x in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash-lite").split(",") if x.strip()]
        legacy_fallback = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
        if legacy_fallback:
            fallbacks.insert(0, legacy_fallback)
        models = list(dict.fromkeys([target] + [name for name in fallbacks if name != target]))
        last_error = None
        for candidate in models:
            try:
                _pace_gemini()
                request_started = time.perf_counter()
                response = client.models.generate_content(
                    model=candidate,
                    contents=user,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=0, response_mime_type="application/json"),
                )
                api_latency = time.perf_counter() - request_started
                usage = getattr(response, "usage_metadata", None)
                return Generation(response.text or "", getattr(usage, "prompt_token_count", 0) or 0, getattr(usage, "candidates_token_count", 0) or 0, candidate, api_latency)
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                # 429 is often an account quota/rate ceiling; cycling models can
                # amplify the burst. Retry only on transient server availability.
                if status not in (500, 502, 503, 504) or candidate == models[-1]:
                    raise
        raise last_error
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        request_started = time.perf_counter()
        response = OpenAI(api_key=key).chat.completions.create(
            model=target,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        api_latency = time.perf_counter() - request_started
        usage = response.usage
        return Generation(response.choices[0].message.content or "", usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0, target, api_latency)
    if LLM_PROVIDER == "groq":
        from groq import Groq
        _pace_groq()
        request_started = time.perf_counter()
        groq_options = {}
        if target.startswith("openai/gpt-oss"):
            groq_options["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
        response = Groq(api_key=key).chat.completions.create(
            model=target,
            temperature=0,
            max_completion_tokens=max_output_tokens or int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "4500")),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **groq_options,
        )
        api_latency = time.perf_counter() - request_started
        usage = response.usage
        return Generation(response.choices[0].message.content or "", usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0, target, api_latency)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def embed_documents(texts: list[str]) -> list[list[float]] | None:
    key = provider_embedding_key()
    if not key or not texts:
        return None
    model = provider_embedding_model()
    if EMBEDDING_PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        output = []
        # Gemini embedding-001 returns one vector per input and supports task types.
        for start in range(0, len(texts), 50):
            _pace_gemini()
            response = client.models.embed_content(
                model=model,
                contents=texts[start:start + 50],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            output.extend([list(item.values) for item in (response.embeddings or [])])
        if len(output) != len(texts):
            raise RuntimeError("Gemini returned an unexpected number of document embeddings.")
        return output
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=key)
        output = []
        for start in range(0, len(texts), 64):
            response = client.embeddings.create(model=model, input=texts[start:start + 64])
            output.extend(item.embedding for item in response.data)
        return output
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. Use gemini or openai; Groq embeddings are not configured.")


@lru_cache(maxsize=512)
def embed_query(text: str) -> tuple[float, ...] | None:
    if EMBEDDING_PROVIDER == "gemini":
        if not provider_embedding_key():
            return None
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=provider_embedding_key())
        _pace_gemini()
        response = client.models.embed_content(model=provider_embedding_model(), contents=text, config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        return tuple(response.embeddings[0].values)
    if EMBEDDING_PROVIDER not in ("openai", "gemini"):
        return None
    vectors = embed_documents([text])
    if not vectors:
        return None
    return vectors[0]
