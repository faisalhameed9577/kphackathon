"""
Gemini API wrapper (Google GenAI SDK).

* Lazy client creation: the app works without a key (callers check ``available``).
* Bounded retries with exponential back-off for rate limits and temporary errors.
* Friendly error messages; API keys never appear in them.
"""
from __future__ import annotations

import math
import time
from typing import Any, List, Optional, Sequence

from config.settings import Settings
from utils.helpers import chunked, get_logger, parse_json_loose, redact_secrets

log = get_logger(__name__)

RETRYABLE_CODES = {429, 500, 502, 503, 504}
EMBED_BATCH_SIZE = 100  # Gemini accepts up to 100 texts per embed request


class GeminiError(Exception):
    """Any failure talking to Gemini. The message is safe to show to users."""


class GeminiUnavailableError(GeminiError):
    """Gemini is not configured (no API key or SDK missing)."""


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_network_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        word in name for word in ("timeout", "connect", "network", "remoteprotocol"))


class GeminiService:
    """Text generation, JSON generation and embeddings through Gemini."""

    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        self._client = client
        self._init_error: Optional[str] = None
        if self._client is None and settings.gemini_api_key:
            try:
                from google import genai  # imported lazily so the app starts without the SDK

                self._client = genai.Client(api_key=settings.gemini_api_key)
            except Exception as exc:  # noqa: BLE001 - SDK import/config problems
                self._init_error = redact_secrets(exc, settings.secrets)
                log.warning("Gemini client could not be created: %s", self._init_error)

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return self._client is not None

    def _require_client(self) -> Any:
        if self._client is None:
            reason = self._init_error or "GEMINI_API_KEY is not set in your .env file."
            raise GeminiUnavailableError(f"Gemini is not available: {reason}")
        return self._client

    # ------------------------------------------------------------- retry core
    def _call(self, func, what: str):
        """Run ``func`` with retries; convert any failure to GeminiError."""
        attempts = max(1, self.settings.max_retries)
        last_code: Optional[int] = None
        last_text = ""
        for attempt in range(attempts):
            try:
                return func()
            except GeminiError:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK raises many exception types
                last_code = _status_code(exc)
                last_text = redact_secrets(exc, self.settings.secrets)
                retryable = (last_code in RETRYABLE_CODES) or (last_code is None and _is_network_error(exc))
                log.warning("Gemini %s attempt %d failed (code=%s)", what, attempt + 1, last_code)
                if not retryable:
                    break
                if attempt < attempts - 1:
                    time.sleep(self.settings.retry_backoff_seconds * (2 ** attempt))
        raise GeminiError(self._friendly(last_code, last_text))

    def _friendly(self, code: Optional[int], text: str) -> str:
        if code in (401, 403):
            return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
        if code == 404:
            return f"Gemini model '{self.settings.gemini_model}' was not found. Check GEMINI_MODEL in your .env file."
        if code == 429:
            return "Gemini rate limit or quota reached. Wait a minute and try again, or use fewer papers."
        if code in RETRYABLE_CODES:
            return "Gemini is temporarily unavailable. Please try again shortly."
        if code is None:
            return "Could not reach Gemini (network problem). Check your internet connection."
        return f"Gemini returned an error ({code}). {text[:200]}"

    # -------------------------------------------------------------- generation
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None,
                      temperature: float = 0.2, json_mode: bool = False) -> str:
        client = self._require_client()

        def _run() -> str:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json" if json_mode else None,
            )
            response = client.models.generate_content(
                model=self.settings.gemini_model, contents=prompt, config=config)
            text = getattr(response, "text", None)
            if not text:
                raise GeminiError("Gemini returned an empty answer (it may have been blocked by a safety filter).")
            return text

        return self._call(_run, "generate")

    def generate_json(self, prompt: str, system_instruction: Optional[str] = None) -> Any:
        """Generate and parse JSON. One extra attempt if the first reply is not valid JSON."""
        for attempt in range(2):
            text = self.generate_text(prompt, system_instruction, temperature=0.1, json_mode=True)
            try:
                return parse_json_loose(text)
            except ValueError:
                log.warning("Gemini returned invalid JSON (attempt %d)", attempt + 1)
        raise GeminiError("Gemini did not return valid structured output. Please try again.")

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Embed texts (batched). Vectors are L2-normalised so cosine search is accurate."""
        client = self._require_client()
        dim = self.settings.embedding_dimension
        vectors: List[List[float]] = []

        for batch in chunked(list(texts), EMBED_BATCH_SIZE):
            def _run(batch=batch):
                from google.genai import types

                config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
                response = client.models.embed_content(
                    model=self.settings.gemini_embedding_model, contents=list(batch), config=config)
                embeddings = getattr(response, "embeddings", None) or []
                if len(embeddings) != len(batch):
                    raise GeminiError("Gemini returned an unexpected number of embeddings.")
                return [list(e.values) for e in embeddings]

            for vec in self._call(_run, "embed"):
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                vectors.append([v / norm for v in vec])
        return vectors
