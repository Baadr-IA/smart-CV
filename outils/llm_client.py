"""
Adaptateur LLM unifié — supporte Anthropic, Gemini, OpenAI, Copilot
et les endpoints OpenAI-compatible locaux.
Lit la config depuis les variables d'environnement (.env).
"""

import logging
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from dotenv import load_dotenv
from outils.metrics import observe_llm_call

# Only retry on genuine transient network/rate-limit errors — NOT on ValueError,
# KeyError, or other programmer errors that would just waste API credits.
def _is_transient_error(exc: BaseException) -> bool:
    """Return True only for errors that are worth retrying (network/rate-limit)."""
    try:
        import openai
        _OPENAI_TRANSIENT = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        if isinstance(exc, _OPENAI_TRANSIENT):
            return True
    except ImportError:
        pass
    try:
        import anthropic
        _ANTHROPIC_TRANSIENT = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )
        if isinstance(exc, _ANTHROPIC_TRANSIENT):
            return True
    except ImportError:
        pass
    return False

# Charger les variables d'environnement au chargement du module
load_dotenv()

logger = logging.getLogger(__name__)

# Modèles par défaut
_DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini":    "gemini-2.0-flash",
    "openai":    "gpt-4o",
    "copilot":   "gpt-4o",
    "local_openai": "qwen-cv",
}

_DEFAULT_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
}

def get_model(provider: str) -> str:
    """Retourne le modèle configuré pour un provider donné."""
    provider = provider.lower()
    if provider in {"local", "openai_compatible", "openai-compatible"}:
        provider = "local_openai"
    if provider == "local_openai":
        return os.environ.get("LOCAL_LLM_MODEL", _DEFAULT_MODELS["local_openai"])
    env_var = f"{provider.upper()}_MODEL"
    return os.environ.get(env_var, _DEFAULT_MODELS.get(provider, "gpt-4o"))

def create_client(provider_override: str | None = None):
    """Crée le client LLM selon LLM_PROVIDER dans .env."""
    provider = (provider_override or os.environ.get("LLM_PROVIDER", "openai")).lower()
    
    if provider == "openai":
        from openai import OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs), provider
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")), provider
    elif provider == "gemini":
        from openai import OpenAI # Gemini via interface OpenAI
        client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client, provider
    elif provider == "copilot":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com"
        )
        return client, provider
    elif provider in {"local_openai", "local", "openai_compatible", "openai-compatible"}:
        from openai import OpenAI
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("LOCAL_LLM_BASE_URL est requis pour LLM_PROVIDER=local_openai.")
        client = OpenAI(
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "dummy"),
            base_url=base_url,
        )
        return client, "local_openai"
    else:
        raise ValueError(f"Provider non supporté : {provider}")


def _get_attr(obj, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _extract_usage(provider: str, response) -> tuple[int, int, int]:
    usage = _get_attr(response, "usage")
    if usage is None:
        return 0, 0, 0
    if provider == "anthropic":
        prompt_tokens = int(_get_attr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(_get_attr(usage, "output_tokens", 0) or 0)
        return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    prompt_tokens = int(_get_attr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get_attr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(_get_attr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _pricing_env_key(model: str, token_type: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in model.upper()).strip("_")
    return f"LLM_COST_{normalized}_{token_type}_PER_1K"


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prompt_rate = os.getenv(_pricing_env_key(model, "PROMPT"))
    completion_rate = os.getenv(_pricing_env_key(model, "COMPLETION"))
    if prompt_rate is not None and completion_rate is not None:
        return (prompt_tokens / 1000.0) * float(prompt_rate) + (completion_tokens / 1000.0) * float(completion_rate)

    rates = _DEFAULT_PRICING_PER_1K.get(model)
    if rates is None:
        return None
    return (prompt_tokens / 1000.0) * rates[0] + (completion_tokens / 1000.0) * rates[1]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    before_sleep=lambda retry_state: logger.warning(f"Retry {retry_state.attempt_number} après erreur...")
)
def llm_call(client, provider: str, system_prompt: str, user_message: str,
             max_tokens: int = 4096, temperature: float = 0.0, operation: str = "generic") -> str:
    model = get_model(provider)
    start = time.perf_counter()

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            content = response.content[0].text
        else:  # openai, gemini, copilot
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            content = response.choices[0].message.content

        prompt_tokens, completion_tokens, total_tokens = _extract_usage(provider, response)
        observe_llm_call(
            provider=provider,
            model=model,
            operation=operation,
            status="success",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=_estimate_cost_usd(model, prompt_tokens, completion_tokens),
        )
        return content
    except Exception as exc:
        observe_llm_call(
            provider=provider,
            model=model,
            operation=operation,
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error_type=exc.__class__.__name__,
        )
        raise
