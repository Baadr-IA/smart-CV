"""
Adaptateur LLM unifié — supporte Anthropic, Gemini, OpenAI et Copilot.
Lit la config depuis les variables d'environnement (.env).
"""

import logging
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from dotenv import load_dotenv

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
}

def get_model(provider: str) -> str:
    """Retourne le modèle configuré pour un provider donné."""
    provider = provider.lower()
    env_var = f"{provider.upper()}_MODEL"
    return os.environ.get(env_var, _DEFAULT_MODELS.get(provider, "gpt-4o"))

def create_client():
    """Crée le client LLM selon LLM_PROVIDER dans .env."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), provider
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
    else:
        raise ValueError(f"Provider non supporté : {provider}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    before_sleep=lambda retry_state: logger.warning(f"Retry {retry_state.attempt_number} après erreur...")
)
def llm_call(client, provider: str, system_prompt: str, user_message: str,
             max_tokens: int = 4096, temperature: float = 0.0) -> str:
    model = get_model(provider)
    
    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
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
        return response.choices[0].message.content
