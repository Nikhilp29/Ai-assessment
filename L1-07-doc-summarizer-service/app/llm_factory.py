"""
Pluggable LLM client. One `call_llm(prompt)` function; the actual provider
(Groq / OpenAI / Gemini) is chosen once at startup from LLM_PROVIDER in the
environment. This keeps summarizer.py provider-agnostic, and it's the same
seam L3-12 (Model Evaluation Dashboard) will hook into to benchmark
providers against each other.
"""
from app.config import get_settings

settings = get_settings()


class LLMConfigError(Exception):
    pass


class LLMCallError(Exception):
    pass


def _require_key(key: str | None, provider: str) -> str:
    if not key:
        raise LLMConfigError(
            f"LLM_PROVIDER is '{provider}' but its API key is not set in the environment."
        )
    return key


def call_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Send `prompt` to the configured provider and return the text response."""
    provider = settings.llm_provider

    try:
        if provider == "groq":
            return _call_groq(prompt, max_tokens, temperature)
        elif provider == "openai":
            return _call_openai(prompt, max_tokens, temperature)
        elif provider == "gemini":
            return _call_gemini(prompt, max_tokens, temperature)
        else:
            raise LLMConfigError(
                f"Unknown LLM_PROVIDER '{provider}'. Use 'groq', 'openai', or 'gemini'."
            )
    except LLMConfigError:
        raise
    except Exception as exc:
        raise LLMCallError(f"Call to provider '{provider}' failed: {exc}") from exc


def _call_groq(prompt: str, max_tokens: int, temperature: float) -> str:
    from groq import Groq

    key = _require_key(settings.groq_api_key, "groq")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    key = _require_key(settings.openai_api_key, "openai")
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, max_tokens: int, temperature: float) -> str:
    import google.generativeai as genai

    key = _require_key(settings.gemini_api_key, "gemini")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        ),
    )
    return response.text.strip()


def current_model_name() -> str:
    provider = settings.llm_provider
    return {
        "groq": settings.groq_model,
        "openai": settings.openai_model,
        "gemini": settings.gemini_model,
    }.get(provider, "unknown")
