"""
Unlike L1-07/L2-01's llm_factory (one configured provider, chosen via
LLM_PROVIDER), this dashboard needs to call MULTIPLE providers for the same
prompt to compare them -- so call_provider() takes an explicit provider
name per call instead of reading one fixed setting.
"""
from app import config

MODEL_NAMES = {
    "groq": config.GROQ_MODEL,
    "openai": config.OPENAI_MODEL,
    "gemini": config.GEMINI_MODEL,
}


class LLMConfigError(Exception):
    pass


class LLMCallError(Exception):
    pass


def available_providers() -> list[str]:
    """Providers that have an API key configured -- these are the ones the
    dashboard can actually benchmark."""
    available = []
    if config.GROQ_API_KEY:
        available.append("groq")
    if config.OPENAI_API_KEY:
        available.append("openai")
    if config.GEMINI_API_KEY:
        available.append("gemini")
    return available


def call_provider(provider: str, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    try:
        if provider == "groq":
            return _call_groq(prompt, max_tokens, temperature)
        elif provider == "openai":
            return _call_openai(prompt, max_tokens, temperature)
        elif provider == "gemini":
            return _call_gemini(prompt, max_tokens, temperature)
        else:
            raise LLMConfigError(f"Unknown provider '{provider}'. Use 'groq', 'openai', or 'gemini'.")
    except LLMConfigError:
        raise
    except Exception as exc:
        raise LLMCallError(f"Call to provider '{provider}' failed: {exc}") from exc


def _call_groq(prompt: str, max_tokens: int, temperature: float) -> str:
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise LLMConfigError("GROQ_API_KEY is not set.")
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise LLMConfigError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, max_tokens: int, temperature: float) -> str:
    import google.generativeai as genai

    if not config.GEMINI_API_KEY:
        raise LLMConfigError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        ),
    )
    return response.text.strip()
