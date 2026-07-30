"""
Approximate per-1K-token pricing, used only to produce a rough cost
ESTIMATE for comparison purposes -- not a billing-accurate figure.

IMPORTANT: these numbers reflect list pricing as of this project's
knowledge cutoff (early 2026) and WILL drift out of date. Before relying on
this for real budget decisions, check each provider's current pricing page:
  - Groq:   https://groq.com/pricing
  - OpenAI: https://openai.com/api/pricing
  - Gemini: https://ai.google.dev/pricing
This is a deliberately simple, swappable table -- update the numbers below
whenever pricing changes, or swap in a live-fetched pricing source if you
want the dashboard to always be current.
"""
from app.config import GROQ_MODEL, OPENAI_MODEL, GEMINI_MODEL

# All figures are USD per 1,000 tokens.
PRICING = {
    "groq": {
        "model": GROQ_MODEL,
        "input_per_1k": 0.00005,
        "output_per_1k": 0.00008,
    },
    "openai": {
        "model": OPENAI_MODEL,
        "input_per_1k": 0.00015,
        "output_per_1k": 0.00060,
    },
    "gemini": {
        "model": GEMINI_MODEL,
        "input_per_1k": 0.000075,
        "output_per_1k": 0.00030,
    },
}


def estimate_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(provider)
    if not rates:
        return 0.0
    cost = (input_tokens / 1000) * rates["input_per_1k"] + (output_tokens / 1000) * rates["output_per_1k"]
    return round(cost, 6)
