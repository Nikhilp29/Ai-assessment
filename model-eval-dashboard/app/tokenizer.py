"""
Approximate token counting, provider-agnostic.

Each provider uses a different real tokenizer (tiktoken for OpenAI, Groq's
Llama tokenizer, Gemini's own), and pulling in all three just for a cost
ESTIMATE is unnecessary weight for what this dashboard needs. The common
rule-of-thumb heuristic -- roughly 4 characters per token for English text
-- is accurate to within ~10-20% for typical prose, which is good enough
for comparing relative cost across providers. Labeled "estimated" in the
UI/API for that reason.
"""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
