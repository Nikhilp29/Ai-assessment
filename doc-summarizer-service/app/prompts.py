"""
Prompt templates per summary style.

Each style has two prompts:
  - MAP prompt: summarizes a single chunk
  - REDUCE prompt: combines chunk summaries (or, for single-chunk docs,
    summarizes the whole document directly) into the final styled output.
"""

STYLE_INSTRUCTIONS = {
    "brief": (
        "Write a brief summary in 3-5 sentences. Capture only the most "
        "important point(s). Do not use bullet points."
    ),
    "detailed": (
        "Write a detailed, well-structured summary covering all major points, "
        "supporting details, and any conclusions. Use short paragraphs."
    ),
    "bullet-points": (
        "Summarize as a concise bulleted list. Each bullet should capture one "
        "distinct point. Use '- ' for bullets. No preamble text."
    ),
}


def build_map_prompt(chunk_text: str) -> str:
    """Prompt used to summarize a single chunk (style-agnostic — we keep the
    map step neutral and apply style only at the reduce/final step, which
    keeps intermediate summaries reusable across styles)."""
    return (
        "Summarize the following section of a document. Preserve key facts, "
        "names, numbers, and conclusions. Be concise but do not omit "
        "important information.\n\n"
        f"SECTION:\n{chunk_text}\n\nSECTION SUMMARY:"
    )


def build_reduce_prompt(style: str, combined_summaries: str) -> str:
    """Prompt used to combine chunk summaries into the final styled summary."""
    instruction = STYLE_INSTRUCTIONS[style]
    return (
        "You are combining several section summaries of one document into a "
        f"single final summary.\n\nStyle instructions: {instruction}\n\n"
        f"SECTION SUMMARIES:\n{combined_summaries}\n\nFINAL SUMMARY:"
    )


def build_direct_prompt(style: str, document_text: str) -> str:
    """Prompt used when the whole document fits in one chunk — no map-reduce
    needed, summarize directly in the requested style."""
    instruction = STYLE_INSTRUCTIONS[style]
    return (
        f"Summarize the following document.\n\nStyle instructions: {instruction}\n\n"
        f"DOCUMENT:\n{document_text}\n\nSUMMARY:"
    )
