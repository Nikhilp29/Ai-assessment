"""
Summarization orchestration: decides whether the document needs map-reduce
(too big for one LLM call) or can be summarized directly, then calls the
configured LLM accordingly.
"""
from app.ingestion import recursive_character_split
from app.llm_factory import call_llm
from app.prompts import build_map_prompt, build_reduce_prompt, build_direct_prompt


def summarize_document(text: str, style: str) -> tuple[str, int, bool]:
    """
    Returns (summary, chunk_count, used_map_reduce).
    """
    chunks = recursive_character_split(text)

    if len(chunks) == 1:
        # Small enough to summarize in a single call — no map-reduce needed.
        prompt = build_direct_prompt(style, chunks[0])
        summary = call_llm(prompt)
        return summary, 1, False

    # MAP step: summarize each chunk independently
    chunk_summaries = [call_llm(build_map_prompt(chunk)) for chunk in chunks]

    # REDUCE step: combine chunk summaries into one final, styled summary
    combined = "\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    reduce_prompt = build_reduce_prompt(style, combined)
    final_summary = call_llm(reduce_prompt)

    return final_summary, len(chunks), True
