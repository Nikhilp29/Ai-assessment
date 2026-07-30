def build_rag_prompt(question: str, context_chunks: list[str], history: list[tuple[str, str]]) -> str:
    """
    context_chunks: retrieved document excerpts, most relevant first
    history: list of (role, content) tuples for prior turns in this session
    """
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"

    history_block = ""
    if history:
        turns = "\n".join(f"{role.capitalize()}: {content}" for role, content in history)
        history_block = f"\nPrior conversation:\n{turns}\n"

    return (
        "You are a helpful assistant answering questions using ONLY the provided "
        "document context below. If the answer isn't in the context, say you "
        "don't have enough information -- do not make things up.\n\n"
        f"CONTEXT:\n{context_block}\n"
        f"{history_block}\n"
        f"Question: {question}\n\nAnswer:"
    )
