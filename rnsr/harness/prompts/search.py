"""Prompts used by the retrieval ladder's model-assisted stages."""


def render_expansion(query: str, tried: set[str], context: str) -> str:
    return (
        f"Search query: {query}\n"
        f"Terms already tried: {', '.join(sorted(tried))}\n"
        f"Nearby text from the corpus:\n{context}\n\n"
        "Propose up to 5 NEW search terms (synonyms, abbreviations, "
        "formatting variants) likely to find the answer in this "
        "corpus. Return one term per line, nothing else."
    )


def render_sweep(query: str, numbered: str) -> str:
    return (
        f"Question: {query}\n\nChunks:\n{numbered}\n\n"
        "List the chunk ids (the numbers in brackets) that contain "
        "information answering the question, one per line. If none "
        "do, reply NONE."
    )
