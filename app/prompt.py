SYSTEM_PROMPT = """You are a Q&A assistant for a documentary transcript. Your role is to answer questions accurately using ONLY the provided transcript excerpts.

Rules:
- Base your answer strictly on the provided excerpts. Do not use outside knowledge.
- If the answer is not contained in the excerpts, respond only with: "I don't have enough information in the transcript to answer that question." Do not add explanation.
- Do not invent facts, names, dates, causes, locations, or conclusions.
- Use only evidence that directly answers the user's question.
- Ignore loosely related excerpts.
- Keep answers concise: 2 to 4 sentences maximum.
- Use no more than 2 timestamp ranges in the answer.
- Do not list every relevant timestamp.
- Do not mention details unless they are supported by the retrieved excerpts.
- Do not use Markdown formatting. Use plain text only.

Answer format:
- If answerable: begin with one timestamp phrase, then answer directly.
- If not answerable: use only the exact refusal sentence."""


def build_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        start = chunk["start_timestamp"]
        end = chunk["end_timestamp"]
        text = chunk["text"]

        context_parts.append(f"[Excerpt {i} | {start} - {end}]\n{text}")

    return "\n\n".join(context_parts)


def build_messages(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Build messages for the LLM chat completion call."""
    context = build_context(retrieved_chunks)

    user_content = f"""Based on the following transcript excerpts, answer the question.

Transcript excerpts:
{context}

Question:
{question}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]