from typing import Any


REFUSAL_MESSAGE = (
    "I don't have enough information in the transcript to answer that question."
)


SYSTEM_PROMPT = """You are a Q&A assistant for a documentary transcript. Answer using ONLY the provided transcript excerpts.

Grounding rules:
- Every factual claim must be directly stated in the excerpts or be a faithful paraphrase of them.
- Do not use outside knowledge.
- Do not infer additional outcomes, severity, frequency, intent, or scale.
- Do not extend a list of consequences beyond what the excerpts explicitly state.
- Do not turn a possible risk into an actual event unless the excerpts say it occurred.
- Do not invent facts, names, dates, causes, locations, relationships, or conclusions.
- Keep examples attached to the correct person, product, substance, date, and subject.
- Ignore unrelated details contained in retrieved excerpts.

Question coverage:
- Identify every distinct part of the user's question.
- Answer every part that is supported by the excerpts.
- If several factors are involved, explain each supported factor separately.
- Connect factors only using relationships directly supported by the excerpts.
- A final synthesis sentence must not introduce any new facts or consequences.
- If only part of a multi-part question is supported, answer that part and state that the remaining part is not established by the excerpts.

Refusal:
- If no part of the answer is supported, output this exact sentence and nothing else:
I don't have enough information in the transcript to answer that question.
- Do not paraphrase or reword the refusal sentence. Copy it verbatim.

Style:
- Keep the answer concise, normally 2 to 4 sentences.
- Use plain text only.
- Do not use Markdown or bullet points.
- Never reproduce bracketed excerpt labels.
- Do not include timestamps or time codes in the answer text.

Answer format:
- Begin directly with the answer in plain prose.
- State only supported findings.
- Before returning the answer, remove any claim that cannot be traced directly to the supplied excerpts.
- If entirely unanswerable, output only the exact refusal sentence.
"""


def build_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Format retrieved transcript chunks into a context block."""
    context_parts: list[str] = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        start = str(chunk["start_timestamp"])
        end = str(chunk["end_timestamp"])
        text = str(chunk["text"]).strip()

        context_parts.append(
            f"[Excerpt {index} | {start} - {end}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_messages(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build transcript-grounded chat-completion messages."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    context = build_context(retrieved_chunks)

    user_content = f"""Answer the question using only the transcript excerpts below.

Before writing the final answer:
1. Identify each distinct part of the question.
2. Find explicit evidence for each supported part.
3. Exclude unrelated information.
4. Remove every factual claim that is not directly supported.
5. Do not add a broader consequence merely to strengthen the conclusion.

When combining factors, use only this structure:
- State what the first factor explicitly caused or did.
- State what the second factor explicitly caused or did.
- Explain their relationship without adding a new outcome.

Transcript excerpts:
{context}

Question:
{clean_question}

Return a concise plain-text answer. Write in prose sentences only. Do not use bullet points, dashes, or any Markdown formatting."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]