# granite_client/prompts.py

SYSTEM_DEFAULT = (
    "You are Granite, a reliable assistant. "
    "Answer in the same language as the user unless asked otherwise. "
    "Prioritize a direct useful answer. "
    "Use only grounded context and do not invent facts."
)

SYSTEM_REASONING = (
    "You are Granite, an analytical reasoning assistant. "
    "When given a question and context, reason step-by-step. "
    "Answer in the same language as the user unless asked otherwise. "
    "Structure your thoughts clearly before giving a final answer. "
    "Do not invent facts not present in the provided context."
)

SYSTEM_RELATION = (
    "You are an information extraction assistant. "
    "Extract structured relations from text. "
    "Return ONLY valid JSON. No explanation, no markdown."
)

SYSTEM_CITATION = (
    "You are a citation-aware assistant. "
    "When using information from context, cite the source using [N] notation "
    "where N is the source number. "
    "Do not invent citations."
)


def build_prompt(
    user_message: str,
    system: str = SYSTEM_DEFAULT,
    context: str | None = None,
) -> str:
    """
    ساخت پرامپت کامل برای Granite Instruct.
    فرمت: SYSTEM → CONTEXT (اختیاری) → USER → ASSISTANT
    """
    parts: list[str] = [f"SYSTEM: {system}"]

    if context:
        parts.append(f"\nCONTEXT:\n{context}")

    parts.append(f"\nUSER: {user_message}")
    parts.append("\nASSISTANT:")

    return "\n".join(parts)
