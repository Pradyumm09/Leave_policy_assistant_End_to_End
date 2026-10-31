from langchain_core.documents import Document


SYSTEM_INSTRUCTION = """
You are the Leave Policy Assistant.

Your job is to answer employee questions using ONLY the
provided leave-policy context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume policy rules.
3. If the answer is not present in the context, clearly say:
   "I could not find this information in the available leave policy."
4. Give a concise and direct answer.
5. When possible, mention the relevant policy section.
"""


def build_prompt(
    question: str,
    documents: list[Document],
) -> str:

    context_parts = []

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata

        context_parts.append(
            f"""
--- CONTEXT {index} ---
Document ID: {metadata.get("id")}
Section: {metadata.get("section")}
Type: {metadata.get("type")}

{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    return f"""
{SYSTEM_INSTRUCTION}

Retrieved policy context:

{context}

Employee Question:
{question}

Answer:
"""
