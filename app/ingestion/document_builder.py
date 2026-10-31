from typing import Any

from langchain_core.documents import Document

from app.core.logger import logger


def build_documents(
    records: list[dict[str, Any]]
) -> list[Document]:
    """
    Convert structured JSON records into LangChain Documents.

    FAQ records:
        Question + Answer are combined because both are
        important for semantic retrieval.

    Paragraph/Table records:
        Content is used directly.
    """

    documents = []

    for record in records:

        record_id = record.get("id", "")
        source = record.get("source", "Unknown")
        section = record.get("section", "Unknown")
        record_type = record.get("type", "unknown")

        # -----------------------------------------
        # Build searchable text
        # -----------------------------------------

        if record_type == "faq":

            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()

            page_content = (
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

        else:

            content = record.get("content", "").strip()

            if not content:
                continue

            page_content = content

        # -----------------------------------------
        # Metadata
        # -----------------------------------------

        metadata = {
            "id": record_id,
            "source": source,
            "section": section,
            "type": record_type,
        }

        document = Document(
            page_content=page_content,
            metadata=metadata,
        )

        documents.append(document)

    logger.info(
        f"Created {len(documents)} LangChain documents."
    )

    return documents
