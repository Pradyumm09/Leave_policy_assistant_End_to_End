import json
from pathlib import Path
from typing import Any

from app.core.logger import logger


def load_knowledge_base(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load the structured leave-policy knowledge base.

    Expected format:
    [
        {
            "id": "...",
            "source": "...",
            "section": "...",
            "type": "...",
            "content": "..."
        }
    ]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {file_path}"
        )

    logger.info(f"Loading knowledge base: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Knowledge base must contain a JSON list of records."
        )

    if not data:
        raise ValueError("Knowledge base is empty.")

    logger.info(f"Loaded {len(data)} records from knowledge base.")

    return data
