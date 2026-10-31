from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import CONFIG
from app.core.logger import logger


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the local BGE-small embedding model.

    CPU-only configuration is intentional because the target
    machine does not have a dedicated GPU.
    """

    model_name = CONFIG["embedding"]["model"]
    device = CONFIG["embedding"]["device"]

    logger.info(
        f"Loading embedding model: {model_name}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    logger.info("Embedding model loaded successfully.")

    return embeddings
