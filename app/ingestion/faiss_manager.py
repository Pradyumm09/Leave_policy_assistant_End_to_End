from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.logger import logger


def build_faiss_index(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    vector_store_path: str | Path,
) -> FAISS:
    """
    Create a FAISS vector store from documents and embeddings.
    """

    if not documents:
        raise ValueError(
            "Cannot build FAISS index: no documents provided."
        )

    vector_store_path = Path(vector_store_path)

    vector_store_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        f"Creating FAISS index from {len(documents)} documents..."
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    vector_store.save_local(
        str(vector_store_path)
    )

    logger.info(
        f"FAISS index saved at: {vector_store_path}"
    )

    return vector_store


def load_faiss_index(
    embeddings: HuggingFaceEmbeddings,
    vector_store_path: str | Path,
) -> FAISS:
    """
    Load an existing FAISS index.
    """

    vector_store_path = Path(vector_store_path)

    if not vector_store_path.exists():
        raise FileNotFoundError(
            f"FAISS vector store not found: {vector_store_path}"
        )

    logger.info(
        f"Loading FAISS index from: {vector_store_path}"
    )

    vector_store = FAISS.load_local(
        str(vector_store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("FAISS index loaded successfully.")

    return vector_store
