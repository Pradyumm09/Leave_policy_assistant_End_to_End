from app.core.config import BASE_DIR, CONFIG
from app.core.logger import logger
from app.ingestion.document_builder import build_documents
from app.ingestion.embedder import create_embedding_model
from app.ingestion.faiss_manager import build_faiss_index
from app.ingestion.json_loader import load_knowledge_base


def main():

    logger.info("=" * 60)
    logger.info("LEAVE POLICY RAG - INDEX BUILDING")
    logger.info("=" * 60)

    # -----------------------------------------
    # 1. Load JSON
    # -----------------------------------------

    knowledge_base_path = (
        BASE_DIR
        / CONFIG["paths"]["knowledge_base"]
    )

    records = load_knowledge_base(
        knowledge_base_path
    )

    # -----------------------------------------
    # 2. Convert to LangChain Documents
    # -----------------------------------------

    documents = build_documents(records)

    # -----------------------------------------
    # 3. Load Embedding Model
    # -----------------------------------------

    embeddings = create_embedding_model()

    # -----------------------------------------
    # 4. Build FAISS
    # -----------------------------------------

    vector_store_path = (
        BASE_DIR
        / CONFIG["paths"]["vector_store"]
    )

    vector_store = build_faiss_index(
        documents=documents,
        embeddings=embeddings,
        vector_store_path=vector_store_path,
    )

    # -----------------------------------------
    # 5. Summary
    # -----------------------------------------

    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETED")
    logger.info(f"Records loaded : {len(records)}")
    logger.info(f"Documents      : {len(documents)}")
    logger.info(
        f"Vector count   : {vector_store.index.ntotal}"
    )
    logger.info(
        f"Vector store   : {vector_store_path}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
