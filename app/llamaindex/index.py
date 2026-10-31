from pathlib import Path
import json

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base.json"
)

STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "llamaindex_store"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE-small-en-v1.5 produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384


# ============================================================
# BUILD LLAMAINDEX
# ============================================================

def build_index():

    print("=" * 60)
    print("Building LlamaIndex Leave Policy Index")
    print("=" * 60)

    # --------------------------------------------------------
    # Check knowledge base
    # --------------------------------------------------------

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found:\n{KNOWLEDGE_BASE_PATH}"
        )

    print(f"Knowledge base: {KNOWLEDGE_BASE_PATH}")

    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        knowledge_base = json.load(f)

    # --------------------------------------------------------
    # Convert JSON records to LlamaIndex Documents
    # --------------------------------------------------------

    documents = []

    # Expected structure:
    # [
    #   {
    #       "document_id": "...",
    #       "source": "...",
    #       "section": "...",
    #       "content": "..."
    #   }
    # ]

    if isinstance(knowledge_base, list):

        records = knowledge_base

    elif isinstance(knowledge_base, dict):

        # Handle common wrapper structures
        if "documents" in knowledge_base:
            records = knowledge_base["documents"]

        elif "data" in knowledge_base:
            records = knowledge_base["data"]

        elif "knowledge_base" in knowledge_base:
            records = knowledge_base["knowledge_base"]

        else:
            records = [knowledge_base]

    else:
        raise ValueError(
            "Unsupported knowledge_base.json format."
        )

    print(f"Records found: {len(records)}")

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            continue

        # ----------------------------------------------------
        # Extract content
        # ----------------------------------------------------

        content = (
            record.get("content")
            or record.get("text")
            or record.get("page_content")
            or record.get("description")
            or ""
        )

        if not content:
            continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "document_id": record.get(
                "document_id",
                record.get("id", f"DOC_{index:04d}")
            ),
            "source": record.get(
                "source",
                "HCLTech Leave Policy"
            ),
            "section": record.get(
                "section",
                ""
            ),
            "type": record.get(
                "type",
                "leave_policy"
            ),
        }

        # Add other useful fields if present
        for field in [
            "title",
            "category",
            "leave_type",
            "eligibility",
        ]:
            if field in record:
                metadata[field] = record[field]

        document = Document(
            text=str(content),
            metadata=metadata,
        )

        documents.append(document)

    if not documents:
        raise ValueError(
            "No valid documents were created from knowledge_base.json."
        )

    print(f"LlamaIndex documents created: {len(documents)}")

    # --------------------------------------------------------
    # Load BGE embedding model
    # --------------------------------------------------------

    print(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    print("BGE embedding model loaded successfully.")

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Explicitly configure LlamaIndex to use BGE.
    # This prevents LlamaIndex from falling back to OpenAI.
    # --------------------------------------------------------

    Settings.embed_model = embed_model

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    print(
        f"Creating FAISS index with dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    faiss_index = faiss.IndexFlatL2(
        EMBEDDING_DIMENSION
    )

    vector_store = FaissVectorStore(
        faiss_index=faiss_index
    )

    # --------------------------------------------------------
    # Storage context
    # --------------------------------------------------------

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # --------------------------------------------------------
    # Build LlamaIndex
    # --------------------------------------------------------

    print("Building LlamaIndex...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # --------------------------------------------------------
    # Create storage directory
    # --------------------------------------------------------

    STORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Persist index
    # --------------------------------------------------------

    index.storage_context.persist(
        persist_dir=str(STORE_PATH)
    )

    print()
    print("=" * 60)
    print("LlamaIndex built successfully!")
    print("=" * 60)
    print(f"Documents : {len(documents)}")
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"Dimension : {EMBEDDING_DIMENSION}")
    print(f"Store     : {STORE_PATH}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_index()