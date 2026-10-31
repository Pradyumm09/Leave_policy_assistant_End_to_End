from pathlib import Path

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.vector_stores.faiss import (
    FaissVectorStore,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "llamaindex_store"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# LEAVE POLICY RETRIEVER
# ============================================================

class LeavePolicyRetriever:

    def __init__(self, top_k: int = 5):

        self.top_k = top_k

        print("=" * 60)
        print("Loading LlamaIndex Leave Policy Retriever")
        print("=" * 60)

        # ----------------------------------------------------
        # Check persisted store
        # ----------------------------------------------------

        if not STORE_PATH.exists():

            raise FileNotFoundError(
                f"""
LlamaIndex store not found:

{STORE_PATH}

Please build the index first using:

python -m app.llamaindex.index
"""
            )

        print(
            f"LlamaIndex store found:\n{STORE_PATH}"
        )

        # ----------------------------------------------------
        # Load BGE embedding model
        # ----------------------------------------------------

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device="cpu",
        )

        print(
            "BGE embedding model loaded successfully."
        )

        # ----------------------------------------------------
        # VERY IMPORTANT
        #
        # Tell LlamaIndex explicitly which embedding model
        # must be used.
        #
        # Without this, LlamaIndex may try to use OpenAI.
        # ----------------------------------------------------

        Settings.embed_model = self.embed_model

        # ----------------------------------------------------
        # Load FAISS vector store
        # ----------------------------------------------------

        print("Loading FAISS vector store...")

        vector_store = (
            FaissVectorStore.from_persist_dir(
                str(STORE_PATH)
            )
        )

        # ----------------------------------------------------
        # Create storage context
        # ----------------------------------------------------

        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(STORE_PATH),
            )
        )

        # ----------------------------------------------------
        # Load existing LlamaIndex
        #
        # Explicit embed_model is important.
        # ----------------------------------------------------

        print("Loading LlamaIndex...")

        self.index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        self.retriever = self.index.as_retriever(
            similarity_top_k=self.top_k
        )

        print(
            "LlamaIndex retriever loaded successfully."
        )

        print("=" * 60)


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(self, query: str):

        if not query or not query.strip():
            return []

        print()
        print(f"Retrieving for query: {query}")

        nodes = self.retriever.retrieve(
            query
        )

        results = []

        for node in nodes:

            results.append(
                {
                    "score": node.score,
                    "content": node.get_content(),
                    "metadata": node.metadata,
                }
            )

        print(
            f"Retrieved {len(results)} results."
        )

        return results