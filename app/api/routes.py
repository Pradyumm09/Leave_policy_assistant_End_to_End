import os

from fastapi import APIRouter, HTTPException

from app.core.config import BASE_DIR, CONFIG
from app.core.logger import logger

from app.guardrails.input_guard import validate_question

from app.ingestion.embedder import create_embedding_model
from app.ingestion.faiss_manager import load_faiss_index

from app.llm.gateway import LLMGateway

from app.models.request_models import QueryRequest
from app.models.response_models import QueryResponse

from app.services.query_service import QueryService


router = APIRouter()


# ============================================================
# Initialize RAG Components
# ============================================================

logger.info(
    "Initializing RAG components..."
)


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = create_embedding_model()


# ============================================================
# 2. FAISS VECTOR STORE
# ============================================================

vector_store_path = (
    BASE_DIR
    / CONFIG["paths"]["vector_store"]
)

vector_store = load_faiss_index(
    embeddings,
    vector_store_path,
)


# ============================================================
# 3. LLM GATEWAY
# ============================================================

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

llm_gateway = LLMGateway(
    groq_api_key=groq_api_key,
    groq_model=CONFIG["llm"]["model"],
    temperature=CONFIG["llm"]["temperature"],
    ollama_model=CONFIG["llm"]["ollama_model"],
)


# ============================================================
# 4. AGENTIC RAG COMPONENTS
# ============================================================

policy_agent = None
policy_retriever = None


try:

    logger.info(
        "Initializing Agentic RAG components..."
    )

    # --------------------------------------------------------
    # CrewAI Policy Agent
    # --------------------------------------------------------

    from app.agents.policy_agent import (
        policy_agent_function
    )

    policy_agent = policy_agent_function

    # --------------------------------------------------------
    # LlamaIndex Retriever
    # --------------------------------------------------------

    from app.llamaindex.retriever import (
        LeavePolicyRetriever
    )

    policy_retriever = LeavePolicyRetriever(
        top_k=CONFIG["retrieval"]["top_k"]
    )

    logger.info(
        "Agentic RAG initialized successfully."
    )


except Exception:

    logger.exception(
        "Agentic RAG initialization failed. "
        "Legacy RAG will be used."
    )


# ============================================================
# 5. QUERY SERVICE
# ============================================================

query_service = QueryService(
    vector_store=vector_store,
    llm_gateway=llm_gateway,
    top_k=CONFIG["retrieval"]["top_k"],
    confidence_threshold=CONFIG["retrieval"]["confidence_threshold"],
    policy_agent=policy_agent,
    policy_retriever=policy_retriever,
)


# ============================================================
# 6. QUERY ENDPOINT
# ============================================================

@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query_policy(
    request: QueryRequest
):

    # --------------------------------------------------------
    # Input Guardrail
    # --------------------------------------------------------

    is_valid, validation_result = (
        validate_question(
            request.question
        )
    )

    # --------------------------------------------------------
    # Out-of-scope question
    # --------------------------------------------------------

    if not is_valid:

        return {
            "answer": validation_result,
            "sources": [],
            "confidence": 1.0,
            "llm_provider": None,
            "intent": "OUT_OF_SCOPE",
        }

    # --------------------------------------------------------
    # HR / Personal action request
    # --------------------------------------------------------

    if validation_result == "HR_ACTION":

        from app.services.hr_service import (
            build_hr_guidance
        )

        return build_hr_guidance(
            request.question
        )

    # --------------------------------------------------------
    # RAG Query
    # --------------------------------------------------------

    try:

        result = await query_service.query(
            request.question
        )

        return result

    except Exception as exc:

        logger.exception(
            "Error while processing query."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc