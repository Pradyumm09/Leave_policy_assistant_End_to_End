import asyncio
from app.core.logger import logger
from app.retrieval.confidence import retrieve_with_confidence
from app.retrieval.prompt_builder import build_prompt


class QueryService:

    def __init__(
        self,
        vector_store,
        llm_gateway,
        top_k: int = 5,
        confidence_threshold: float = 0.45,
        policy_agent=None,
        policy_retriever=None,
    ):
        """
        Query Service supports:

        1. Agentic RAG
           CrewAI -> Leave Policy Search Tool -> LlamaIndex

        2. Legacy RAG fallback
           FAISS -> Prompt Builder -> LLM Gateway
        """

        self.vector_store = vector_store
        self.llm_gateway = llm_gateway
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        # Agentic RAG components
        self.policy_agent = policy_agent
        self.policy_retriever = policy_retriever

    # ============================================================
    # AGENTIC RAG SOURCE METADATA
    # ============================================================

    def _get_agentic_sources(self, question: str):

        sources = []
        scores = []

        if self.policy_retriever is None:
            return sources, 0.0

        try:

            retrieved = self.policy_retriever.retrieve(question)

            for result in retrieved:

                metadata = {}
                score = None

                # ------------------------------------------------
                # Dictionary-style result
                # ------------------------------------------------

                if isinstance(result, dict):

                    metadata = result.get("metadata", {}) or {}

                    score = result.get("score")

                    if score is None:
                        score = metadata.get("score")

                # ------------------------------------------------
                # Object-style result
                # ------------------------------------------------

                else:

                    metadata = getattr(
                        result,
                        "metadata",
                        {},
                    ) or {}

                    score = getattr(
                        result,
                        "score",
                        None,
                    )

                    # LlamaIndex NodeWithScore
                    if score is None:

                        node = getattr(
                            result,
                            "node",
                            None,
                        )

                        if node is not None:

                            metadata = getattr(
                                node,
                                "metadata",
                                {},
                            ) or {}

                            score = getattr(
                                result,
                                "score",
                                None,
                            )

                # ------------------------------------------------
                # Source metadata
                # ------------------------------------------------

                source_id = (
                    metadata.get("document_id")
                    or metadata.get("id")
                    or metadata.get("document")
                )

                section = metadata.get("section")

                source_type = metadata.get("type")

                sources.append(
                    {
                        "id": source_id,
                        "section": section,
                        "type": source_type,
                    }
                )

                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                if score is not None:

                    try:

                        score = float(score)

                        score = max(
                            0.0,
                            min(1.0, score),
                        )

                        scores.append(score)

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            # ----------------------------------------------------
            # Remove duplicate sources
            # ----------------------------------------------------

            unique_sources = []
            seen = set()

            for source in sources:

                key = (
                    source.get("id"),
                    source.get("section"),
                    source.get("type"),
                )

                if key not in seen:

                    seen.add(key)
                    unique_sources.append(source)

            sources = unique_sources

            # ----------------------------------------------------
            # Confidence
            # ----------------------------------------------------

            confidence = max(scores) if scores else 0.0

            return sources, round(confidence, 3)

        except Exception:

            logger.exception(
                "Unable to retrieve Agentic RAG source metadata."
            )

            return [], 0.0

    # ============================================================
    # MAIN QUERY
    # ============================================================

    async def query(self, question: str) -> dict:

        logger.info(
            f"Processing question: {question}"
        )

        # ========================================================
        # 1. PRIMARY PATH - AGENTIC RAG
        # ========================================================

        if self.policy_agent is not None:

            logger.info(
                "Using Agentic RAG: CrewAI + LlamaIndex"
            )

            try:

                # ------------------------------------------------
                # IMPORTANT:
                # policy_agent is ASYNC.
                # Therefore directly await it.
                # ------------------------------------------------

                agent_result = await asyncio.to_thread(
                    self.policy_agent,
                    question
                )

                # CrewAI may return CrewOutput/object
                answer = str(agent_result).strip()

                if not answer:

                    raise RuntimeError(
                        "Agent returned an empty response."
                    )

                logger.info(
                    "Agentic RAG response generated successfully."
                )

                # ------------------------------------------------
                # Get source metadata
                # ------------------------------------------------

                sources, confidence = (
                    self._get_agentic_sources(
                        question
                    )
                )

                logger.info(
                    f"Agentic retrieval confidence: "
                    f"{confidence:.3f}"
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "llm_provider": "groq",
                    "intent": "POLICY",
                    "retrieval_method": "agentic_rag",
                }

            except Exception:

                logger.exception(
                    "Agentic RAG failed. "
                    "Falling back to legacy RAG."
                )

        # ========================================================
        # 2. LEGACY RAG FALLBACK
        # ========================================================

        logger.info(
            "Using legacy RAG pipeline."
        )

        try:

            documents, confidence = (
                retrieve_with_confidence(
                    vector_store=self.vector_store,
                    question=question,
                    top_k=self.top_k,
                    threshold=self.confidence_threshold,
                )
            )

        except Exception:

            logger.exception(
                "Legacy retrieval failed."
            )

            return {
                "answer": (
                    "I was unable to search the "
                    "leave policy at the moment."
                ),
                "sources": [],
                "confidence": 0.0,
                "llm_provider": None,
                "intent": "POLICY",
                "retrieval_method": "legacy_rag",
            }

        # ========================================================
        # Normalize confidence
        # ========================================================

        try:

            confidence = float(
                confidence or 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        logger.info(
            f"Legacy retrieval confidence: "
            f"{confidence:.3f}"
        )

        # ========================================================
        # 3. LOW-CONFIDENCE FALLBACK
        # ========================================================

        if not documents:

            logger.warning(
                "No sufficiently relevant policy "
                "context found."
            )

            return {
                "answer": (
                    "I could not find sufficiently "
                    "relevant information in the "
                    "available leave policy."
                ),
                "sources": [],
                "confidence": round(
                    confidence,
                    3,
                ),
                "llm_provider": None,
                "intent": "POLICY",
                "retrieval_method": "legacy_rag",
            }

        # ========================================================
        # 4. BUILD GROUNDED PROMPT
        # ========================================================

        prompt = build_prompt(
            question=question,
            documents=documents,
        )

        # ========================================================
        # 5. GENERATE ANSWER
        # ========================================================

        try:

            answer, provider = (
                self.llm_gateway.generate(
                    prompt
                )
            )

        except Exception:

            logger.exception(
                "All LLM providers failed."
            )

            return {
                "answer": (
                    "I was unable to generate "
                    "an answer at the moment. "
                    "Please try again."
                ),
                "sources": [],
                "confidence": round(
                    confidence,
                    3,
                ),
                "llm_provider": None,
                "intent": "POLICY",
                "retrieval_method": "legacy_rag",
            }

        # ========================================================
        # 6. PREPARE SOURCES
        # ========================================================

        sources = []

        for document in documents:

            metadata = (
                getattr(
                    document,
                    "metadata",
                    {},
                ) or {}
            )

            sources.append(
                {
                    "id": metadata.get("id"),
                    "section": metadata.get("section"),
                    "type": metadata.get("type"),
                }
            )

        # ========================================================
        # 7. FINAL LEGACY RESPONSE
        # ========================================================

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(
                confidence,
                3,
            ),
            "llm_provider": provider,
            "intent": "POLICY",
            "retrieval_method": "legacy_rag",
        }

    # ============================================================
    # BACKWARD COMPATIBILITY
    # ============================================================

    async def query_async(self, question: str) -> dict:

        return await self.query(question)