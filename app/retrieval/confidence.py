from langchain_community.vectorstores import FAISS


def retrieve_with_confidence(
    vector_store: FAISS,
    question: str,
    top_k: int = 5,
    threshold: float = 0.45,
):
    """
    Retrieve documents with LangChain's normalized relevance scores.

    Higher score = more relevant.
    """

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=top_k,
    )

    if not results:
        return [], 0.0

    documents = [
        document
        for document, score in results
        if score >= threshold
    ]

    best_score = max(
        score
        for _, score in results
    )

    return documents, float(best_score)